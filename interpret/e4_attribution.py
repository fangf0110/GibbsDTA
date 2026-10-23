"""
E4 - Gradient-based attribution (Integrated Gradients) as an attention-independent
check, and branch-level attribution for y_main, H, S and y_phy.

For each curated complex and model tag (s1_test, s3_test):
  * IG with an all-zero feature baseline (masks fixed to "all valid") and a
    midpoint Riemann sum, separately for y_main, H and S (y_phy = H - T*S by linearity),
  * token attributions = sum over feature dimension of (x - x0) * mean gradient,
  * completeness check: sum(attributions) vs f(x) - f(x0),
  * residue |IG| -> binding-site AUROC (same protocol as E1); atom |IG| -> contact AUROC,
  * agreement with attention: Spearman and top-20 overlap,
  * branch agreement: Spearman(|IG y_main|, |IG y_phy|), Spearman(IG_H, IG_S) (signed).

Outputs (interpret/outputs/e4/)
    ig_metrics.csv, ig_summary.txt
    ig_residue_scores/{case}_{tag}.csv, ig_atom_scores/{case}_{tag}.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

from interpret import config as C
from interpret import common as U
from interpret import case_infer as CI
from interpret.e1_struct_labels import load_cases, usable_cases
from interpret.e1_pocket_stats import residue_metrics, atom_metrics, model_specs, get_model, wilcoxon_greater

from interpret.case_infer import integrated_gradients, fold_atoms

TARGETS = ("y_main", "H", "S")


def topk_overlap(a, b, k=20):
    k = min(k, len(a))
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--tags", nargs="+", default=["s1_test", "s3_test"])
    args = ap.parse_args()

    device = C.device()
    rng = np.random.default_rng(C.SEED)
    out_dir = C.out("e4")
    lab_dir = os.path.join(C.OUT_DIR, "e1", "labels")
    meta = pd.read_csv(os.path.join(lab_dir, "cases_meta.csv"))
    ok_cases = usable_cases(meta)
    rows = []

    for case in load_cases(args.cases):
        cid = case["case_id"]
        if cid not in ok_cases:
            continue
        ds = case.get("dataset", "davis")
        store = U.FeatureStore(ds)
        res_df = pd.read_csv(os.path.join(lab_dir, f"{cid}_residues.csv"))
        lig_df = pd.read_csv(os.path.join(lab_dir, f"{cid}_ligand_atoms.csv"))
        specs = model_specs(case)
        tf = [s for s in specs if s["tag"].startswith("s1_f") and s["split"] == "test"]
        if tf:
            specs.append(dict(tf[0], tag="s1_test"))
        specs = [s for s in specs if s["tag"] in args.tags]

        for sp in specs:
            model = get_model(sp["ckpt"], sp["cross"], device)
            o = CI.case_forward(model, store, ds, case["drug_id"], case["protein_id"], device)
            smi = store.ligand(case["drug_id"]).unsqueeze(0).to(device)
            prot = store.protein(case["protein_id"]).unsqueeze(0).to(device)
            attr, delta = integrated_gradients(model, smi, prot, args.steps, args.chunk)

            n_res = o["n_res"]
            mp_res = CI.residue_scores(o, "maxpool")
            mp_atom = CI.atom_scores(o, "maxpool")
            att_res = CI.residue_scores(o, "attn")
            att_atom = CI.atom_scores(o, "attn")
            res_tab = {"davis_idx": np.arange(n_res)}
            atom_tab = {"atom_idx": np.arange(o["n_heavy"])}
            for t in ("y_main", "H", "S", "y_phy"):
                a_s, a_p = attr[t]
                completeness = (a_s.sum() + a_p.sum() - delta[t]) / (abs(delta[t]) + 1e-8)
                r_signed = a_p[1:1 + n_res]
                r_abs = np.abs(r_signed)
                at_signed = fold_atoms(a_s, o)
                at_abs = fold_atoms(np.abs(a_s), o)
                res_tab[f"ig_{t}"] = r_signed
                atom_tab[f"ig_{t}"] = at_signed
                m, _ = residue_metrics(r_abs, res_df, rng, args.n_perm)
                am = atom_metrics(at_abs, lig_df)
                row = {"case_id": cid, "tag": sp["tag"], "target": t, "split": sp["split"],
                       "completeness_rel_err": float(completeness),
                       "protein_share": float(np.abs(a_p).sum() / (np.abs(a_p).sum() + np.abs(a_s).sum() + 1e-12)),
                       **m, "atom_auroc_contact": am["auroc_contact"]}
                if mp_res is not None:
                    row["spearman_ig_vs_maxpool_res"] = float(stats.spearmanr(r_abs, mp_res)[0])
                    row["top20_overlap_ig_maxpool_res"] = topk_overlap(r_abs, mp_res)
                    row["spearman_ig_vs_maxpool_atom"] = float(stats.spearmanr(at_abs, mp_atom)[0])
                if att_res is not None:
                    row["spearman_ig_vs_attn_res"] = float(stats.spearmanr(r_abs, att_res)[0])
                    row["top20_overlap_ig_attn_res"] = topk_overlap(r_abs, att_res)
                    row["spearman_ig_vs_attn_atom"] = float(stats.spearmanr(at_abs, att_atom)[0])
                rows.append(row)
            # branch agreement
            agree = {
                "case_id": cid, "tag": sp["tag"], "target": "branch_agreement", "split": sp["split"],
                "spearman_absIG_main_vs_phy_res": float(stats.spearmanr(np.abs(res_tab["ig_y_main"]),
                                                                        np.abs(res_tab["ig_y_phy"]))[0]),
                "spearman_IG_H_vs_S_res_signed": float(stats.spearmanr(res_tab["ig_H"], res_tab["ig_S"])[0]),
                "spearman_absIG_main_vs_phy_atom": float(stats.spearmanr(np.abs(atom_tab["ig_y_main"]),
                                                                         np.abs(atom_tab["ig_y_phy"]))[0]),
                "spearman_IG_H_vs_S_atom_signed": float(stats.spearmanr(atom_tab["ig_H"], atom_tab["ig_S"])[0]),
            }
            rows.append(agree)
            pd.DataFrame(res_tab).merge(res_df[["davis_idx", "label", "min_dist", "resseq"]],
                                        on="davis_idx", how="left").assign(maxpool=mp_res, attn=att_res) \
                .to_csv(C.out("e4", "ig_residue_scores", f"{cid}_{sp['tag']}.csv"), index=False)
            pd.DataFrame(atom_tab).merge(lig_df, on="atom_idx", how="left") \
                .to_csv(C.out("e4", "ig_atom_scores", f"{cid}_{sp['tag']}.csv"), index=False)
            print(f"[{cid} {sp['tag']}] IG y_main site AUROC="
                  f"{[r for r in rows if r['case_id'] == cid and r['tag'] == sp['tag'] and r['target'] == 'y_main'][-1]['auroc_site']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "ig_metrics.csv"), index=False)
    lines = ["=== IG: binding-site recovery and agreement with maxpool / attention ==="]
    main_rows = df[df.target != "branch_agreement"]
    for (tag, t), g in main_rows.groupby(["tag", "target"]):
        lines.append(
            f"{tag:8s} {t:7s} n={len(g):3d} AUROC_site median={g.auroc_site.median():.3f} "
            f"(Wilcoxon>0.5 p={wilcoxon_greater(g.auroc_site):.2e}) atom_AUROC median="
            f"{g.atom_auroc_contact.median():.3f} | rho(IG,maxpool) res={g.get('spearman_ig_vs_maxpool_res', pd.Series()).median():.3f} "
            f"top20 overlap={g.get('top20_overlap_ig_maxpool_res', pd.Series()).median():.3f} "
            f"| completeness |err| median={g.completeness_rel_err.abs().median():.4f}")
    ba = df[df.target == "branch_agreement"]
    if not ba.empty:
        lines.append("\n=== Branch agreement (median over cases) ===")
        for tag, g in ba.groupby("tag"):
            lines.append(f"{tag}: rho(|IG main|,|IG phy|) res={g.spearman_absIG_main_vs_phy_res.median():.3f} "
                         f"atom={g.spearman_absIG_main_vs_phy_atom.median():.3f} | rho(IG H, IG S) signed "
                         f"res={g.spearman_IG_H_vs_S_res_signed.median():.3f} atom={g.spearman_IG_H_vs_S_atom_signed.median():.3f}")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "ig_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
