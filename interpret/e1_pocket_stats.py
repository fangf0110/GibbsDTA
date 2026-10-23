"""
E1b - Does model attribution concentrate on the experimentally observed binding pocket?

For every curated complex (labels from e1_struct_labels.py) and every relevant
checkpoint, residue- and atom-level importance scores are compared against the
crystal-structure labels.

Model tags per case
    s1_f{k}        setting-1 model of fold k (k = 0..4); `split` column says whether
                   the pair was in that fold's test or training set
    s1_test        alias of the setting-1 fold in which the pair is a TEST pair (primary)
    s3_test        setting-3 (cold-target) fold in which the pair is a TEST pair
    abl_{variant}_f{k}   ablation checkpoints (setting 1) when available

Importance types
    ig_main        |Integrated Gradients| of y_main (PRIMARY — faithful to full network)
    ig_phy         |Integrated Gradients| of y_phy = H - T*S
    maxpool        number of max-pool channels won by the residue / atom
    hybrid         0.5 * z(ig_main) + 0.5 * z(maxpool)
    attn           raw ligand->protein cross-attention (diagnostic)

Outputs (interpret/outputs/e1/)
    case_metrics.csv          one row per case x tag x importance (residue level)
    atom_metrics.csv          one row per case x tag x importance (ligand-atom level)
    dmff_zone_table.csv/.txt  DMFF-DTA Fig. 3a-style Binding Site / Range / Outside table
    ligand_element_stats.csv  Ligand atom importance by chemical element, group & contact
    residue_scores/{case}_{tag}.csv, atom_scores/{case}_{tag}.csv
    attention/{case}/{tag}.npz
    fold_consistency.csv      Spearman between fold models' residue importance
    group_stats.txt / group_stats.csv
    fig_auroc_by_tag.png, fig_zone_bars.png, fig_dmff_zones.png
"""
import os
import argparse
import itertools

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from interpret import config as C  # noqa: E402
from interpret import common as U  # noqa: E402
from interpret import case_infer as CI  # noqa: E402
from interpret.e1_struct_labels import load_cases, usable_cases  # noqa: E402

ZONES = ["site", "range", "outside"]
ZONE_LABELS = {"site": "Binding Site (≤5 Å)", "range": "Binding Range (5–8 Å)", "outside": "Outside (>8 Å)"}


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def safe_auc(y, s):
    y = np.asarray(y).astype(int)
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def p_to_star(p):
    if np.isnan(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def circular_perm_p(y, s, observed, rng, n_perm):
    """Null by circular shifts of the score track along the sequence (keeps autocorrelation)."""
    n = len(s)
    if np.isnan(observed) or n < 10 or n_perm <= 0:
        return np.nan
    shifts = rng.integers(1, n, size=n_perm)
    null = np.array([safe_auc(y, np.roll(s, k)) for k in shifts])
    return float((1 + np.sum(null >= observed)) / (1 + n_perm))


def residue_metrics(scores, res_df, rng, n_perm):
    lab = res_df[res_df.esm_covered].sort_values("davis_idx")
    idx = lab.davis_idx.values.astype(int)
    s = scores[idx]
    y_site = (lab.label == "site").values
    y_near = lab.label.isin(["site", "range"]).values
    z_all = (scores - scores.mean()) / (scores.std() + 1e-12)
    z = z_all[idx]
    norm_rank = stats.rankdata(s) / max(1, len(s))
    norm_minmax = (s - s.min()) / (s.max() - s.min() + 1e-12)
    auc_site = safe_auc(y_site, s)
    order = np.argsort(-s)
    k10 = min(10, len(s))
    k5 = max(1, int(round(0.05 * len(s))))
    base = y_site.mean() if len(y_site) else np.nan
    # whole-sequence view: unresolved residues count as non-site
    y_full = np.zeros(len(scores), dtype=bool)
    y_full[idx[y_site]] = True
    order_full = np.argsort(-scores)
    out = {
        "n_mapped": len(lab), "n_site": int(y_site.sum()), "n_near": int(y_near.sum()),
        "auroc_site": auc_site,
        "auroc_near": safe_auc(y_near, s),
        "auroc_site_fullseq": safe_auc(y_full, scores),
        "precision_top10": float(y_site[order[:k10]].mean()) if k10 else np.nan,
        "ef_top5pct": float(y_site[order[:k5]].mean() / base) if base and base > 0 else np.nan,
        "hit_top10_fullseq": int(y_full[order_full[:10]].sum()),
        "perm_p_site": circular_perm_p(y_site, s, auc_site, rng, n_perm),
    }
    for zname in ZONES:
        m = (lab.label == zname).values
        out[f"z_{zname}"] = float(z[m].mean()) if m.any() else np.nan
        out[f"rank_{zname}_mean"] = float(norm_rank[m].mean()) if m.any() else np.nan
        out[f"rank_{zname}_sd"] = float(norm_rank[m].std()) if m.sum() > 1 else np.nan
        out[f"minmax_{zname}_mean"] = float(norm_minmax[m].mean()) if m.any() else np.nan
        out[f"minmax_{zname}_sd"] = float(norm_minmax[m].std()) if m.sum() > 1 else np.nan
    for a, b in [("site", "outside"), ("range", "outside"), ("site", "range")]:
        va = norm_rank[(lab.label == a).values]
        vb = norm_rank[(lab.label == b).values]
        if len(va) >= 2 and len(vb) >= 2:
            p_t = float(stats.ttest_ind(va, vb, equal_var=False, alternative="greater").pvalue)
            p_mw = float(stats.mannwhitneyu(va, vb, alternative="greater").pvalue)
        else:
            p_t, p_mw = np.nan, np.nan
        out[f"p_ttest_{a}_gt_{b}"] = p_t
        out[f"p_mwu_{a}_gt_{b}"] = p_mw
    df_tab = pd.DataFrame({
        "davis_idx": idx, "score": s, "norm_rank": norm_rank, "norm_minmax": norm_minmax,
        "z": z, "label": lab.label.values, "min_dist": lab.min_dist.values,
        "resseq": lab.resseq.values, "aa": lab.aa_davis.values,
    })
    return out, df_tab


def atom_metrics(scores, lig_df):
    d = lig_df.dropna(subset=["contact"])
    if scores is None or d.empty:
        return {"auroc_contact": np.nan, "n_contact": 0, "n_atoms_mapped": len(d), "spearman_neg_dist": np.nan}
    idx = d.atom_idx.values.astype(int)
    y = d.contact.astype(bool).values
    return {"auroc_contact": safe_auc(y, scores[idx]), "n_contact": int(y.sum()),
            "n_atoms_mapped": len(d),
            "spearman_neg_dist": float(stats.spearmanr(scores[idx], -d.min_dist.values)[0])}


# ----------------------------------------------------------------------------
# model registry for one case
# ----------------------------------------------------------------------------
def model_specs(case):
    ds = case.get("dataset", "davis")
    specs = []
    for f in C.FOLDS:
        ck = C.ckpt_path(ds, 1, f)
        if os.path.exists(ck):
            split = U.pair_split(ds, 1, f, case["drug_id"], case["protein_id"])
            specs.append({"tag": f"s1_f{f}", "ckpt": ck, "cross": True, "split": split, "setting": 1, "fold": f})
    for f in C.FOLDS:
        if U.pair_split(ds, 3, f, case["drug_id"], case["protein_id"]) == "test":
            ck = C.ckpt_path(ds, 3, f)
            if os.path.exists(ck):
                specs.append({"tag": "s3_test", "ckpt": ck, "cross": True, "split": "test", "setting": 3, "fold": f})
    for variant in C.RESULTS_ABLATION.get(ds, {}):
        for f in C.FOLDS:
            ck = C.ablation_ckpt_path(ds, variant, f)
            if ck and os.path.exists(ck):
                split = U.pair_split(ds, 1, f, case["drug_id"], case["protein_id"])
                specs.append({"tag": f"abl_{variant}_f{f}", "ckpt": ck, "cross": C.ABLATION_USES_CROSS[variant],
                              "split": split, "setting": 1, "fold": f})
    return specs


_MODEL_CACHE = {}


def get_model(ckpt, cross, device):
    key = (ckpt, cross)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = U.load_model(ckpt, use_cross_attention=cross, device=device)
    return _MODEL_CACHE[key]


# ----------------------------------------------------------------------------
# DMFF-DTA Fig. 3a-style table & group statistics
# ----------------------------------------------------------------------------
def wilcoxon_greater(x, mu=0.5):
    x = np.asarray([v for v in x if not np.isnan(v)])
    if len(x) < 3:
        return np.nan
    return float(stats.wilcoxon(x - mu, alternative="greater").pvalue)


def build_dmff_table(cm, pooled_z, cases_by_id, primary_imp="ig_main"):
    """Build a table matching DMFF-DTA Fig. 3a (Binding Site / Binding Range / Outside mean ± SD + sig)."""
    sub = cm[(cm.tag == "s1_test") & (cm.importance == primary_imp)].copy()
    if sub.empty:
        return pd.DataFrame(), ""
    rows = []
    for _, r in sub.iterrows():
        cid = r.case_id
        c = cases_by_id.get(cid, {})
        p_so = r.get("p_ttest_site_gt_outside", np.nan)
        p_ro = r.get("p_ttest_range_gt_outside", np.nan)
        p_sr = r.get("p_ttest_site_gt_range", np.nan)
        rows.append({
            "case_id": cid,
            "pdb_id": c.get("pdb_id", cid.split("_")[0]),
            "drug_name": c.get("drug_name", c.get("drug_id", "")),
            "protein_gene": c.get("protein_gene", c.get("protein_id", "")),
            "tier": c.get("tier", "core"),
            "main_figure": bool(c.get("main_figure", False)),
            "binding_site_mean_sd": f"{r.rank_site_mean:.3f} ± {r.rank_site_sd:.3f}",
            "binding_range_mean_sd": f"{r.rank_range_mean:.3f} ± {r.rank_range_sd:.3f}",
            "outside_mean_sd": f"{r.rank_outside_mean:.3f} ± {r.rank_outside_sd:.3f}",
            "site_mean": r.rank_site_mean, "site_sd": r.rank_site_sd,
            "range_mean": r.rank_range_mean, "range_sd": r.rank_range_sd,
            "outside_mean": r.rank_outside_mean, "outside_sd": r.rank_outside_sd,
            "p_site_vs_outside": p_so, "sig_site_vs_outside": p_to_star(p_so),
            "p_range_vs_outside": p_ro, "sig_range_vs_outside": p_to_star(p_ro),
            "p_site_vs_range": p_sr, "sig_site_vs_range": p_to_star(p_sr),
            "auroc_site": r.auroc_site, "ef_top5pct": r.ef_top5pct, "precision_top10": r.precision_top10,
        })
    # Add pooled summary row across all cases in pooled_z
    if not pooled_z.empty:
        vs = pooled_z[pooled_z.label == "site"].norm_rank
        vr = pooled_z[pooled_z.label == "range"].norm_rank
        vo = pooled_z[pooled_z.label == "outside"].norm_rank
        p_so = float(stats.ttest_ind(vs, vo, equal_var=False, alternative="greater").pvalue) if len(vs) > 1 and len(vo) > 1 else np.nan
        p_ro = float(stats.ttest_ind(vr, vo, equal_var=False, alternative="greater").pvalue) if len(vr) > 1 and len(vo) > 1 else np.nan
        p_sr = float(stats.ttest_ind(vs, vr, equal_var=False, alternative="greater").pvalue) if len(vs) > 1 and len(vr) > 1 else np.nan
        rows.append({
            "case_id": "POOLED_ALL", "pdb_id": "ALL", "drug_name": f"All ({sub.case_id.nunique()} complexes)",
            "protein_gene": "All", "tier": "all", "main_figure": True,
            "binding_site_mean_sd": f"{vs.mean():.3f} ± {vs.std():.3f}",
            "binding_range_mean_sd": f"{vr.mean():.3f} ± {vr.std():.3f}",
            "outside_mean_sd": f"{vo.mean():.3f} ± {vo.std():.3f}",
            "site_mean": float(vs.mean()), "site_sd": float(vs.std()),
            "range_mean": float(vr.mean()), "range_sd": float(vr.std()),
            "outside_mean": float(vo.mean()), "outside_sd": float(vo.std()),
            "p_site_vs_outside": p_so, "sig_site_vs_outside": p_to_star(p_so),
            "p_range_vs_outside": p_ro, "sig_range_vs_outside": p_to_star(p_ro),
            "p_site_vs_range": p_sr, "sig_site_vs_range": p_to_star(p_sr),
            "auroc_site": float(sub.auroc_site.median()),
            "ef_top5pct": float(sub.ef_top5pct.mean()),
            "precision_top10": float(sub.precision_top10.mean()),
        })
    df = pd.DataFrame(rows)
    disp_cols = ["pdb_id", "drug_name", "protein_gene", "binding_site_mean_sd",
                 "binding_range_mean_sd", "outside_mean_sd", "sig_site_vs_outside",
                 "sig_range_vs_outside", "auroc_site", "ef_top5pct"]
    mf = df[df.main_figure]
    txt = "=== DMFF-DTA Fig. 3a Style Table (Main-Figure Cases + Pooled Benchmark, importance=" + primary_imp + ") ===\n"
    txt += mf[disp_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    return df, txt


def group_report(cm, am, pooled_z, dmff_txt, primary_imp="ig_main"):
    lines, rows = [], []
    for (tag, imp), g in cm.groupby(["tag", "importance"]):
        aucs = g.auroc_site.dropna()
        r = {"tag": tag, "importance": imp, "n_cases": len(aucs),
             "auroc_site_median": aucs.median(), "auroc_site_mean": aucs.mean(),
             "auroc_site_q1": aucs.quantile(0.25), "auroc_site_q3": aucs.quantile(0.75),
             "wilcoxon_p_vs_0.5": wilcoxon_greater(aucs),
             "frac_cases_perm_p<0.05": float((g.perm_p_site < 0.05).mean()),
             "precision_top10_mean": g.precision_top10.mean(), "ef_top5pct_mean": g.ef_top5pct.mean()}
        for a, b in [("site", "outside"), ("range", "outside"), ("site", "range")]:
            x, y = g[f"z_{a}"].values, g[f"z_{b}"].values
            ok = ~(np.isnan(x) | np.isnan(y))
            p = stats.wilcoxon(x[ok], y[ok], alternative="greater").pvalue if ok.sum() >= 3 else np.nan
            r[f"wilcoxon_{a}_vs_{b}_bonf"] = min(1.0, 3 * p) if not np.isnan(p) else np.nan
        rows.append(r)
    gs = pd.DataFrame(rows)

    lines.append("=== Residue level: binding-site recovery (AUROC site vs non-site, resolved residues) ===")
    lines.append(gs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if dmff_txt:
        lines.append("\n" + dmff_txt)

    lines.append(f"\n=== Pooled residue importance by zone (tag=s1_test, importance={primary_imp}) ===")
    pz = pooled_z
    if not pz.empty:
        for zname in ZONES:
            v_z = pz[pz.label == zname].z
            v_r = pz[pz.label == zname].norm_rank
            lines.append(f"{zname:8s} n={len(v_z):5d}  z={v_z.mean():+.4f}±{v_z.std():.4f}  "
                         f"norm_rank={v_r.mean():.4f}±{v_r.std():.4f}")
        groups = [pz[pz.label == z].norm_rank.values for z in ZONES]
        if all(len(g) > 1 for g in groups):
            lines.append(f"Kruskal-Wallis p={stats.kruskal(*groups).pvalue:.3e}")
            for a, b in [("site", "outside"), ("range", "outside"), ("site", "range")]:
                p = stats.mannwhitneyu(pz[pz.label == a].norm_rank, pz[pz.label == b].norm_rank,
                                       alternative="greater").pvalue
                lines.append(f"  Mann-Whitney {a} > {b}: p_bonf={min(1, 3 * p):.3e} ({p_to_star(min(1, 3 * p))})")

    # full vs ablation (paired over cases, same fold)
    lines.append("\n=== Full model vs ablations (paired by case, same fold) ===")
    for tag in sorted(cm.tag.unique()):
        if not tag.startswith("abl_"):
            continue
        fold = tag.split("_f")[-1]
        for imp in [primary_imp, "maxpool"]:
            a = cm[(cm.tag == f"s1_f{fold}") & (cm.importance == imp)].set_index("case_id").auroc_site
            b = cm[(cm.tag == tag) & (cm.importance == imp)].set_index("case_id").auroc_site
            j = a.index.intersection(b.index)
            j = [k for k in j if not (np.isnan(a[k]) or np.isnan(b[k]))]
            if len(j) >= 3:
                p = stats.wilcoxon(a[j], b[j]).pvalue if not np.allclose(a[j], b[j]) else 1.0
                lines.append(f"{imp:8s} full(s1_f{fold}) median={a[j].median():.3f} vs {tag} "
                             f"median={b[j].median():.3f}  n={len(j)} Wilcoxon p={p:.3e}")

    lines.append("\n=== Ligand atoms: contact (<=4 Å) recovery ===")
    if not am.empty:
        ag = am.groupby(["tag", "importance"]).auroc_contact.agg(["count", "median", "mean"]).reset_index()
        ag["wilcoxon_p_vs_0.5"] = [wilcoxon_greater(am[(am.tag == t) & (am.importance == i)].auroc_contact)
                                   for t, i in zip(ag.tag, ag.importance)]
        lines.append(ag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return "\n".join(lines), gs


def plots(cm, pooled_z, dmff_df, out_dir, primary_imp="ig_main"):
    imps = [primary_imp, "ig_phy", "maxpool", "hybrid"]
    sel = cm[cm.importance.isin(imps)]
    tags = [t for t in ["s1_test", "s3_test"] + sorted(t for t in sel.tag.unique() if t.startswith("abl_"))
            if t in set(sel.tag)]
    fig, ax = plt.subplots(figsize=(1.6 * max(3, len(tags)) + 2.5, 4.2))
    data, labels = [], []
    for t in tags:
        for imp in imps:
            v = sel[(sel.tag == t) & (sel.importance == imp)].auroc_site.dropna().values
            if len(v):
                data.append(v)
                labels.append(f"{t}\n{imp}")
    if data:
        ax.boxplot(data, showfliers=False)
        for i, v in enumerate(data, 1):
            ax.scatter(np.random.normal(i, 0.05, len(v)), v, s=12, alpha=0.65, color="#2c3e50")
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        ax.axhline(0.5, ls="--", c="grey")
        ax.set_ylabel("AUROC (binding-site residues)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_auroc_by_tag.png"), dpi=300)
    plt.close(fig)

    if not pooled_z.empty:
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        m = [pooled_z[pooled_z.label == z].norm_rank.mean() for z in ZONES]
        se = [pooled_z[pooled_z.label == z].norm_rank.std() / np.sqrt(max(1, (pooled_z.label == z).sum())) for z in ZONES]
        ax.bar([ZONE_LABELS[z] for z in ZONES], m, yerr=se, color=["#c0392b", "#e69f00", "#7f8c8d"], capsize=3)
        ax.axhline(0.5, c="k", ls="--", lw=0.8)
        ax.set_ylim(0.35, max(0.75, max(m) + 0.08))
        ax.set_ylabel("Normalized residue attribution (mean ± s.e.)")
        ax.tick_params(axis="x", labelsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_zone_bars.png"), dpi=300)
        plt.close(fig)

    # DMFF-DTA Fig. 3a style grouped bar chart for main-figure cases + pooled
    if dmff_df is not None and not dmff_df.empty:
        mf = dmff_df[dmff_df.main_figure].copy()
        if not mf.empty:
            fig, ax = plt.subplots(figsize=(max(5.5, 1.35 * len(mf)), 3.6))
            x = np.arange(len(mf))
            w = 0.25
            colors = [("#c0392b", "site"), ("#e69f00", "range"), ("#7f8c8d", "outside")]
            for k, (col, z) in enumerate(colors):
                vals = mf[f"{z}_mean"].values
                ax.bar(x + (k - 1) * w, vals, width=w, color=col, label=ZONE_LABELS[z], alpha=0.9)
            for i, (_, r) in enumerate(mf.iterrows()):
                top = max(r.site_mean, r.range_mean, r.outside_mean) + 0.02
                sig = r.sig_site_vs_outside
                ax.text(i, top, sig, ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="#c0392b" if sig != "ns" else "grey")
            xlabels = [f"{r.pdb_id}\n({r.drug_name[:10]}–{r.protein_gene})" if r.pdb_id != "ALL"
                       else "Pooled\nBenchmark" for _, r in mf.iterrows()]
            ax.set_xticks(x, xlabels, fontsize=7.5)
            ax.axhline(0.5, ls="--", color="grey", lw=0.8)
            ax.set_ylim(0.30, 0.82)
            ax.set_ylabel("Normalized Attribution Score")
            ax.set_title("Spatial Distribution of Residue Attribution Across Structural Zones", fontsize=9.5)
            ax.legend(fontsize=7.5, ncol=3, loc="upper right", frameon=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_dmff_zones.png"), dpi=300)
            plt.close(fig)


# ----------------------------------------------------------------------------
def annotate_ligand_chemistry(smiles, n_heavy):
    mol = Chem.MolFromSmiles(smiles)
    elems, aroms, groups = [], [], []
    for i in range(n_heavy):
        if mol is not None and i < mol.GetNumAtoms():
            a = mol.GetAtomWithIdx(i)
            el = a.GetSymbol()
            ar = bool(a.GetIsAromatic())
        else:
            el, ar = "?", False
        elems.append(el)
        aroms.append(ar)
        if el != "C":
            groups.append("Heteroatom (N/O/S/Cl/F)")
        elif ar:
            groups.append("Aromatic C")
        else:
            groups.append("Aliphatic C")
    return elems, aroms, groups


def summarize_ligand_elements(atom_tables, out_dir):
    if not atom_tables:
        return
    df = pd.concat(atom_tables, ignore_index=True)
    rows = []
    for col in ["elem", "chem_group", "contact"]:
        if col not in df.columns:
            continue
        for val, g in df.dropna(subset=[col]).groupby(col):
            rows.append({
                "group_by": col, "category": str(val), "n_atoms": len(g),
                "n_cases": g.case_id.nunique(),
                "ig_rank_mean": float(g.ig_rank.mean()), "ig_rank_sd": float(g.ig_rank.std()),
                "ig_main_mean": float(g.ig_main.mean()),
                "maxpool_mean": float(g.maxpool.mean()),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "ligand_element_stats.csv"), index=False)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--primary_imp", default="ig_main", choices=["ig_main", "ig_phy", "maxpool", "hybrid", "attn"])
    ap.add_argument("--ig_steps_primary", type=int, default=32)
    ap.add_argument("--ig_steps_aux", type=int, default=16)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--no_npz", action="store_true")
    args = ap.parse_args()

    device = C.device()
    rng = np.random.default_rng(C.SEED)
    out_dir = C.out("e1")
    lab_dir = os.path.join(out_dir, "labels")
    meta = pd.read_csv(os.path.join(lab_dir, "cases_meta.csv"))
    ok_cases = usable_cases(meta)
    all_cases = load_cases(args.cases)
    cases_by_id = {c["case_id"]: c for c in all_cases}

    cm_rows, am_rows, cons_rows, pooled, s1_atom_tables = [], [], [], [], []
    for case in all_cases:
        cid = case["case_id"]
        if cid not in ok_cases:
            print(f"[skip] {cid}: no labels")
            continue
        ds = case.get("dataset", "davis")
        store = U.FeatureStore(ds)
        smiles = U.load_meta(ds)["ligands"][str(case["drug_id"])]
        res_df = pd.read_csv(os.path.join(lab_dir, f"{cid}_residues.csv"))
        lig_df = pd.read_csv(os.path.join(lab_dir, f"{cid}_ligand_atoms.csv"))
        site_idx = res_df[(res_df.label == "site") & res_df.esm_covered].davis_idx.astype(int).tolist()

        specs = model_specs(case)
        test_fold = [s for s in specs if s["tag"].startswith("s1_f") and s["split"] == "test"]
        if test_fold:
            specs.append(dict(test_fold[0], tag="s1_test"))
        fold_vecs = {}
        for sp in specs:
            model = get_model(sp["ckpt"], sp["cross"], device)
            is_primary_tag = sp["tag"] in ("s1_test", "s3_test")
            ig_steps = args.ig_steps_primary if is_primary_tag else args.ig_steps_aux
            ig_targets = ("y_main", "H", "S") if is_primary_tag else ("y_main",)
            o = CI.case_forward(model, store, ds, case["drug_id"], case["protein_id"], device,
                                ig_steps=ig_steps, ig_targets=ig_targets)
            if not args.no_npz and not sp["tag"].startswith("s1_test"):
                CI.save_npz(os.path.join(out_dir, "attention", cid, f"{sp['tag']}.npz"), o)
            base = {"case_id": cid, "tag": sp["tag"], "split": sp["split"], "setting": sp["setting"],
                    "fold": sp["fold"], "y_true": case.get("y"), "y_main": o["y_main"], "H": o["H"],
                    "S": o["S"], "y_phy": o["y_phy"], "extra_token_mass": CI.extra_token_mass(o),
                    "ig_completeness_main": o["ig_completeness"].get("y_main", np.nan)}

            kinds = ["ig_main", "maxpool", "hybrid"]
            if "y_phy" in o["ig_res"]:
                kinds += ["ig_phy", "ig_H", "ig_S"]
            if o["a_l2p"] is not None:
                kinds.append("attn")

            primary_table = None
            for kind in kinds:
                sc = CI.residue_scores(o, kind)
                if sc is None:
                    continue
                n_p = args.n_perm if (is_primary_tag and kind in (args.primary_imp, "ig_main", "maxpool", "hybrid")) else 0
                m, table = residue_metrics(sc, res_df, rng, n_p)
                cm_rows.append({**base, "importance": kind, **m})
                if kind == args.primary_imp:
                    primary_table = table
                if kind == args.primary_imp and sp["tag"] == "s1_test":
                    pooled.append(table.assign(case_id=cid))
                if kind == args.primary_imp and sp["tag"].startswith("s1_f"):
                    fold_vecs[sp["tag"]] = (sc, sp["split"])

            if is_primary_tag and primary_table is not None:
                # Attach all residue tracks for e2_figures.py
                idx = primary_table.davis_idx.values.astype(int)
                for k_extra in ("ig_main", "ig_phy", "maxpool", "hybrid", "attn"):
                    v = CI.residue_scores(o, k_extra)
                    if v is not None:
                        primary_table[k_extra] = v[idx]
                for t_sig in ("y_main", "y_phy", "H", "S"):
                    if t_sig in o["ig_res"]:
                        primary_table[f"ig_{t_sig}_signed"] = o["ig_res"][t_sig][idx]
                primary_table.to_csv(C.out("e1", "residue_scores", f"{cid}_{sp['tag']}.csv"), index=False)

            # ligand atoms
            elems, aroms, groups = annotate_ligand_chemistry(smiles, o["n_heavy"])
            atom_tab = {
                "atom_idx": np.arange(o["n_heavy"]),
                "elem": elems, "is_aromatic": aroms, "chem_group": groups,
            }
            atom_kinds = [("ig_main", None), ("maxpool", None), ("hybrid", None)]
            if "y_phy" in o["ig_atom_abs"]:
                atom_kinds += [("ig_phy", None), ("ig_H", None), ("ig_S", None)]
            if o["a_p2l"] is not None:
                atom_kinds += [("attn_all", None), ("attn_site", site_idx)]
            for kind, q in atom_kinds:
                sc = CI.atom_scores(o, kind, query_residues=q)
                atom_tab[kind] = sc if sc is not None else np.full(o["n_heavy"], np.nan)
                am_rows.append({**base, "importance": kind, **atom_metrics(sc, lig_df)})
            if "ig_main" in atom_tab:
                atom_tab["ig_rank"] = stats.rankdata(atom_tab["ig_main"]) / max(1, o["n_heavy"])
            for t_sig in ("y_main", "y_phy", "H", "S"):
                if t_sig in o["ig_atom"]:
                    atom_tab[f"ig_{t_sig}_signed"] = o["ig_atom"][t_sig]
            if is_primary_tag:
                at = pd.DataFrame(atom_tab).merge(lig_df, on="atom_idx", how="left")
                at.to_csv(C.out("e1", "atom_scores", f"{cid}_{sp['tag']}.csv"), index=False)
                if sp["tag"] == "s1_test":
                    s1_atom_tables.append(at.assign(case_id=cid))

        for (ta, (va, sa)), (tb, (vb, sb)) in itertools.combinations(sorted(fold_vecs.items()), 2):
            cons_rows.append({"case_id": cid, "a": ta, "b": tb, "split_a": sa, "split_b": sb,
                              "spearman": float(stats.spearmanr(va, vb)[0])})
        r = [x for x in cm_rows if x["case_id"] == cid and x["tag"] == "s1_test" and x["importance"] == args.primary_imp]
        if r:
            print(f"[{cid}] s1_test ({args.primary_imp}) AUROC={r[0]['auroc_site']:.3f} "
                  f"p={r[0]['perm_p_site']:.3g} Site>Out={p_to_star(r[0]['p_ttest_site_gt_outside'])} "
                  f"top10={r[0]['precision_top10']:.2f} y={case.get('y')} pred={r[0]['y_main']:.2f}")

    cm, am = pd.DataFrame(cm_rows), pd.DataFrame(am_rows)
    cm.to_csv(os.path.join(out_dir, "case_metrics.csv"), index=False)
    am.to_csv(os.path.join(out_dir, "atom_metrics.csv"), index=False)
    pd.DataFrame(cons_rows).to_csv(os.path.join(out_dir, "fold_consistency.csv"), index=False)
    summarize_ligand_elements(s1_atom_tables, out_dir)
    pooled_z = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame(columns=["label", "z", "norm_rank"])
    dmff_df, dmff_txt = build_dmff_table(cm, pooled_z, cases_by_id, args.primary_imp)
    if not dmff_df.empty:
        dmff_df.to_csv(os.path.join(out_dir, "dmff_zone_table.csv"), index=False)
        with open(os.path.join(out_dir, "dmff_zone_table.txt"), "w") as f:
            f.write(dmff_txt + "\n")
    text, gs = group_report(cm, am, pooled_z, dmff_txt, args.primary_imp)
    if cons_rows:
        cr = pd.DataFrame(cons_rows)
        text += f"\n\n=== Cross-fold consistency of residue attribution ({args.primary_imp}, Spearman) ===\n"
        text += f"mean={cr.spearman.mean():.3f} median={cr.spearman.median():.3f} n_pairs={len(cr)}"
    gs.to_csv(os.path.join(out_dir, "group_stats.csv"), index=False)
    with open(os.path.join(out_dir, "group_stats.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    plots(cm, pooled_z, dmff_df, out_dir, args.primary_imp)


if __name__ == "__main__":
    main()
