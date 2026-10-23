"""
E0 - Re-inference of every trained checkpoint on its own test fold.

For each dataset / setting / fold this script
  * rebuilds the ordered test set (drug_id, protein_id, y) from the fold file,
  * checks that y_true matches the stored test_predictions.csv row by row,
  * runs the instrumented forward pass and stores y_main, H, S, y_phy and
    per-sample attention-entropy summaries,
  * recomputes MSE / CI and compares them with the stored metrics.txt,
  * also runs the available ablation checkpoints (setting 1).

Outputs (under interpret/outputs/e0/)
    predictions/{dataset}_s{setting}_f{fold}.csv
    predictions_ablation/{dataset}_{variant}_f{fold}.csv
    summary.csv
    sanity_report.txt         <- read this first; the script exits 1 on failure

Usage
    python -m interpret.e0_infer                       # everything
    python -m interpret.e0_infer --datasets davis --settings 1 --folds 0
"""
import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import torch

from interpret import config as C
from interpret import common as U


def residue_importance(a_l2p, smi_mask, prot_mask):
    """a_l2p (B,h,N_L,N_P) -> (B,N_P): head-mean, then mean over valid ligand query tokens."""
    a = a_l2p.mean(dim=1)                                   # (B, N_L, N_P)
    valid_l = (~smi_mask).float().unsqueeze(-1)             # (B, N_L, 1)
    imp = (a * valid_l).sum(1) / valid_l.sum(1).clamp(min=1)
    return imp.masked_fill(prot_mask, 0.0)


def atom_importance(a_p2l, smi_mask, prot_mask):
    """a_p2l (B,h,N_P,N_L) -> (B,N_L): head-mean, then mean over valid protein query tokens."""
    a = a_p2l.mean(dim=1)
    valid_p = (~prot_mask).float().unsqueeze(-1)
    imp = (a * valid_p).sum(1) / valid_p.sum(1).clamp(min=1)
    return imp.masked_fill(smi_mask, 0.0)


def normalized_entropy(p, mask):
    """Entropy of a distribution over valid tokens, divided by log(n_valid)."""
    p = p.masked_fill(mask, 0.0)
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
    ent = -(p * torch.log(p.clamp(min=1e-12))).sum(-1)
    n = (~mask).sum(-1).clamp(min=2).float()
    return ent / torch.log(n)


@torch.no_grad()
def run_split(model, store, meta, samples, device, batch_size):
    lig_keys, prot_keys = meta["lig_keys"], meta["prot_keys"]
    rows = []
    first_check = None
    for b in range(0, len(samples), batch_size):
        chunk = samples[b:b + batch_size]
        smi_list = [store.ligand(lig_keys[r]) for r, _, _ in chunk]
        prot_list = [store.protein(prot_keys[c]) for _, c, _ in chunk]
        smi, prot, smi_mask, prot_mask = (t.to(device) for t in U.collate(smi_list, prot_list))
        if first_check is None:
            first_check = U.check_forward_equivalence(model, smi, prot, smi_mask, prot_mask)
        o = U.forward_full(model, smi, prot, smi_mask, prot_mask, per_head=True)
        if o["a_l2p"] is not None:
            r_imp = residue_importance(o["a_l2p"], smi_mask, prot_mask)
            a_imp = atom_importance(o["a_p2l"], smi_mask, prot_mask)
            ent_p = normalized_entropy(r_imp, prot_mask).cpu().numpy()
            ent_l = normalized_entropy(a_imp, smi_mask).cpu().numpy()
        else:
            ent_p = ent_l = np.full(len(chunk), np.nan)
        y_main, H, S, y_phy = (o[k].cpu().numpy() for k in ("y_main", "H", "S", "y_phy"))
        for i, (r, c, y) in enumerate(chunk):
            rows.append({
                "drug_id": lig_keys[r], "protein_id": prot_keys[c], "y_true": y,
                "y_main": float(y_main[i]), "H": float(H[i]), "S": float(S[i]), "y_phy": float(y_phy[i]),
                "attn_entropy_prot": float(ent_p[i]), "attn_entropy_lig": float(ent_l[i]),
            })
    return pd.DataFrame(rows), first_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=C.DATASETS)
    ap.add_argument("--settings", nargs="+", type=int, default=C.SETTINGS)
    ap.add_argument("--folds", nargs="+", type=int, default=C.FOLDS)
    ap.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--skip_ablation", action="store_true")
    ap.add_argument("--mse_tol", type=float, default=2e-3,
                    help="Allowed |MSE_recomputed - MSE_metrics.txt| (metrics.txt has 4 decimals)")
    args = ap.parse_args()

    device = C.device()
    torch.manual_seed(C.SEED)
    pred_dir = C.out("e0", "predictions")
    abl_dir = C.out("e0", "predictions_ablation")
    report_lines, summary, failures = [], [], []

    def log(msg):
        print(msg, flush=True)
        report_lines.append(msg)

    log(f"E0 re-inference | device={device} | root={C.PROJECT_ROOT}")
    for ds in args.datasets:
        meta = U.load_meta(ds)
        store = U.FeatureStore(ds)
        for s in args.settings:
            for f in args.folds:
                ck = C.ckpt_path(ds, s, f)
                if not os.path.exists(ck):
                    log(f"[MISSING] {ds} s{s} f{f}: {ck}")
                    failures.append(f"{ds} s{s} f{f} missing checkpoint")
                    continue
                t0 = time.time()
                samples = U.split_samples(ds, s, f, "test")
                ref = pd.read_csv(C.test_pred_path(ds, s, f))
                tag = f"{ds} s{s} f{f}"

                # 1) test-set order check
                order_ok = len(ref) == len(samples) and np.allclose(
                    ref["y_true"].values, [y for _, _, y in samples], atol=1e-4)
                if not order_ok:
                    log(f"[FAIL] {tag}: test order mismatch (n_csv={len(ref)}, n_rebuilt={len(samples)})")
                    failures.append(f"{tag} order")
                    continue

                model = U.load_model(ck, use_cross_attention=True, device=device)
                df, (eq_ok, eq_diff) = run_split(model, store, meta, samples, device, args.batch_size)
                df.insert(0, "fold", f)
                df.insert(0, "setting", s)
                df.insert(0, "dataset", ds)
                df["y_pred_saved"] = ref["y_pred"].values
                df.to_csv(os.path.join(pred_dir, f"{ds}_s{s}_f{f}.csv"), index=False)

                # 2) metric reproduction
                ref_m = U.read_metrics_txt(C.test_metrics_path(ds, s, f))
                m_main = U.mse(df.y_true, df.y_main)
                ci_main = U.concordance_index(df.y_true.values, df.y_main.values)
                m_phy = U.mse(df.y_true, df.y_phy)
                ci_phy = U.concordance_index(df.y_true.values, df.y_phy.values)
                max_dev = float(np.abs(df.y_main - df.y_pred_saved).max())
                ref_mse = ref_m.get("Test MSE", np.nan)
                mse_ok = np.isnan(ref_mse) or abs(m_main - ref_mse) <= args.mse_tol
                status = "OK" if (eq_ok and mse_ok) else ("WARN" if eq_ok else "FAIL")
                if status == "FAIL":
                    failures.append(f"{tag} eq_ok={eq_ok} mse_ok={mse_ok}")
                log(f"[{status}] {tag}: n={len(df)} MSE={m_main:.4f} (saved {ref_mse:.4f}) "
                    f"CI={ci_main:.4f} (saved {ref_m.get('Test CI', np.nan):.4f}) "
                    f"max|y_main-y_saved|={max_dev:.2e} fwd_equiv={eq_ok}({eq_diff:.1e}) "
                    f"| y_phy MSE={m_phy:.4f} CI={ci_phy:.4f} | {time.time() - t0:.1f}s")
                summary.append({
                    "dataset": ds, "setting": s, "fold": f, "variant": "full", "n": len(df),
                    "mse_main": m_main, "ci_main": ci_main, "mse_phy": m_phy, "ci_phy": ci_phy,
                    "mse_saved": ref_mse, "ci_saved": ref_m.get("Test CI", np.nan),
                    "pearson_main_phy": float(np.corrcoef(df.y_main, df.y_phy)[0, 1]),
                    "pearson_H_S": float(np.corrcoef(df.H, df.S)[0, 1]),
                    "max_dev_saved": max_dev, "fwd_equiv": eq_ok,
                    "attn_entropy_prot": float(df.attn_entropy_prot.mean()),
                    "attn_entropy_lig": float(df.attn_entropy_lig.mean()),
                })
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

        # ---- ablation checkpoints (setting 1) ----
        if args.skip_ablation or 1 not in args.settings:
            continue
        for variant in C.RESULTS_ABLATION.get(ds, {}):
            for f in args.folds:
                ck = C.ablation_ckpt_path(ds, variant, f)
                if ck is None or not os.path.exists(ck):
                    continue
                samples = U.split_samples(ds, 1, f, "test")
                model = U.load_model(ck, use_cross_attention=C.ABLATION_USES_CROSS[variant], device=device)
                df, _ = run_split(model, store, meta, samples, device, args.batch_size)
                df.insert(0, "fold", f)
                df.insert(0, "variant", variant)
                df.insert(0, "dataset", ds)
                df.to_csv(os.path.join(abl_dir, f"{ds}_{variant}_f{f}.csv"), index=False)
                m = U.mse(df.y_true, df.y_main)
                ci = U.concordance_index(df.y_true.values, df.y_main.values)
                log(f"[ABL ] {ds} {variant} f{f}: n={len(df)} MSE={m:.4f} CI={ci:.4f}")
                summary.append({"dataset": ds, "setting": 1, "fold": f, "variant": variant,
                                "n": len(df), "mse_main": m, "ci_main": ci,
                                "attn_entropy_prot": float(df.attn_entropy_prot.mean()),
                                "attn_entropy_lig": float(df.attn_entropy_lig.mean())})
                del model

    pd.DataFrame(summary).to_csv(C.out("e0", "summary.csv"), index=False)
    log("")
    log("RESULT: " + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
    with open(C.out("e0", "sanity_report.txt"), "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()

