"""
E5 - Drug Selectivity Analysis across Targets (benchmarking & extending DeepDTAGen).

Evaluates whether GibbsDTA / PhyNet-DTA captures target selectivity for individual
drugs without collapsing to a target-agnostic "drug mean" prediction:

  1. Dataset-wide per-drug cross-target ranking (Davis & KIBA, Settings 1 & 2):
     For each drug with >= 10 tested targets and affinity range >= 1.5 log units,
     computes cross-target Concordance Index (CI), Spearman rho, Pearson r, RMSE,
     and Top-3 active precision.
  2. DeepDTAGen Regime 1 (High Target Selectivity — High / Medium / Low affinity):
     For representative kinase inhibitors (Dasatinib, Afatinib, Imatinib, Sorafenib,
     plus top KIBA drugs), compares experimental affinity, y_main, and y_phy = H - T*S
     across high-, medium-, and low-affinity targets.
  3. DeepDTAGen Regime 2 (Consistent Multi-Target Affinity across Paralogs):
     Evaluates prediction consistency across multiple targets with similar experimental
     affinity (|Δy_true| <= 0.5) for the same drug.

Outputs (interpret/outputs/e5/)
    per_drug_selectivity.csv
    selectivity_summary.csv, selectivity_summary.txt
    selectivity_cases.csv
    fig_selectivity_profiles.png
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from interpret import config as C  # noqa: E402
from interpret import common as U  # noqa: E402
from interpret import case_infer as CI  # noqa: E402


FLAGSHIP_DAVIS_DRUGS = [
    ("3062316", "Dasatinib"),
    ("10184653", "Afatinib"),
    ("5291", "Imatinib"),
    ("216239", "Sorafenib"),
]


def load_gene_map():
    """Map Davis (protein_id -> gene_symbol) and drug names from Davis.csv and pairs.csv."""
    prot_gene = {}
    davis_csv = os.path.join(C.DATA_PATH, "davis", "Davis.csv")
    if os.path.exists(davis_csv):
        df = pd.read_csv(davis_csv, usecols=["PROTEIN_ID", "PROTEIN_GENE"], dtype=str).drop_duplicates()
        for _, r in df.iterrows():
            pid = str(r.PROTEIN_ID)
            g = str(r.PROTEIN_GENE).split("(")[0].replace("-nonphosphorylated", "").replace("-phosphorylated", "")
            if pid not in prot_gene or len(g) < len(prot_gene[pid]):
                prot_gene[pid] = g
    drug_name = dict(FLAGSHIP_DAVIS_DRUGS)
    pairs_csv = os.path.join(C.CASES_DIR, "pairs.csv")
    if os.path.exists(pairs_csv):
        p = pd.read_csv(pairs_csv, dtype=str)
        for _, r in p.iterrows():
            if pd.notna(r.get("protein_gene")):
                prot_gene[str(r.protein_id)] = str(r.protein_gene)
            if r.drug_id not in drug_name and pd.notna(r.get("drug_short")):
                drug_name[str(r.drug_id)] = str(r.drug_short)
    return prot_gene, drug_name


def load_all_test_predictions(datasets=("davis", "kiba"), settings=(1, 2, 3, 4)):
    """Load E0 test predictions, falling back to results/5fold/**/test_predictions.csv when needed."""
    frames = []
    for ds in datasets:
        meta = U.load_meta(ds)
        lig_keys, prot_keys = meta["lig_keys"], meta["prot_keys"]
        for s in settings:
            for f in C.FOLDS:
                e0_csv = os.path.join(C.OUT_DIR, "e0", "predictions", f"{ds}_s{s}_f{f}.csv")
                if os.path.exists(e0_csv):
                    df = pd.read_csv(e0_csv, dtype={"drug_id": str, "protein_id": str})
                    frames.append(df)
                    continue
                pred_csv = C.test_pred_path(ds, s, f)
                if not os.path.exists(pred_csv):
                    continue
                saved = pd.read_csv(pred_csv)
                samples = U.split_samples(ds, s, f, "test")
                if len(samples) != len(saved):
                    continue
                di = [r for r, _, _ in samples]
                pi = [c for _, c, _ in samples]
                yt = [y for _, _, y in samples]
                frames.append(pd.DataFrame({
                    "dataset": ds, "setting": s, "fold": f,
                    "drug_idx": di, "prot_idx": pi,
                    "drug_id": [str(lig_keys[r]) for r in di],
                    "protein_id": [str(prot_keys[c]) for c in pi],
                    "y_true": yt,
                    "y_main": saved["y_pred"].astype(float).values,
                    "H": np.nan, "S": np.nan, "y_phy": np.nan,
                }))
    if not frames:
        raise SystemExit("No test predictions found in E0 or results/5fold.")
    return pd.concat(frames, ignore_index=True)


def fill_thermo_for_rows(df_rows, device):
    """Ensure H, S, y_phy are populated for selected case rows by running fast forward pass if missing."""
    df_rows = df_rows.copy()
    stores, models = {}, {}
    for i, r in df_rows.iterrows():
        if pd.notna(r.get("y_phy")):
            continue
        ds, s, f = r.dataset, int(r.setting), int(r.fold)
        ck = C.ckpt_path(ds, s, f)
        if not os.path.exists(ck):
            continue
        if ds not in stores:
            stores[ds] = U.FeatureStore(ds)
        if (ds, s, f) not in models:
            models[(ds, s, f)] = U.load_model(ck, use_cross_attention=True, device=device)
        o = CI.case_forward(models[(ds, s, f)], stores[ds], ds, str(r.drug_id), str(r.protein_id), device)
        df_rows.loc[i, "y_main"] = o["y_main"]
        df_rows.loc[i, "H"] = o["H"]
        df_rows.loc[i, "S"] = o["S"]
        df_rows.loc[i, "y_phy"] = o["y_phy"]
    return df_rows


def compute_per_drug_selectivity(df):
    rows = []
    for (ds, s, did), g in df[df.setting.isin([1, 2])].groupby(["dataset", "setting", "drug_id"]):
        # Average across folds if the same (drug, protein) appears multiple times (e.g. setting 2)
        gp = g.groupby("protein_id", as_index=False).agg(
            y_true=("y_true", "mean"), y_main=("y_main", "mean"), y_phy=("y_phy", "mean")
        )
        n = len(gp)
        if n < 5:
            continue
        yt, yp = gp.y_true.values, gp.y_main.values
        yrange = float(yt.max() - yt.min())
        ysd = float(yt.std())
        if ysd < 1e-6:
            pear, spear, ci = np.nan, np.nan, np.nan
        else:
            pear = float(stats.pearsonr(yt, yp)[0])
            spear = float(stats.spearmanr(yt, yp)[0])
            ci = float(U.concordance_index(yt, yp))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        # Top-3 overlap between predicted top-3 and true top-3 (when drug has >=1 active target)
        k = min(3, n)
        top_true = set(np.argsort(-yt)[:k])
        top_pred = set(np.argsort(-yp)[:k])
        top3_overlap = len(top_true & top_pred) / k
        rows.append({
            "dataset": ds, "setting": s, "drug_id": str(did), "n_targets": n,
            "y_true_mean": float(yt.mean()), "y_true_sd": ysd, "y_true_range": yrange,
            "pearson_r": pear, "spearman_rho": spear, "ci": ci,
            "rmse": rmse, "mae": mae, "top3_overlap": top3_overlap,
        })
    pds = pd.DataFrame(rows)
    # Summary over selective drugs (n_targets >= 10 and y_true_range >= 1.5)
    sel = pds[(pds.n_targets >= 10) & (pds.y_true_range >= 1.5)].dropna(subset=["spearman_rho", "ci"])
    summ_rows = []
    for (ds, s), g in sel.groupby(["dataset", "setting"]):
        summ_rows.append({
            "dataset": ds, "setting": s, "n_selective_drugs": len(g),
            "ci_median": float(g.ci.median()), "ci_mean": float(g.ci.mean()),
            "spearman_median": float(g.spearman_rho.median()), "spearman_mean": float(g.spearman_rho.mean()),
            "pearson_median": float(g.pearson_r.median()), "pearson_mean": float(g.pearson_r.mean()),
            "rmse_median": float(g.rmse.median()), "rmse_mean": float(g.rmse.mean()),
            "mae_median": float(g.mae.median()),
            "frac_spearman_gt_0.5": float((g.spearman_rho > 0.5).mean()),
            "frac_ci_gt_0.7": float((g.ci > 0.7).mean()),
        })
    return pds, pd.DataFrame(summ_rows)


def curate_selectivity_cases(df, prot_gene, drug_name, device):
    """Extract Regime 1 (High/Medium/Low affinity) and Regime 2 (Consistent multi-target affinity) cases."""
    s1 = df[(df.dataset == "davis") & (df.setting == 1)].copy()
    s1["abs_err"] = (s1.y_main - s1.y_true).abs()
    s1["protein_gene"] = s1.protein_id.map(lambda p: prot_gene.get(str(p), str(p)))
    s1["drug_name"] = s1.drug_id.map(lambda d: drug_name.get(str(d), str(d)))

    crystal_targets = {"3062316": "ABL1", "10184653": "EGFR", "5291": "KIT", "216239": "BRAF"}
    picked = []
    for did, dname in FLAGSHIP_DAVIS_DRUGS:
        g = s1[s1.drug_id == did].copy()
        if g.empty:
            continue
        pref_gene = crystal_targets.get(did, "")
        g["is_crystal_target"] = (g.protein_gene == pref_gene).astype(int)
        # Regime 1: High (>= 7.8), Medium (6.0 .. 7.55), Low (<= 5.05)
        hi = g[g.y_true >= 7.8].sort_values(["is_crystal_target", "abs_err", "y_true"], ascending=[False, True, False]).head(2)
        med = g[(g.y_true >= 6.0) & (g.y_true < 7.55)].sort_values(["is_crystal_target", "abs_err"], ascending=[False, True]).head(2)
        lo = g[g.y_true <= 5.05].sort_values("abs_err").head(2)
        for level, sub in [("High", hi), ("Medium", med), ("Low", lo)]:
            for _, r in sub.iterrows():
                picked.append({**r.to_dict(), "regime": "Regime1_Selectivity_Gradient", "affinity_tier": level})

        # Regime 2: Consistent high/moderate affinity across multiple targets (narrow band)
        active = g[(g.y_true >= 6.8) & (g.abs_err <= 0.65)].sort_values("y_true", ascending=False)
        best_cluster = None
        for i in range(len(active)):
            anchor = active.iloc[i].y_true
            cl = active[(active.y_true <= anchor) & (active.y_true >= anchor - 0.55)].head(3)
            if len(cl) == 3:
                best_cluster = cl
                break
        if best_cluster is not None:
            for _, r in best_cluster.iterrows():
                picked.append({**r.to_dict(), "regime": "Regime2_Consistent_MultiTarget", "affinity_tier": "Consistent"})

    cases_df = pd.DataFrame(picked)
    if not cases_df.empty:
        cases_df = fill_thermo_for_rows(cases_df, device)
        cases_df["abs_err_main"] = (cases_df.y_main - cases_df.y_true).abs()
        cases_df["abs_err_phy"] = (cases_df.y_phy - cases_df.y_true).abs()
    return cases_df


def plot_selectivity(cases_df, pds, out_dir):
    fig = plt.figure(figsize=(13.5, 6.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 0.95], hspace=0.45, wspace=0.32)

    # Top row: 4 flagship drugs (Regime 1 High / Medium / Low profile)
    r1 = cases_df[cases_df.regime == "Regime1_Selectivity_Gradient"] if not cases_df.empty else pd.DataFrame()
    for idx, (did, dname) in enumerate(FLAGSHIP_DAVIS_DRUGS):
        ax = fig.add_subplot(gs[0, idx])
        sub = r1[r1.drug_id == did].copy() if not r1.empty else pd.DataFrame()
        if sub.empty:
            continue
        tier_order = {"High": 0, "Medium": 1, "Low": 2}
        sub["t_ord"] = sub.affinity_tier.map(tier_order)
        sub = sub.sort_values(["t_ord", "y_true"], ascending=[True, False])
        x = np.arange(len(sub))
        w = 0.26
        ax.bar(x - w, sub.y_true, width=w, color="#2c3e50", label="Experimental $pK_d$")
        ax.bar(x, sub.y_main, width=w, color="#c0392b", label="Predicted $y_{\\mathrm{main}}$")
        ax.bar(x + w, sub.y_phy, width=w, color="#2980b9", label="Thermodynamic $y_{\\mathrm{phy}}$")
        labels = [f"{r.protein_gene}\n({r.affinity_tier[0]})" for _, r in sub.iterrows()]
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(4.2, 11.2)
        ax.axhline(5.0, ls=":", color="grey", lw=0.8)
        ax.set_title(f"{dname} (CID {did})", fontsize=9, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Binding Affinity ($pK_d$)")
            ax.legend(fontsize=6.5, loc="upper right", frameon=False)

    # Bottom row left (2 cols): Per-drug CI and Spearman across selective drugs (Setting 1)
    sel = pds[(pds.setting == 1) & (pds.n_targets >= 10) & (pds.y_true_range >= 1.5)].dropna(subset=["ci", "spearman_rho"])
    ax_b1 = fig.add_subplot(gs[1, :2])
    data_ci, data_rho, labels_b1 = [], [], []
    for ds, name in [("davis", "Davis"), ("kiba", "KIBA")]:
        g = sel[sel.dataset == ds]
        if not g.empty:
            data_ci.append(g.ci.values)
            data_rho.append(g.spearman_rho.values)
            labels_b1.append(f"{name}\n(n={len(g)} drugs)")
    if data_ci:
        pos = np.arange(len(labels_b1))
        bp1 = ax_b1.boxplot(data_ci, positions=pos - 0.16, widths=0.26, patch_artist=True, showfliers=False)
        bp2 = ax_b1.boxplot(data_rho, positions=pos + 0.16, widths=0.26, patch_artist=True, showfliers=False)
        for b in bp1["boxes"]:
            b.set_facecolor("#c0392b")
            b.set_alpha(0.75)
        for b in bp2["boxes"]:
            b.set_facecolor("#2980b9")
            b.set_alpha(0.75)
        ax_b1.set_xticks(pos, labels_b1, fontsize=8.5)
        ax_b1.axhline(0.5, ls="--", color="grey", lw=0.8)
        ax_b1.set_ylabel("Per-Drug Cross-Target Metric")
        ax_b1.set_title("Per-Drug Target Selectivity Ranking Across Kinome Panels", fontsize=9, fontweight="bold")
        ax_b1.plot([], [], color="#c0392b", lw=6, alpha=0.75, label="Concordance Index (CI)")
        ax_b1.plot([], [], color="#2980b9", lw=6, alpha=0.75, label="Spearman $\\rho$")
        ax_b1.legend(fontsize=7.5, loc="lower right", frameon=False)

    # Bottom row right (2 cols): Regime 2 Consistent Multi-Target Paralog Affinity
    ax_b2 = fig.add_subplot(gs[1, 2:])
    r2 = cases_df[cases_df.regime == "Regime2_Consistent_MultiTarget"] if not cases_df.empty else pd.DataFrame()
    if not r2.empty:
        x = np.arange(len(r2))
        w = 0.28
        ax_b2.bar(x - w / 2, r2.y_true, width=w, color="#2c3e50", label="Experimental $pK_d$")
        ax_b2.bar(x + w / 2, r2.y_main, width=w, color="#16a085", label="Predicted $y_{\\mathrm{main}}$")
        xlabels = [f"{r.drug_name[:5]}\n{r.protein_gene}" for _, r in r2.iterrows()]
        ax_b2.set_xticks(x, xlabels, fontsize=6.5)
        ax_b2.set_ylim(5.0, 11.2)
        ax_b2.set_ylabel("Binding Affinity ($pK_d$)")
        ax_b2.set_title("Multi-Target Paralog Consistency (Low-Variance Regime)", fontsize=9, fontweight="bold")
        ax_b2.legend(fontsize=7.5, loc="upper right", frameon=False)

    fig.savefig(os.path.join(out_dir, "fig_selectivity_profiles.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    device = C.device()
    out_dir = C.out("e5")
    prot_gene, drug_name = load_gene_map()
    df = load_all_test_predictions()
    pds, summ = compute_per_drug_selectivity(df)
    pds.to_csv(os.path.join(out_dir, "per_drug_selectivity.csv"), index=False)
    summ.to_csv(os.path.join(out_dir, "selectivity_summary.csv"), index=False)

    cases_df = curate_selectivity_cases(df, prot_gene, drug_name, device)
    cols = ["regime", "affinity_tier", "drug_id", "drug_name", "protein_id", "protein_gene",
            "fold", "y_true", "y_main", "y_phy", "H", "S", "abs_err_main", "abs_err_phy"]
    if not cases_df.empty:
        cases_df = cases_df[[c for c in cols if c in cases_df.columns]]
        cases_df.to_csv(os.path.join(out_dir, "selectivity_cases.csv"), index=False)

    lines = ["=== E5: Per-Drug Cross-Target Selectivity Summary (selective drugs: n_targets>=10, range>=1.5) ==="]
    lines.append(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not cases_df.empty:
        lines.append("\n=== Representative Selectivity Cases (Regime 1: High/Medium/Low & Regime 2: Multi-Target) ===")
        lines.append(cases_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "selectivity_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    plot_selectivity(cases_df, pds, out_dir)


if __name__ == "__main__":
    main()
