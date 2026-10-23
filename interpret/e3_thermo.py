"""
E3 - Behaviour of the thermodynamic (H - T*S) branch.

Inputs: E0 per-fold predictions (y_main, H, S, y_phy for every test pair).

Because y_phy = H - T*S is invariant to H -> H + T*c, S -> S + c, only quantities
that are stable across the 5 independently trained folds are interpreted, and
the descriptor analysis is reported as exploratory.

Analyses
  1. Branch agreement   : MSE / CI of y_phy, Pearson(y_main, y_phy) per dataset x setting x fold
  2. Decomposition shape: Pearson(H, S), Var(H)/Var(T*S), behaviour of the gauge
                          direction g = (H + T*S)/2 (the part y_phy does not see)
  3. Descriptor links   : drug-level partial Spearman of H and S with RDKit
                          descriptors controlling for y_phy; sign consistency over folds

Outputs (interpret/outputs/e3/)
    branch_summary.csv, branch_summary_agg.csv
    descriptor_partial_spearman.csv, descriptor_consistency.csv
    fig_main_vs_phy_{ds}.png, fig_H_vs_S_{ds}.png, fig_descriptor_heatmap_{ds}.png
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit import RDLogger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from interpret import config as C  # noqa: E402
from interpret import common as U  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DESCRIPTORS = {
    "HBD": Lipinski.NumHDonors,
    "HBA": Lipinski.NumHAcceptors,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "HeavyAtoms": Descriptors.HeavyAtomCount,
    "LogP": Descriptors.MolLogP,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
}


def load_predictions():
    files = sorted(glob.glob(os.path.join(C.OUT_DIR, "e0", "predictions", "*.csv")))
    if not files:
        raise SystemExit("No E0 predictions found - run e0_infer first.")
    return pd.concat([pd.read_csv(f, dtype={"drug_id": str, "protein_id": str}) for f in files],
                     ignore_index=True)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the rank-linear effect of z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def branch_summary(df):
    rows = []
    T = C.TEMPERATURE
    for (ds, s, f), g in df.groupby(["dataset", "setting", "fold"]):
        gauge = 0.5 * (g.H + T * g.S)
        rows.append({
            "dataset": ds, "setting": s, "fold": f, "n": len(g),
            "mse_main": U.mse(g.y_true, g.y_main), "mse_phy": U.mse(g.y_true, g.y_phy),
            "ci_main": U.concordance_index(g.y_true.values, g.y_main.values),
            "ci_phy": U.concordance_index(g.y_true.values, g.y_phy.values),
            "pearson_main_phy": stats.pearsonr(g.y_main, g.y_phy)[0],
            "mse_main_vs_phy": U.mse(g.y_main, g.y_phy),
            "pearson_H_S": stats.pearsonr(g.H, g.S)[0],
            "var_H_over_var_TS": g.H.var() / (T ** 2 * g.S.var() + 1e-12),
            "mean_H": g.H.mean(), "mean_S": g.S.mean(), "sd_H": g.H.std(), "sd_S": g.S.std(),
            "sd_gauge": gauge.std(), "sd_yphy": g.y_phy.std(),
            "pearson_gauge_y": stats.pearsonr(gauge, g.y_true)[0],
            "pearson_H_y": stats.pearsonr(g.H, g.y_true)[0],
            "pearson_S_y": stats.pearsonr(g.S, g.y_true)[0],
        })
    out = pd.DataFrame(rows)
    num = out.columns.drop(["dataset", "setting", "fold"])
    agg = out.groupby(["dataset", "setting"])[list(num)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return out, agg.reset_index()


def descriptor_analysis(df, settings=(1, 2)):
    rows = []
    desc_cache = {}
    for ds in df.dataset.unique():
        meta = U.load_meta(ds)
        for k, smi in meta["ligands"].items():
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                desc_cache[(ds, k)] = {n: fn(mol) for n, fn in DESCRIPTORS.items()}
    for (ds, s, f), g in df[df.setting.isin(settings)].groupby(["dataset", "setting", "fold"]):
        drug = g.groupby("drug_id")[["H", "S", "y_phy", "y_true"]].mean().reset_index()
        drug = drug[[(ds, k) in desc_cache for k in drug.drug_id]]
        if len(drug) < 10:
            continue
        D = pd.DataFrame([desc_cache[(ds, k)] for k in drug.drug_id])
        for name in DESCRIPTORS:
            for comp in ("H", "S"):
                r, p = partial_spearman(drug[comp].values, D[name].values, drug.y_phy.values)
                rows.append({"dataset": ds, "setting": s, "fold": f, "descriptor": name,
                             "component": comp, "partial_rho": r, "p": p, "n_drugs": len(drug)})
    res = pd.DataFrame(rows)
    if res.empty:
        return res, res
    cons = (res.groupby(["dataset", "setting", "descriptor", "component"])
            .agg(rho_mean=("partial_rho", "mean"), rho_sd=("partial_rho", "std"),
                 n_folds=("partial_rho", "size"),
                 n_pos=("partial_rho", lambda v: int((v > 0).sum())),
                 n_sig=("p", lambda v: int((v < 0.05).sum())))
            .reset_index())
    cons["sign_consistent"] = (cons.n_folds >= 3) & ((cons.n_pos == cons.n_folds) | (cons.n_pos == 0))
    return res, cons


def plots(df, cons, out_dir):
    for ds in df.dataset.unique():
        g = df[(df.dataset == ds) & (df.setting == 1)]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.hexbin(g.y_main, g.y_phy, gridsize=60, bins="log", cmap="viridis")
        lim = [min(g.y_main.min(), g.y_phy.min()), max(g.y_main.max(), g.y_phy.max())]
        ax.plot(lim, lim, "w--", lw=0.8)
        ax.set_xlabel(r"$y_{\mathrm{main}}$")
        ax.set_ylabel(r"$y_{\mathrm{phy}} = H - T\cdot S$")
        ax.set_title(f"{ds} (setting 1, 5 folds)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_main_vs_phy_{ds}.png"), dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=False, sharey=False)
        for f, ax in zip(C.FOLDS, axes):
            gf = g[g.fold == f]
            if gf.empty:
                continue
            sc = ax.scatter(gf.S, gf.H, c=gf.y_true, s=2, cmap="coolwarm", rasterized=True)
            ax.set_title(f"fold {f}  r={stats.pearsonr(gf.H, gf.S)[0]:.2f}", fontsize=8)
            ax.set_xlabel("S")
            ax.set_ylabel("H")
        fig.colorbar(sc, ax=axes, label="affinity")
        fig.savefig(os.path.join(out_dir, f"fig_H_vs_S_{ds}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        if cons is not None and not cons.empty:
            c = cons[(cons.dataset == ds) & (cons.setting == 1)]
            if c.empty:
                continue
            piv = c.pivot(index="descriptor", columns="component", values="rho_mean")
            flag = c.pivot(index="descriptor", columns="component", values="sign_consistent")
            fig, ax = plt.subplots(figsize=(3.2, 3.8))
            im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            ax.set_xticks(range(piv.shape[1]), piv.columns)
            ax.set_yticks(range(piv.shape[0]), piv.index)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    mark = "*" if flag.values[i, j] else ""
                    ax.text(j, i, f"{piv.values[i, j]:.2f}{mark}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, label="partial Spearman | y_phy")
            ax.set_title(f"{ds}: * = same sign in all folds", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"fig_descriptor_heatmap_{ds}.png"), dpi=300)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out_dir = C.out("e3")
    df = load_predictions()
    summ, agg = branch_summary(df)
    summ.to_csv(os.path.join(out_dir, "branch_summary.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "branch_summary_agg.csv"), index=False)
    res, cons = descriptor_analysis(df)
    res.to_csv(os.path.join(out_dir, "descriptor_partial_spearman.csv"), index=False)
    cons.to_csv(os.path.join(out_dir, "descriptor_consistency.csv"), index=False)
    plots(df, cons, out_dir)
    cols = ["dataset", "setting", "mse_main_mean", "mse_phy_mean", "ci_main_mean", "ci_phy_mean",
            "pearson_main_phy_mean", "pearson_H_S_mean", "pearson_H_S_std", "var_H_over_var_TS_mean"]
    print(agg[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not cons.empty:
        print("\nSign-consistent descriptor links (setting 1):")
        print(cons[(cons.setting == 1) & cons.sign_consistent]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
