"""
E2 - Case-study figures.

For each curated complex (all cases for the supplementary sequence tracks,
`main_figure: true` cases for the 3D / 2D panels):

  1. Ligand attribution maps (RDKit): heavy atoms coloured by Integrated Gradients
     attribution (`ig_main` by default, with hydrogen attribution folded onto the
     parent heavy atom), plus separate enthalpy (`ig_H`) / entropy (`ig_S`) maps
     and a ground-truth panel highlighting crystal contact atoms (<= 4 Å).
  2. 3D pocket view: PDB with normalized attribution written to the B-factor column
     (chosen chain + ligand only) and a PyMOL script; rendered headless when
     PyMOL is available (`pymol -cq` or `python -m pymol -cq`).
  3. Sequence track: z-scored residue attribution along the Davis sequence with
     binding-site / binding-range residues marked (setting 1 vs setting 3 model).

Outputs (interpret/outputs/e2/{case_id}/)
    ligand_attention.png/.svg, ligand_contacts.png/.svg
    ligand_enthalpy_H.png/.svg, ligand_entropy_S.png/.svg
    pocket_attention.pdb, render.pml, pocket_attention.png
    sequence_track.png
    top_residues.csv
"""
import os
import shutil
import argparse
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm as mcm  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdDepictor  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
from Bio.PDB import MMCIFParser, PDBIO, Select  # noqa: E402

from interpret import config as C  # noqa: E402
from interpret import common as U  # noqa: E402
from interpret.e1_struct_labels import load_cases, fetch_structure, usable_cases  # noqa: E402


# ----------------------------------------------------------------------------
# 2D ligand
# ----------------------------------------------------------------------------
def _draw(mol, colors, radii, path_base, legend="", size=(600, 450)):
    for ext, cls in (("png", rdMolDraw2D.MolDraw2DCairo), ("svg", rdMolDraw2D.MolDraw2DSVG)):
        d = cls(*size)
        opts = d.drawOptions()
        opts.addAtomIndices = False
        opts.clearBackground = True
        opts.legendFontSize = 18
        opts.fillHighlights = True
        atoms = list(colors.keys())
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, legend=legend, highlightAtoms=atoms,
                                           highlightAtomColors=colors, highlightAtomRadii=radii,
                                           highlightBonds=[])
        d.FinishDrawing()
        data = d.GetDrawingText()
        mode = "wb" if ext == "png" else "w"
        with open(f"{path_base}.{ext}", mode) as f:
            f.write(data)


def _draw_continuous_map(mol, atom_df, col, path_base, cmap="Reds", legend=""):
    if col not in atom_df.columns:
        return
    w = atom_df.set_index("atom_idx")[col].reindex(range(mol.GetNumAtoms())).fillna(0).values
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    cmap_f = matplotlib.colormaps[cmap]
    colors = {i: tuple(float(c) for c in cmap_f(0.15 + 0.85 * w[i])[:3]) for i in range(mol.GetNumAtoms())}
    radii = {i: 0.25 + 0.35 * w[i] for i in range(mol.GetNumAtoms())}
    _draw(mol, colors, radii, path_base, legend=legend)


def ligand_panels(smiles, atom_df, out_dir, score_col="ig_main", cmap="Reds"):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    col = score_col if score_col in atom_df.columns else ("ig_main" if "ig_main" in atom_df.columns else "maxpool")
    _draw_continuous_map(mol, atom_df, col, os.path.join(out_dir, "ligand_attention"), cmap=cmap)
    _draw_continuous_map(mol, atom_df, "ig_H", os.path.join(out_dir, "ligand_enthalpy_H"),
                         cmap="Reds", legend="Enthalpy branch |IG(H)|")
    _draw_continuous_map(mol, atom_df, "ig_S", os.path.join(out_dir, "ligand_entropy_S"),
                         cmap="Blues", legend="Entropy branch |IG(S)|")

    if "contact" in atom_df:
        cont = atom_df.set_index("atom_idx")["contact"].reindex(range(mol.GetNumAtoms()))
        cc = {i: (0.25, 0.5, 0.9) for i, v in cont.items() if v is True or v == 1 or v == "True"}
        _draw(mol, cc, {i: 0.4 for i in cc}, os.path.join(out_dir, "ligand_contacts"),
              legend="crystal contacts (<= 4 Å)")

    # colour bar for the figure legend
    cmap_f = matplotlib.colormaps[cmap]
    fig, ax = plt.subplots(figsize=(2.4, 0.35))
    fig.colorbar(mcm.ScalarMappable(cmap=cmap_f), cax=ax, orientation="horizontal", label="relative attribution")
    fig.savefig(os.path.join(out_dir, "ligand_colorbar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3D pocket
# ----------------------------------------------------------------------------
class ChainLigandSelect(Select):
    def __init__(self, chain_id, lig_chain, comp_id):
        self.chain_id, self.lig_chain, self.comp_id = chain_id, lig_chain, comp_id.upper()

    def accept_residue(self, res):
        ch = res.get_parent().id
        if res.id[0] == "W":
            return False
        if res.get_resname().strip().upper() == self.comp_id:
            return ch == self.lig_chain
        return ch == self.chain_id and (res.id[0] == " " or res.id[0].startswith("H_"))

    def accept_atom(self, atom):
        return (not atom.is_disordered()) or atom.get_altloc() in ("A", " ", "")


def write_bfactor_pdb(case, meta_row, res_scores, labels, out_pdb):
    path = fetch_structure(case["pdb_id"])
    model = next(iter(MMCIFParser(QUIET=True).get_structure(case["pdb_id"], path)))
    chain_id, lig_chain = meta_row["protein_chain"], meta_row["ligand_chain"]
    s = res_scores.set_index("davis_idx").score
    s = (s - s.min()) / (s.max() - s.min() + 1e-12) * 100.0
    lab = labels.set_index("davis_idx")
    key2b = {}
    for di, b in s.items():
        if di in lab.index:
            icode = lab.loc[di, "icode"] if isinstance(lab.loc[di, "icode"], str) else " "
            key2b[(int(lab.loc[di, "resseq"]), icode or " ")] = float(b)
    for ch in model:
        for res in ch:
            b = key2b.get((res.id[1], res.id[2] or " "), 0.0) if ch.id == chain_id else 0.0
            for atom in res.get_atoms():
                atom.set_bfactor(b)
    io = PDBIO()
    io.set_structure(model)
    io.save(out_pdb, ChainLigandSelect(chain_id, lig_chain, case.get("ligand_comp") or case["comp_id"]))


def write_pml(case, meta_row, top_res, top_site_res, out_dir):
    chain = meta_row["protein_chain"]
    comp = case.get("ligand_comp") or case["comp_id"]
    resi_top = "+".join(str(int(r)) for r in top_res.resseq)
    resi_site = "+".join(str(int(r)) for r in top_site_res.resseq) if not top_site_res.empty else resi_top
    pml = f"""reinitialize
load pocket_attention.pdb, cplx
hide everything
bg_color white
set ray_opaque_background, 0
set cartoon_transparency, 0.15
show cartoon, cplx and polymer
spectrum b, white_red, cplx and polymer, minimum=0, maximum=100
show sticks, cplx and resn {comp}
util.cbac cplx and resn {comp}
set stick_radius, 0.22, cplx and resn {comp}
select topres, cplx and polymer and chain {chain} and resi {resi_site}
show sticks, topres and not name N+C+O
color tv_red, topres and elem C
set stick_radius, 0.16, topres
set label_size, 18
set label_color, black
label topres and name CA, "%s%s" % (resn, resi)
orient cplx and resn {comp}
zoom cplx and resn {comp}, 9
set ray_trace_mode, 1
png pocket_attention.png, width=2400, height=1800, dpi=300, ray=1
save pocket_attention.pse
"""
    with open(os.path.join(out_dir, "render.pml"), "w") as f:
        f.write(pml)


def try_render(out_dir):
    cmds = []
    if shutil.which("pymol"):
        cmds.append(["pymol", "-cq", "render.pml"])
    cmds.append(["python", "-m", "pymol", "-cq", "render.pml"])
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and os.path.exists(os.path.join(out_dir, "pocket_attention.png")):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


# ----------------------------------------------------------------------------
# sequence track
# ----------------------------------------------------------------------------
def sequence_track(cid, labels, tracks, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    for name, df, color in tracks:
        if df is None:
            continue
        ax.plot(df.davis_idx, df.z, lw=0.8, color=color, label=name)
    lab = labels[labels.esm_covered]
    for zone, col, y0 in (("site", "#c0392b", -0.06), ("range", "#e69f00", -0.12)):
        idx = lab[lab.label == zone].davis_idx
        ax.scatter(idx, np.full(len(idx), y0), transform=ax.get_xaxis_transform(),
                   marker="|", s=40, color=col, clip_on=False, label=f"binding {zone}")
    lo, hi = lab.davis_idx.min(), lab.davis_idx.max()
    ax.axvspan(lo, hi, color="grey", alpha=0.08, lw=0)
    ax.set_xlabel("Davis residue index")
    ax.set_ylabel("attribution (z)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=4, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--all_3d", action="store_true", help="render 3D/2D panels for every case")
    ap.add_argument("--atom_score", default="ig_main",
                    choices=["ig_main", "ig_phy", "maxpool", "hybrid", "attn_all", "attn_site"])
    ap.add_argument("--primary_imp", default="ig_main")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    e1 = os.path.join(C.OUT_DIR, "e1")
    meta = pd.read_csv(os.path.join(e1, "labels", "cases_meta.csv"))
    meta = meta[meta.case_id.isin(usable_cases(meta))].set_index("case_id")
    cm = pd.read_csv(os.path.join(e1, "case_metrics.csv")) if os.path.exists(os.path.join(e1, "case_metrics.csv")) else None

    for case in load_cases(args.cases):
        cid = case["case_id"]
        if cid not in meta.index:
            continue
        out_dir = C.out("e2", cid)
        labels = pd.read_csv(os.path.join(e1, "labels", f"{cid}_residues.csv"))
        r1p = os.path.join(e1, "residue_scores", f"{cid}_s1_test.csv")
        r3p = os.path.join(e1, "residue_scores", f"{cid}_s3_test.csv")
        r1 = pd.read_csv(r1p) if os.path.exists(r1p) else None
        r3 = pd.read_csv(r3p) if os.path.exists(r3p) else None
        auc = ""
        if cm is not None:
            g = cm[(cm.case_id == cid) & (cm.importance == args.primary_imp)].set_index("tag").auroc_site
            auc = "  ".join(f"{t}: AUROC={g[t]:.2f}" for t in ("s1_test", "s3_test") if t in g)
        title = f"{case.get('drug_name', case['drug_id'])} – {case.get('protein_gene', case['protein_id'])} " \
                f"({case['pdb_id']})  {auc}"
        sequence_track(cid, labels, [("setting 1 (warm)", r1, "#2c3e50"), ("setting 3 (cold target)", r3, "#16a085")],
                       os.path.join(out_dir, "sequence_track.png"), title)

        if not (case.get("main_figure") or args.all_3d) or r1 is None:
            continue
        top = r1.sort_values("score", ascending=False).head(args.top_k)
        top_site = r1[r1.label.isin(["site", "range"])].sort_values("score", ascending=False).head(min(8, args.top_k))
        top.to_csv(os.path.join(out_dir, "top_residues.csv"), index=False)
        top_site.to_csv(os.path.join(out_dir, "top_pocket_residues.csv"), index=False)
        smiles = U.load_meta(case.get("dataset", "davis"))["ligands"][str(case["drug_id"])]
        ap_path = os.path.join(e1, "atom_scores", f"{cid}_s1_test.csv")
        if os.path.exists(ap_path):
            ligand_panels(smiles, pd.read_csv(ap_path), out_dir, args.atom_score)
        write_bfactor_pdb(case, meta.loc[cid], r1, labels, os.path.join(out_dir, "pocket_attention.pdb"))
        write_pml(case, meta.loc[cid], top, top_site, out_dir)
        rendered = False if args.no_render else try_render(out_dir)
        print(f"[{cid}] figures written to {out_dir} (3D rendered: {rendered})")


if __name__ == "__main__":
    main()
