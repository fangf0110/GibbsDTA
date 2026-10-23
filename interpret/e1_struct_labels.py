"""
E1a - Structure-derived ground-truth labels for each curated complex.

For every case in interpret/cases/cases.yaml:
  * load the mmCIF (downloaded on demand into interpret/structures/),
  * pick the ligand copy (comp_id) and the protein chain it binds,
  * align the observed chain sequence to the Davis sequence (local, BLOSUM62),
  * label each mapped Davis residue by its minimum heavy-atom distance to the
    ligand: site (<=5 Å), range (5-8 Å), outside (>8 Å),
  * map crystal ligand atoms onto the Davis SMILES atoms (bond orders from
    template; MCS fallback) and flag atoms in contact (<=4 Å) with the protein.

Outputs (interpret/outputs/e1/labels/)
    {case_id}_residues.csv     davis_idx, aa, resseq, min_dist, label, esm_covered
    {case_id}_ligand_atoms.csv atom_idx (Davis SMILES heavy-atom index), element, min_dist, contact
    cases_meta.csv             alignment identity, coverage, mapping method, ...
"""
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import requests
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit import RDLogger
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa

from interpret import config as C
from interpret import common as U

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

try:
    from Bio.Data.PDBData import protein_letters_3to1_extended as THREE2ONE
except ImportError:  # older biopython
    from Bio.Data.SCOPData import protein_letters_3to1 as THREE2ONE


def load_cases(path=None):
    path = path or os.path.join(C.CASES_DIR, "cases.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["cases"]


def fetch_structure(pdb_id):
    os.makedirs(C.STRUCT_DIR, exist_ok=True)
    path = os.path.join(C.STRUCT_DIR, f"{pdb_id.upper()}.cif")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "w") as f:
            f.write(r.text)
    return path


def heavy_coords(atoms):
    xs = [a.coord for a in atoms if (a.element or "").upper() not in ("H", "D")]
    return np.array(xs, dtype=np.float64) if xs else np.zeros((0, 3))


def min_dist(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def one_letter(resname):
    r = resname.upper()
    return THREE2ONE.get(r, THREE2ONE.get(r.capitalize(), "X"))


def chain_residues(chain):
    """Observed amino-acid residues (incl. modified residues such as MSE/PTR)."""
    out = []
    for res in chain:
        het = res.id[0]
        if het == "W":
            continue
        if het == " " or (het.startswith("H_") and is_aa(res, standard=False) and "CA" in res):
            aa = one_letter(res.get_resname())
            if aa != "X" or het == " ":
                out.append((res, aa))
    return out


def make_aligner():
    al = Align.PairwiseAligner()
    al.mode = "local"
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score = -10.0
    al.extend_gap_score = -0.5
    return al


def align_chain_to_davis(davis_seq, chain_seq, aligner):
    aln = aligner.align(davis_seq, chain_seq)[0]
    mapping = {}
    for (d0, d1), (c0, c1) in zip(*aln.aligned):
        for k in range(d1 - d0):
            mapping[c0 + k] = d0 + k
    n_id = sum(davis_seq[d] == chain_seq[c] for c, d in mapping.items())
    identity = n_id / max(len(mapping), 1)
    coverage = len(mapping) / max(len(chain_seq), 1)
    return mapping, identity, coverage


def ligand_copies(model, comp_id):
    out = []
    for chain in model:
        for res in chain:
            if res.get_resname().strip().upper() == comp_id.upper() and res.id[0] != " ":
                out.append((chain.id, res))
    return out


def ligand_to_rdkit(res):
    lines, serial = [], 1
    for a in res:
        el = (a.element or a.get_name()[0]).strip().capitalize()
        if el.upper() in ("H", "D"):
            continue
        name = a.get_name()
        name = f" {name:<3}" if len(name) < 4 else name[:4]
        x, y, z = a.coord
        lines.append(f"HETATM{serial:5d} {name} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")
        serial += 1
    block = "\n".join(lines) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False, proximityBonding=True)
    return mol


def map_ligand_atoms(crystal, template):
    """Return (dict template_idx -> crystal_idx, method)."""
    if crystal is None:
        return {}, "failed"
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, Chem.Mol(crystal))
        match = fixed.GetSubstructMatch(template)
        if match and len(match) == template.GetNumAtoms():
            return {t: c for t, c in enumerate(match)}, "template"
    except Exception:  # noqa: BLE001
        pass
    try:
        res = rdFMCS.FindMCS(
            [template, crystal], atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=False,
            matchValences=False, timeout=20)
        q = Chem.MolFromSmarts(res.smartsString)
        mt = template.GetSubstructMatch(q)
        mc = crystal.GetSubstructMatch(q, useChirality=False)
        if mt and mc:
            return {t: c for t, c in zip(mt, mc)}, f"mcs({len(mt)}/{template.GetNumAtoms()})"
    except Exception:  # noqa: BLE001
        pass
    return {}, "failed"


def process_case(case, aligner):
    ds = case.get("dataset", "davis")
    meta = U.load_meta(ds)
    davis_seq = meta["proteins"][case["protein_id"]]
    smiles = meta["ligands"][str(case["drug_id"])]
    path = fetch_structure(case["pdb_id"])
    structure = MMCIFParser(QUIET=True).get_structure(case["pdb_id"], path)
    model = next(iter(structure))

    lig_comp = case.get("ligand_comp") or case["comp_id"]   # e.g. covalent adduct component
    copies = ligand_copies(model, lig_comp)
    if case.get("ligand_chain"):
        copies = [c for c in copies if c[0] == case["ligand_chain"]] or copies
    if not copies:
        raise ValueError(f"ligand {lig_comp} not found in {case['pdb_id']}")

    # candidate protein chains (restrict if specified)
    wanted = str(case.get("protein_chain") or "").split(",")
    wanted = [w.strip() for w in wanted if w.strip()]
    chains = [ch for ch in model if (not wanted or ch.id in wanted)]

    best = None
    for lig_chain, lig in copies:
        L = heavy_coords(lig.get_atoms())
        for ch in chains:
            residues = chain_residues(ch)
            if len(residues) < 50:
                continue
            dists = [min_dist(heavy_coords(r.get_atoms()), L) for r, _ in residues]
            n_site = int(np.sum(np.array(dists) <= C.SITE_CUTOFF))
            if n_site == 0:
                continue
            chain_seq = "".join(aa for _, aa in residues)
            mapping, ident, cov = align_chain_to_davis(davis_seq, chain_seq, aligner)
            score = n_site * ident
            if best is None or score > best["score"]:
                best = dict(score=score, lig_chain=lig_chain, lig=lig, L=L, chain=ch, residues=residues,
                            dists=dists, mapping=mapping, identity=ident, coverage=cov, n_site=n_site)
    if best is None:
        raise ValueError("no protein chain in contact with the ligand")

    # ---- residue labels ----
    rows = []
    for ci, ((res, aa), d) in enumerate(zip(best["residues"], best["dists"])):
        if ci not in best["mapping"]:
            continue
        di = best["mapping"][ci]
        label = "site" if d <= C.SITE_CUTOFF else ("range" if d <= C.RANGE_CUTOFF else "outside")
        rows.append({
            "davis_idx": di, "aa_davis": davis_seq[di], "aa_pdb": aa,
            "chain": best["chain"].id, "resseq": res.id[1], "icode": res.id[2].strip(),
            "min_dist": round(d, 3), "label": label,
            "esm_covered": di < C.ESM_MAX_RESIDUES,
        })
    res_df = pd.DataFrame(rows).sort_values("davis_idx")

    # ---- ligand atom labels ----
    template = Chem.MolFromSmiles(smiles)
    crystal = ligand_to_rdkit(best["lig"])
    amap, method = map_ligand_atoms(crystal, template)
    P = heavy_coords([a for r, _ in best["residues"] for a in r.get_atoms()])
    lig_rows = []
    if crystal is not None and amap:
        conf = crystal.GetConformer()
        for t in range(template.GetNumAtoms()):
            if t not in amap:
                lig_rows.append({"atom_idx": t, "element": template.GetAtomWithIdx(t).GetSymbol(),
                                 "crystal_name": "", "min_dist": np.nan, "contact": np.nan})
                continue
            p = conf.GetAtomPosition(amap[t])
            xyz = np.array([[p.x, p.y, p.z]])
            d = min_dist(xyz, P)
            info = crystal.GetAtomWithIdx(amap[t]).GetPDBResidueInfo()
            lig_rows.append({"atom_idx": t, "element": template.GetAtomWithIdx(t).GetSymbol(),
                             "crystal_name": info.GetName().strip() if info else "",
                             "min_dist": round(d, 3), "contact": bool(d <= C.CONTACT_CUTOFF)})
    lig_df = pd.DataFrame(lig_rows)

    site = res_df[res_df.label == "site"]
    meta_row = {
        "case_id": case["case_id"], "pdb_id": case["pdb_id"], "comp_id": case["comp_id"],
        "drug_id": case["drug_id"], "protein_id": case["protein_id"],
        "protein_chain": best["chain"].id, "ligand_chain": best["lig_chain"],
        "align_identity": round(best["identity"], 4), "align_coverage": round(best["coverage"], 4),
        "n_mapped": len(res_df), "n_site": int(len(site)),
        "n_range": int((res_df.label == "range").sum()),
        "site_esm_covered_frac": float(site.esm_covered.mean()) if len(site) else np.nan,
        "davis_len": len(davis_seq),
        "lig_map_method": method,
        "lig_atoms_mapped": int(lig_df.min_dist.notna().sum()) if len(lig_df) else 0,
        "lig_atoms_total": template.GetNumAtoms(),
        "lig_contact_frac": float(lig_df.contact.dropna().astype(float).mean()) if len(lig_df) else np.nan,
    }
    return res_df, lig_df, meta_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    args = ap.parse_args()
    out_dir = C.out("e1", "labels")
    aligner = make_aligner()
    metas = []
    for case in load_cases(args.cases):
        try:
            res_df, lig_df, m = process_case(case, aligner)
            res_df.to_csv(os.path.join(out_dir, f"{case['case_id']}_residues.csv"), index=False)
            lig_df.to_csv(os.path.join(out_dir, f"{case['case_id']}_ligand_atoms.csv"), index=False)
            m["status"] = "ok"
            m["ligand_comp"] = case.get("ligand_comp") or case["comp_id"]
            flag = ""
            if m["align_identity"] < 0.9:
                flag += " LOW_IDENTITY"
            if m["site_esm_covered_frac"] < 1.0:
                flag += " SITE_TRUNCATED_BY_ESM"
            if m["lig_map_method"] == "failed":
                flag += " LIGAND_MAP_FAILED"
            if m["n_site"] < MIN_SITE or (m["lig_contact_frac"] < MIN_LIG_CONTACT):
                flag += " LIGAND_SURFACE"   # likely a packing / non-pocket copy (check ligand_comp)
            m["flags"] = flag.strip()
            m["usable"] = not any(f in m["flags"] for f in EXCLUDE_FLAGS)
            print(f"[ok] {case['case_id']}: chain {m['protein_chain']} id={m['align_identity']:.3f} "
                  f"site={m['n_site']} range={m['n_range']} lig={m['lig_map_method']} "
                  f"contact={m['lig_contact_frac']:.2f} {flag}")
        except Exception as e:  # noqa: BLE001
            m = {"case_id": case["case_id"], "status": f"error: {e}", "flags": "ERROR", "usable": False}
            print(f"[ERR] {case['case_id']}: {e}")
        metas.append(m)
    df = pd.DataFrame(metas)
    df.to_csv(os.path.join(out_dir, "cases_meta.csv"), index=False)
    print(f"\n{int(df.usable.sum())}/{len(df)} cases usable for statistics; excluded by flag: "
          f"{df.loc[~df.usable.astype(bool), 'flags'].value_counts().to_dict()}")


MIN_SITE = 10
MIN_LIG_CONTACT = 0.5
EXCLUDE_FLAGS = ("ERROR", "LOW_IDENTITY", "SITE_TRUNCATED_BY_ESM", "LIGAND_SURFACE")


def usable_cases(meta_df):
    """Case IDs that enter the statistics (single rule used by E1, E2 and E4)."""
    if "usable" in meta_df:
        return set(meta_df[meta_df.usable.astype(bool)].case_id)
    return set(meta_df[meta_df.status == "ok"].case_id)


if __name__ == "__main__":
    main()
