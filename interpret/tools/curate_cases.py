"""
Curate drug-target co-crystal complexes for the Davis dataset from the RCSB PDB.

This is a one-off, network-dependent helper. It is run once (locally or on the
server) and its outputs are shipped together with the interpretability code:

    interpret/cases/candidates.csv    all (drug, Davis protein, PDB entry) hits
    interpret/cases/pairs.csv         best entry per (drug, Davis protein) pair

Pipeline
    1. Davis ligands -> RCSB chemical components (graph-relaxed SMILES search,
       filtered by InChIKey connectivity block so stereo/protonation do not matter).
    2. Chemical component -> PDB entries containing it.
    3. Entry polymer sequences -> Davis protein keys by k-mer containment
       (cheap identity screen; exact residue mapping is done later by alignment).
    4. Keep pairs that exist in the Davis affinity matrix.

Usage
    python interpret/tools/curate_cases.py --data_path data --dataset davis
"""
import os
import json
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/Title/JSON"

# Common non-drug components that must never be treated as the ligand of interest.
IGNORE_COMPS = {"HOH", "SO4", "GOL", "EDO", "PEG", "CL", "NA", "MG", "ZN", "ACT", "DMS", "PO4"}


def _post(url, payload, retries=4):
    for k in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code == 204:  # search API: no hits
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"  retry {k + 1} after error: {e}")
            time.sleep(2 * (k + 1))


def _get(url, retries=4):
    for k in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def inchikey_block(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol).split("-")[0]


def search_comps_by_smiles(smiles):
    payload = {
        "query": {
            "type": "terminal",
            "service": "chemical",
            "parameters": {
                "value": smiles,
                "type": "descriptor",
                "descriptor_type": "SMILES",
                "match_type": "graph-relaxed",
            },
        },
        "return_type": "mol_definition",
        "request_options": {"return_all_hits": True},
    }
    res = _post(SEARCH_URL, payload)
    if not res:
        return []
    return [h["identifier"] for h in res.get("result_set", [])]


def chem_comp_info(comp_ids):
    if not comp_ids:
        return {}
    q = """query($ids:[String!]!){ chem_comps(comp_ids:$ids){
            chem_comp{ id name }
            rcsb_chem_comp_descriptor{ InChIKey SMILES } } }"""
    res = _post(GRAPHQL_URL, {"query": q, "variables": {"ids": comp_ids}})
    out = {}
    for c in res["data"]["chem_comps"] or []:
        if c is None:
            continue
        cid = c["chem_comp"]["id"]
        desc = c.get("rcsb_chem_comp_descriptor") or {}
        out[cid] = {"name": c["chem_comp"]["name"], "inchikey": desc.get("InChIKey")}
    return out


def search_entries_by_comp(comp_id):
    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
                "operator": "exact_match",
                "value": comp_id,
            },
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True},
    }
    res = _post(SEARCH_URL, payload)
    if not res:
        return []
    return [h["identifier"] for h in res.get("result_set", [])]


def entry_details(entry_ids):
    q = """query($ids:[String!]!){ entries(entry_ids:$ids){
        rcsb_id
        exptl{ method }
        rcsb_entry_info{ resolution_combined }
        polymer_entities{
          rcsb_id
          entity_poly{ pdbx_seq_one_letter_code_can rcsb_entity_polymer_type pdbx_strand_id }
          rcsb_polymer_entity_container_identifiers{
            reference_sequence_identifiers{ database_accession database_name } }
          rcsb_entity_source_organism{ ncbi_scientific_name rcsb_gene_name{ value } }
        }
        nonpolymer_entities{
          nonpolymer_comp{ chem_comp{ id } }
          rcsb_nonpolymer_entity_container_identifiers{ auth_asym_ids }
        } } }"""
    out = []
    for i in range(0, len(entry_ids), 50):
        res = _post(GRAPHQL_URL, {"query": q, "variables": {"ids": entry_ids[i:i + 50]}})
        out.extend([e for e in res["data"]["entries"] if e is not None])
    return out


def kmer_set(seq, k=5):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def best_davis_match(pdb_seq, davis_kmers, k=5, min_len=150):
    """Return (davis_key, containment) of the Davis protein that best contains the PDB chain."""
    pdb_seq = pdb_seq.replace("\n", "")
    if len(pdb_seq) < min_len:
        return None, 0.0
    q = kmer_set(pdb_seq, k)
    if not q:
        return None, 0.0
    best_key, best_c = None, 0.0
    for key, ks in davis_kmers.items():
        c = len(q & ks) / len(q)
        if c > best_c:
            best_key, best_c = key, c
    return best_key, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data")
    ap.add_argument("--dataset", default="davis")
    ap.add_argument("--out_dir", default="interpret/cases")
    ap.add_argument("--min_containment", type=float, default=0.80,
                    help="Fraction of PDB-chain 5-mers found in the Davis sequence")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    root = os.path.join(args.data_path, args.dataset)
    ligands = json.load(open(os.path.join(root, "ligands_can.txt")))
    proteins = json.load(open(os.path.join(root, "proteins.txt")))
    Y = pickle.load(open(os.path.join(root, "Y"), "rb"), encoding="latin1")
    lig_keys, prot_keys = list(ligands), list(proteins)

    gene_map = {}
    csv_path = os.path.join(root, "Davis.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        gene_map = df.groupby("PROTEIN_ID").PROTEIN_GENE.first().to_dict()

    # ---- drug names from PubChem (best effort) ----
    names = {}
    try:
        cids = ",".join(lig_keys)
        res = _get(PUBCHEM_URL.format(cids))
        for p in res["PropertyTable"]["Properties"]:
            names[str(p["CID"])] = p.get("Title", "")
    except Exception as e:  # noqa: BLE001
        print(f"PubChem lookup failed ({e}); continuing without names")

    davis_kmers = {k: kmer_set(s) for k, s in proteins.items()}

    rows = []
    for li, (lkey, smi) in enumerate(ligands.items()):
        block = inchikey_block(smi)
        comps = [c for c in search_comps_by_smiles(smi) if c not in IGNORE_COMPS]
        info = chem_comp_info(comps)
        comps = [c for c in comps if info.get(c, {}).get("inchikey") and
                 info[c]["inchikey"].split("-")[0] == block]
        print(f"[{li + 1}/{len(ligands)}] {lkey} {names.get(lkey, '')}: comps={comps}")
        for comp in comps:
            entries = search_entries_by_comp(comp)
            if not entries:
                continue
            for e in entry_details(entries):
                method = ",".join(x["method"] for x in (e.get("exptl") or []))
                res_list = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or [None]
                resolution = res_list[0]
                lig_chains = []
                for npe in e.get("nonpolymer_entities") or []:
                    if npe["nonpolymer_comp"]["chem_comp"]["id"] == comp:
                        lig_chains += npe["rcsb_nonpolymer_entity_container_identifiers"]["auth_asym_ids"] or []
                for pe in e.get("polymer_entities") or []:
                    ep = pe.get("entity_poly") or {}
                    if ep.get("rcsb_entity_polymer_type") != "Protein":
                        continue
                    seq = ep.get("pdbx_seq_one_letter_code_can") or ""
                    pkey, cont = best_davis_match(seq, davis_kmers)
                    if pkey is None or cont < args.min_containment:
                        continue
                    pi = prot_keys.index(pkey)
                    y = Y[li, pi]
                    if np.isnan(y):
                        continue
                    accs = [r["database_accession"] for r in
                            ((pe.get("rcsb_polymer_entity_container_identifiers") or {})
                             .get("reference_sequence_identifiers") or [])
                            if r.get("database_name") == "UniProt"]
                    rows.append({
                        "drug_id": lkey,
                        "drug_name": names.get(lkey, ""),
                        "comp_id": comp,
                        "protein_id": pkey,
                        "protein_gene": gene_map.get(pkey, ""),
                        "davis_len": len(proteins[pkey]),
                        "y": float(y),
                        "pdb_id": e["rcsb_id"],
                        "method": method,
                        "resolution": resolution,
                        "protein_chains": ep.get("pdbx_strand_id", ""),
                        "ligand_chains": ",".join(sorted(set(lig_chains))),
                        "pdb_uniprot": ",".join(accs),
                        "pdb_seq_len": len(seq),
                        "kmer_containment": round(cont, 3),
                    })

    cand = pd.DataFrame(rows).drop_duplicates(["drug_id", "protein_id", "pdb_id"])
    cand.to_csv(os.path.join(args.out_dir, "candidates.csv"), index=False)
    print(f"\n{len(cand)} candidate (drug, protein, entry) rows")
    if cand.empty:
        return

    xr = cand[cand.method.str.contains("X-RAY", na=False)].copy()
    xr["resolution"] = pd.to_numeric(xr["resolution"], errors="coerce")
    xr = xr.sort_values(["drug_id", "protein_id", "resolution", "kmer_containment"],
                        ascending=[True, True, True, False])
    pairs = xr.groupby(["drug_id", "protein_id"], as_index=False).first()
    n_entries = xr.groupby(["drug_id", "protein_id"]).pdb_id.nunique().rename("n_entries")
    pairs = pairs.merge(n_entries.reset_index(), on=["drug_id", "protein_id"])
    pairs = pairs.sort_values(["y", "resolution"], ascending=[False, True])
    pairs.to_csv(os.path.join(args.out_dir, "pairs.csv"), index=False)
    print(f"{len(pairs)} unique (drug, protein) pairs with X-ray structures")
    print(pairs[["drug_name", "protein_gene", "y", "pdb_id", "resolution", "comp_id",
                 "kmer_containment"]].to_string(index=False))


if __name__ == "__main__":
    main()

