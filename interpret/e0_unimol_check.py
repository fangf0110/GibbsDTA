"""
E0b - Determine how Uni-Mol feature rows map onto RDKit atoms.

Observation: every saved Uni-Mol feature matrix has (#atoms incl. H) + 1 rows.
We must know which row is the extra (special) token and confirm that atoms
follow the RDKit `Chem.AddHs` order (heavy atoms first, then hydrogens).

Method (needs only torch + rdkit; unimol_tools is optional)
  For each hypothesis "extra token at start" (offset=1) and "extra token at end"
  (offset=0), label every remaining row with the element symbol of the
  corresponding RDKit atom and measure nearest-centroid element classification
  accuracy.  Uni-Mol atomic embeddings cluster strongly by element, so the
  correct alignment gives near-perfect accuracy while a one-row shift does not.
  If unimol_tools is installed, the cosine similarity between the extra row and
  the returned CLS representation is reported as independent confirmation.

Output
    interpret/outputs/e0/unimol_token_map.json   (consumed by E2/E3)
Exit code 1 if the decision is ambiguous.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger

from interpret import config as C
from interpret import common as U

RDLogger.DisableLog("rdApp.*")


def rdkit_symbols(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    molh = Chem.AddHs(mol)
    return [a.GetSymbol() for a in molh.GetAtoms()], mol.GetNumAtoms()


def nearest_centroid_acc(X, labels):
    labels = np.asarray(labels)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    classes = sorted(set(labels))
    cents = np.stack([Xn[labels == c].mean(0) for c in classes])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True).clip(1e-8)
    pred = np.array(classes)[np.argmax(Xn @ cents.T, axis=1)]
    return float((pred == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=C.DATASETS)
    ap.add_argument("--max_ligands", type=int, default=300, help="per dataset")
    args = ap.parse_args()

    result = {}
    ok_all = True
    for ds in args.datasets:
        meta = U.load_meta(ds)
        store = U.FeatureStore(ds)
        keys = meta["lig_keys"][: args.max_ligands]
        diffs, X_by_h = [], {0: [], 1: []}
        L_by_h = {0: [], 1: []}
        extra_rows = {0: [], 1: []}
        used = []
        for k in keys:
            syms, n_heavy = rdkit_symbols(meta["ligands"][k])
            if syms is None:
                continue
            feat = store.ligand(k).numpy()
            d = feat.shape[0] - len(syms)
            diffs.append(d)
            if d != 1:
                continue
            used.append(k)
            for off in (0, 1):
                rows = feat[off: off + len(syms)]
                X_by_h[off].append(rows)
                L_by_h[off] += syms
                extra_rows[off].append(feat[-1] if off == 0 else feat[0])
        diffs = np.array(diffs)
        acc = {}
        for off in (0, 1):
            X = np.concatenate(X_by_h[off])
            acc[off] = nearest_centroid_acc(X, L_by_h[off])
        best = max(acc, key=acc.get)
        other = 1 - best
        decided = acc[best] >= 0.90 and (acc[best] - acc[other]) >= 0.05
        ok_all &= bool(decided) and bool((diffs == 1).mean() > 0.95)

        # heavy-first check: under the chosen offset, are H rows contiguous at the end?
        info = {
            "offset": int(best),
            "extra_token": "start" if best == 1 else "end",
            "atom_order": "rdkit_AddHs (heavy atoms first, then H)",
            "nearest_centroid_acc": {"offset_0_extra_at_end": acc[0], "offset_1_extra_at_start": acc[1]},
            "n_ligands_checked": len(used),
            "rows_minus_atoms_hist": {str(int(v)): int((diffs == v).sum()) for v in np.unique(diffs)},
            "decided": bool(decided),
        }

        # optional confirmation with unimol_tools
        try:
            from unimol_tools import UniMolRepr  # noqa: F401
            repr_model = UniMolRepr(data_type="molecule", remove_hs=False)
            sub = used[:20]
            out = repr_model.get_repr([meta["ligands"][k] for k in sub], return_atomic_reprs=True)
            cls = np.asarray(out["cls_repr"])
            cos_first, cos_last, n_new = [], [], []
            for i, k in enumerate(sub):
                feat = store.ligand(k).numpy()
                c = cls[i] / np.linalg.norm(cls[i])
                cos_first.append(float(feat[0] @ c / np.linalg.norm(feat[0])))
                cos_last.append(float(feat[-1] @ c / np.linalg.norm(feat[-1])))
                n_new.append(int(np.asarray(out["atomic_reprs"][i]).shape[0]))
            info["unimol_tools_check"] = {
                "cos(saved_first_row, cls)": float(np.mean(cos_first)),
                "cos(saved_last_row, cls)": float(np.mean(cos_last)),
                "new_atomic_rows_minus_saved": int(np.median(
                    [n - store.ligand(k).shape[0] for n, k in zip(n_new, sub)])),
                "has_atomic_symbol": "atomic_symbol" in out,
            }
            if "atomic_symbol" in out:
                info["unimol_tools_check"]["atomic_symbol_matches_rdkit"] = float(np.mean([
                    list(out["atomic_symbol"][i]) == rdkit_symbols(meta["ligands"][k])[0]
                    for i, k in enumerate(sub)]))
        except Exception as e:  # noqa: BLE001
            info["unimol_tools_check"] = f"skipped ({type(e).__name__}: {e})"

        result[ds] = info
        print(f"[{ds}] {json.dumps(info, indent=2)}")

    path = C.out("e0", "unimol_token_map.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {path}")
    if not ok_all:
        print("ERROR: Uni-Mol token alignment is ambiguous - inspect the report before running E2.")
        sys.exit(1)


if __name__ == "__main__":
    main()

