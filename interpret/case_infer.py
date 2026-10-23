"""
Per-complex inference helpers shared by E1 (pocket statistics), E2 (figures)
and E4 (robustness).

Token conventions (verified)
  Uni-Mol ligand rows : 0 .. N_atoms-1 = RDKit Chem.AddHs(MolFromSmiles(smi)) atom order
                        (heavy atoms first, then H);  row N_atoms = extra special token.
  ESM-2 protein rows  : 0 = [CLS], 1..L = residues (L <= 1022), L+1 = [EOS].
                        Davis residue index i  <->  token i+1.
"""
import json
import os
from functools import lru_cache

import numpy as np
import torch
from rdkit import Chem

from interpret import config as C
from interpret import common as U


@lru_cache(maxsize=None)
def unimol_offset(dataset):
    path = os.path.join(C.OUT_DIR, "e0", "unimol_token_map.json")
    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)
        if dataset in info:
            return int(info[dataset]["offset"])
    return 0  # verified locally: extra token is the LAST row


@lru_cache(maxsize=None)
def ligand_topology(smiles):
    mol = Chem.MolFromSmiles(smiles)
    molh = Chem.AddHs(mol)
    n_heavy = mol.GetNumAtoms()
    parent = np.arange(molh.GetNumAtoms())
    for a in molh.GetAtoms():
        if a.GetIdx() >= n_heavy:
            nb = a.GetNeighbors()
            parent[a.GetIdx()] = nb[0].GetIdx() if nb else a.GetIdx()
    return n_heavy, molh.GetNumAtoms(), parent


def _zscore(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-12)


def fold_atoms(vec_rows, o, fold_h=True):
    v = vec_rows[o["atom_rows"]]
    n_heavy = o["n_heavy"]
    heavy = v[:n_heavy].copy()
    if fold_h and o["n_atoms"] > n_heavy:
        np.add.at(heavy, o["h_parent"][n_heavy:], v[n_heavy:])
    return heavy


def integrated_gradients(model, smi, prot, steps=16, chunk=16, targets=("y_main", "H", "S")):
    device = smi.device
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    g_s = {t: torch.zeros_like(smi[0]) for t in targets}
    g_p = {t: torch.zeros_like(prot[0]) for t in targets}
    for b in range(0, steps, chunk):
        a = alphas[b:b + chunk].view(-1, 1, 1)
        s_in = (a * smi).detach().requires_grad_(True)
        p_in = (a * prot).detach().requires_grad_(True)
        sm = torch.zeros(s_in.shape[:2], dtype=torch.bool, device=device)
        pm = torch.zeros(p_in.shape[:2], dtype=torch.bool, device=device)
        o = U.forward_full(model, s_in, p_in, sm, pm, per_head=False)
        for i, t in enumerate(targets):
            gs, gp = torch.autograd.grad(o[t].sum(), [s_in, p_in], retain_graph=i < len(targets) - 1)
            g_s[t] += gs.sum(0)
            g_p[t] += gp.sum(0)
    attr = {}
    for t in targets:
        attr[t] = ((smi[0] * g_s[t] / steps).sum(-1).detach().cpu().numpy(),
                   (prot[0] * g_p[t] / steps).sum(-1).detach().cpu().numpy())
    with torch.no_grad():
        sm = torch.zeros(smi.shape[:2], dtype=torch.bool, device=device)
        pm = torch.zeros(prot.shape[:2], dtype=torch.bool, device=device)
        f1 = U.forward_full(model, smi, prot, sm, pm, per_head=False)
        f0 = U.forward_full(model, torch.zeros_like(smi), torch.zeros_like(prot), sm, pm, per_head=False)
    delta = {t: float(f1[t] - f0[t]) for t in targets}
    if "H" in attr and "S" in attr:
        attr["y_phy"] = (attr["H"][0] - C.TEMPERATURE * attr["S"][0],
                         attr["H"][1] - C.TEMPERATURE * attr["S"][1])
        delta["y_phy"] = delta["H"] - C.TEMPERATURE * delta["S"]
    return attr, delta


def case_forward(model, store, dataset, drug_id, prot_id, device,
                 ig_steps=0, ig_chunk=16, ig_targets=("y_main", "H", "S")):
    meta = U.load_meta(dataset)
    smi = store.ligand(drug_id).unsqueeze(0).to(device)
    prot = store.protein(prot_id).unsqueeze(0).to(device)
    smi_mask = torch.zeros(smi.shape[:2], dtype=torch.bool, device=device)
    prot_mask = torch.zeros(prot.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        o = U.forward_full(model, smi, prot, smi_mask, prot_mask, per_head=True)
    n_heavy, n_atoms, parent = ligand_topology(meta["ligands"][drug_id])
    off = unimol_offset(dataset)
    n_tok_l, n_tok_p = smi.shape[1], prot.shape[1]
    assert n_tok_l == n_atoms + 1, f"{drug_id}: {n_tok_l} ligand rows vs {n_atoms} atoms (+1 expected)"
    n_res = n_tok_p - 2
    assert n_res == min(len(meta["proteins"][prot_id]), C.ESM_MAX_RESIDUES), f"{prot_id}: unexpected ESM length"
    out = {
        "y_main": float(o["y_main"]), "H": float(o["H"]), "S": float(o["S"]), "y_phy": float(o["y_phy"]),
        "win_L": o["win_L"][0].cpu().numpy(), "win_P": o["win_P"][0].cpu().numpy(),
        "atom_rows": np.arange(off, off + n_atoms), "n_heavy": n_heavy, "n_atoms": n_atoms,
        "h_parent": parent, "n_res": n_res, "n_tok_l": n_tok_l, "n_tok_p": n_tok_p,
        "a_l2p": None, "a_p2l": None,
        "ig_res": {}, "ig_atom": {}, "ig_atom_abs": {}, "ig_completeness": {},
    }
    if o["a_l2p"] is not None:
        out["a_l2p"] = o["a_l2p"][0].cpu().numpy()   # (h, N_L, N_P)
        out["a_p2l"] = o["a_p2l"][0].cpu().numpy()   # (h, N_P, N_L)
    if ig_steps and ig_steps > 0:
        attr, delta = integrated_gradients(model, smi, prot, steps=ig_steps, chunk=ig_chunk, targets=ig_targets)
        for t, (a_s, a_p) in attr.items():
            comp = (a_s.sum() + a_p.sum() - delta[t]) / (abs(delta[t]) + 1e-8)
            out["ig_res"][t] = a_p[1:1 + n_res]
            out["ig_atom"][t] = fold_atoms(a_s, out, fold_h=True)
            out["ig_atom_abs"][t] = fold_atoms(np.abs(a_s), out, fold_h=True)
            out["ig_completeness"][t] = float(comp)
    return out


_IG_KEY_MAP = {
    "ig_main": "y_main",
    "ig_y_main": "y_main",
    "ig_phy": "y_phy",
    "ig_y_phy": "y_phy",
    "ig_H": "H",
    "ig_S": "S",
}


def residue_scores(o, kind="ig_main", head=None):
    """Importance of each Davis residue (length n_res, index = Davis residue index)."""
    n_res = o["n_res"]
    if kind in _IG_KEY_MAP:
        t = _IG_KEY_MAP[kind]
        if t not in o.get("ig_res", {}):
            return None
        return np.abs(o["ig_res"][t])
    if kind == "maxpool":
        cnt = np.bincount(o["win_P"], minlength=o["n_tok_p"]).astype(float)
        return cnt[1:1 + n_res]
    if kind == "hybrid":
        ig = residue_scores(o, "ig_main")
        mp = residue_scores(o, "maxpool")
        if ig is None or mp is None:
            return None
        s = 0.5 * _zscore(ig) + 0.5 * _zscore(mp)
        return s - s.min()
    if kind == "attn":
        if o["a_l2p"] is None:
            return None
        a = o["a_l2p"] if head is None else o["a_l2p"][head:head + 1]
        s = a[:, o["atom_rows"], :].mean(axis=(0, 1))       # mean over heads & atom queries
        return s[1:1 + n_res]
    raise ValueError(kind)


def atom_scores(o, kind="ig_main", query_residues=None, fold_h=True, head=None):
    """Importance of each heavy atom (length n_heavy, index = RDKit heavy-atom index).

    query_residues: optional Davis residue indices whose attention rows are averaged
                    (e.g. binding-site residues); default = all residues.
    fold_h: add hydrogen scores onto their parent heavy atom.
    """
    n_heavy, n_atoms = o["n_heavy"], o["n_atoms"]
    if kind in _IG_KEY_MAP:
        t = _IG_KEY_MAP[kind]
        if t not in o.get("ig_atom_abs", {}):
            return None
        return o["ig_atom_abs"][t].copy()
    if kind == "maxpool":
        cnt = np.bincount(o["win_L"], minlength=o["n_tok_l"]).astype(float)
        s_all = cnt[o["atom_rows"]]
        heavy = s_all[:n_heavy].copy()
        if fold_h and n_atoms > n_heavy:
            np.add.at(heavy, o["h_parent"][n_heavy:], s_all[n_heavy:])
        return heavy
    if kind == "hybrid":
        ig = atom_scores(o, "ig_main", fold_h=fold_h)
        mp = atom_scores(o, "maxpool", fold_h=fold_h)
        if ig is None or mp is None:
            return None
        s = 0.5 * _zscore(ig) + 0.5 * _zscore(mp)
        return s - s.min()
    if kind in ("attn", "attn_all", "attn_site"):
        if o["a_p2l"] is None:
            return None
        a = o["a_p2l"] if head is None else o["a_p2l"][head:head + 1]
        if query_residues is None:
            q = np.arange(1, 1 + o["n_res"])
        else:
            q = np.asarray([r + 1 for r in query_residues if r < o["n_res"]], dtype=int)
            if len(q) == 0:
                return None
        s_all = a[:, q, :].mean(axis=(0, 1))[o["atom_rows"]]  # (n_atoms,)
        heavy = s_all[:n_heavy].copy()
        if fold_h and n_atoms > n_heavy:
            np.add.at(heavy, o["h_parent"][n_heavy:], s_all[n_heavy:])
        return heavy
    raise ValueError(kind)


def extra_token_mass(o):
    """Fraction of protein->ligand attention that lands on the extra (non-atom) Uni-Mol token."""
    if o["a_p2l"] is None:
        return np.nan
    a = o["a_p2l"].mean(0)[1:1 + o["n_res"]]
    extra = np.setdiff1d(np.arange(o["n_tok_l"]), o["atom_rows"])
    return float(a[:, extra].sum(-1).mean())


def save_npz(path, o):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(
        a_l2p=o["a_l2p"].astype(np.float16) if o["a_l2p"] is not None else np.zeros(0),
        a_p2l=o["a_p2l"].astype(np.float16) if o["a_p2l"] is not None else np.zeros(0),
        win_L=o["win_L"], win_P=o["win_P"], atom_rows=o["atom_rows"],
        scalars=np.array([o["y_main"], o["H"], o["S"], o["y_phy"]]),
    )
    for t, v in o.get("ig_res", {}).items():
        kw[f"ig_res_{t}"] = v.astype(np.float32)
    for t, v in o.get("ig_atom", {}).items():
        kw[f"ig_atom_{t}"] = v.astype(np.float32)
    np.savez_compressed(path, **kw)


