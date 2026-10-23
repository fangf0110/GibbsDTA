"""
Shared utilities: dataset metadata, fold reconstruction, feature loading,
model loading and an instrumented forward pass.

The instrumented forward (`forward_full`) re-implements `PhyNetDTA.forward`
step by step using the model's own sub-modules, so that per-head attention,
pooled vectors and max-pool winners can be returned.  `check_forward_equivalence`
verifies that it reproduces the original forward exactly.
"""
import os
import sys
import json
import pickle
from functools import lru_cache

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from interpret import config as C

if C.PROJECT_ROOT not in sys.path:
    sys.path.insert(0, C.PROJECT_ROOT)
from model import PhyNetDTA  # noqa: E402


# ----------------------------------------------------------------------------
# Dataset metadata
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_meta(dataset):
    root = os.path.join(C.DATA_PATH, dataset)
    with open(os.path.join(root, "ligands_can.txt")) as f:
        ligands = json.load(f)
    with open(os.path.join(root, "proteins.txt")) as f:
        proteins = json.load(f)
    with open(os.path.join(root, "Y"), "rb") as f:
        Y = pickle.load(f, encoding="latin1")
    return {
        "ligands": ligands,
        "proteins": proteins,
        "lig_keys": list(ligands.keys()),
        "prot_keys": list(proteins.keys()),
        "Y": Y,
    }


@lru_cache(maxsize=None)
def load_folds(dataset, setting):
    path = os.path.join(C.DATA_PATH, dataset, "folds", C.FOLD_FILE.format(setting=setting))
    with open(path) as f:
        return json.load(f)


def split_samples(dataset, setting, fold, split="test"):
    """Replicates PhyNetDataset ordering: iterate fold indices, drop NaN labels.

    Returns a list of (row, col, y) where row indexes ligands and col proteins.
    """
    meta = load_meta(dataset)
    Y = meta["Y"]
    n_prot = Y.shape[1]
    idx = load_folds(dataset, setting)[fold][split]
    out = []
    for i in idx:
        r, c = int(i) // n_prot, int(i) % n_prot
        y = Y[r, c]
        if np.isnan(y):
            continue
        out.append((r, c, float(y)))
    return out


def pair_split(dataset, setting, fold, drug_id, prot_id):
    """Return 'train' / 'val' / 'test' / None for a (drug, protein) pair in a fold."""
    meta = load_meta(dataset)
    r = meta["lig_keys"].index(drug_id)
    c = meta["prot_keys"].index(prot_id)
    flat = r * meta["Y"].shape[1] + c
    folds = load_folds(dataset, setting)[fold]
    for split in ("test", "train", "val"):
        if flat in set(folds.get(split, [])):
            return split
    return None


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
class FeatureStore:
    """Caches pre-computed Uni-Mol / ESM-2 token features in memory."""

    def __init__(self, dataset):
        self.dataset = dataset
        base = os.path.join(C.DATA_PATH, dataset, "features")
        self.lig_dir = os.path.join(base, C.SMI_FEAT_DIR)
        self.prot_dir = os.path.join(base, C.PROT_FEAT_DIR)
        self._lig, self._prot = {}, {}

    def ligand(self, key):
        if key not in self._lig:
            self._lig[key] = torch.load(os.path.join(self.lig_dir, f"{key}.pt"), weights_only=True).float()
        return self._lig[key]

    def protein(self, key):
        if key not in self._prot:
            self._prot[key] = torch.load(os.path.join(self.prot_dir, f"{key}.pt"), weights_only=True).float()
        return self._prot[key]


def collate(smi_list, prot_list):
    """Identical padding / masking convention to dataset.collate_fn (True = padded)."""
    smi = pad_sequence(smi_list, batch_first=True, padding_value=0.0)
    prot = pad_sequence(prot_list, batch_first=True, padding_value=0.0)
    smi_mask = smi.sum(dim=-1) == 0.0
    prot_mask = prot.sum(dim=-1) == 0.0
    return smi, prot, smi_mask, prot_mask


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model(path, use_cross_attention=True, device="cpu"):
    model = PhyNetDTA(
        smi_feat_dim=C.SMI_FEAT_DIM,
        prot_feat_dim=C.PROT_FEAT_DIM,
        embed_dim=C.EMBED_DIM,
        num_heads=C.NUM_HEADS,
        use_cross_attention=use_cross_attention,
    )
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def forward_full(model, smi, prot, smi_mask, prot_mask, per_head=True):
    """Instrumented forward pass mirroring PhyNetDTA.forward.

    Returns a dict with
        y_main, H, S, y_phy            (B,)
        a_l2p  (B, h, N_L, N_P)  ligand tokens -> protein residues   (or None)
        a_p2l  (B, h, N_P, N_L)  protein residues -> ligand tokens   (or None)
        v_L, v_P                       (B, d) pooled vectors
        win_L, win_P                   (B, d) index of the token winning each max-pool channel
    """
    smi_seq = model.smi_proj(smi)
    prot_seq = model.prot_proj(prot)
    a_l2p = a_p2l = None
    if model.use_cross_attention:
        smi_out, a_l2p = model.attn_l2p(
            smi_seq, prot_seq, prot_seq, key_padding_mask=prot_mask,
            need_weights=True, average_attn_weights=not per_head)
        prot_out, a_p2l = model.attn_p2l(
            prot_seq, smi_seq, smi_seq, key_padding_mask=smi_mask,
            need_weights=True, average_attn_weights=not per_head)
        smi_seq = smi_seq + smi_out
        prot_seq = prot_seq + prot_out
        if not per_head:
            a_l2p, a_p2l = a_l2p.unsqueeze(1), a_p2l.unsqueeze(1)

    smi_seq = smi_seq.masked_fill(smi_mask.unsqueeze(-1), -1e9)
    prot_seq = prot_seq.masked_fill(prot_mask.unsqueeze(-1), -1e9)
    v_L, win_L = smi_seq.max(dim=1)   # == AdaptiveMaxPool1d(1) over tokens
    v_P, win_P = prot_seq.max(dim=1)
    v = torch.cat([v_L, v_P], dim=1)

    y_main = model.fc_main(v).squeeze(-1)
    H = model.fc_h(v).squeeze(-1)
    S = model.fc_s(v).squeeze(-1)
    return {
        "y_main": y_main, "H": H, "S": S, "y_phy": H - C.TEMPERATURE * S,
        "a_l2p": a_l2p, "a_p2l": a_p2l,
        "v_L": v_L, "v_P": v_P, "win_L": win_L, "win_P": win_P,
    }


@torch.no_grad()
def check_forward_equivalence(model, smi, prot, smi_mask, prot_mask, atol=1e-5):
    y_main, h, s, _, _ = model(smi, prot, smi_mask, prot_mask)
    o = forward_full(model, smi, prot, smi_mask, prot_mask, per_head=True)
    ok = (torch.allclose(y_main.squeeze(-1), o["y_main"], atol=atol)
          and torch.allclose(h.squeeze(-1), o["H"], atol=atol)
          and torch.allclose(s.squeeze(-1), o["S"], atol=atol))
    diff = float((y_main.squeeze(-1) - o["y_main"]).abs().max())
    return ok, diff


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def concordance_index(y_true, y_pred):
    """O(n log n) CI with the Heaviside convention h(0)=0.5 used in the paper."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    order = np.argsort(y_true, kind="mergesort")
    yt, yp = y_true[order], y_pred[order]
    _, pr = np.unique(yp, return_inverse=True)
    pr = pr + 1
    n_rank = pr.max()
    tree = np.zeros(n_rank + 1, dtype=np.int64)

    def add(i):
        while i <= n_rank:
            tree[i] += 1
            i += i & -i

    def query(i):
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    concordant, total, n_seen = 0.0, 0, 0
    i, n = 0, len(yt)
    while i < n:
        j = i
        while j < n and yt[j] == yt[i]:
            j += 1
        for k in range(i, j):          # compare against all strictly smaller y
            less = query(pr[k] - 1)
            eq = query(pr[k]) - less
            concordant += less + 0.5 * eq
            total += n_seen
        for k in range(i, j):
            add(pr[k])
        n_seen += j - i
        i = j
    return concordant / total if total else float("nan")


def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def read_metrics_txt(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    pass
    return out

