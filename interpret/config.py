"""
Central configuration for the interpretability pipeline.

All paths are relative to the project root (the directory that contains
`model.py`, `data/` and `results/`), so the same code runs unchanged on the
local machine and on the Ubuntu server.  Override with environment variables
if the server layout differs:

    PHYNET_ROOT      project root            (default: parent of interpret/)
    PHYNET_OUT       output directory        (default: interpret/outputs)
    PHYNET_DEVICE    cuda / cpu              (default: cuda if available)
"""
import os

INTERPRET_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("PHYNET_ROOT", os.path.dirname(INTERPRET_DIR))
OUT_DIR = os.environ.get("PHYNET_OUT", os.path.join(INTERPRET_DIR, "outputs"))

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
DATASETS = ["davis", "kiba"]
SETTINGS = [1, 2, 3, 4]
FOLDS = [0, 1, 2, 3, 4]

# Candidate base directories for 5-fold checkpoints (checked in order; first existing file wins).
# Covers both the Mac layout (results/5fold/result_5fold_{ds} or results_5fold_{ds})
# and the default run.py server layout (results/{ds}).
def _candidate_5fold_bases(dataset):
    r = os.path.join(PROJECT_ROOT, "results")
    return [
        os.path.join(r, "5fold", f"result_5fold_{dataset}"),
        os.path.join(r, "5fold", f"results_5fold_{dataset}"),
        os.path.join(r, "5fold", f"result_5fold_{dataset}", dataset),
        os.path.join(r, "5fold", f"results_5fold_{dataset}", dataset),
        os.path.join(r, "5fold", dataset),
        os.path.join(r, f"result_5fold_{dataset}"),
        os.path.join(r, f"results_5fold_{dataset}"),
        os.path.join(r, f"result_5fold_{dataset}", dataset),
        os.path.join(r, f"results_5fold_{dataset}", dataset),
        os.path.join(r, dataset),
        os.path.join(r, "cold-start", f"results_cold_{dataset}", dataset),
        os.path.join(r, "cold-start", f"result_cold_{dataset}", dataset),
    ]


def _resolve_5fold_file(dataset, setting, fold, sub_dir, filename):
    rel = os.path.join(f"setting_{setting}", sub_dir, f"fold_{fold}", filename)
    candidates = _candidate_5fold_bases(dataset)
    for base in candidates:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return os.path.join(candidates[0], rel)


RESULTS_5FOLD = {
    "davis": os.path.join(PROJECT_ROOT, "results", "5fold", "result_5fold_davis"),
    "kiba": os.path.join(PROJECT_ROOT, "results", "5fold", "result_5fold_kiba"),
}

# Ablation checkpoints (setting 1 only; fold availability is checked at run time)
RESULTS_ABLATION = {
    "davis": {
        "no_phy": os.path.join(PROJECT_ROOT, "results", "ab", "davis", "results_davis_no_phy", "davis"),
        "no_cross": os.path.join(PROJECT_ROOT, "results", "ab", "davis", "results_davis_no_cross", "davis"),
        "no_phy_cross": os.path.join(PROJECT_ROOT, "results", "ab", "davis", "results_davis_no_phy_cross", "davis"),
    },
    "kiba": {
        "no_phy": os.path.join(PROJECT_ROOT, "results", "ab", "kiba", "results_kiba_no_phy", "kiba"),
        "no_phy_cross": os.path.join(PROJECT_ROOT, "results", "ab", "kiba", "results_kiba_no_phy_cross", "kiba"),
    },
}
ABLATION_USES_CROSS = {"no_phy": True, "no_cross": False, "no_phy_cross": False}

# Model / feature configuration actually used by the checkpoints.
# smi_proj.weight in every checkpoint is 256 x 512  ->  Uni-Mol atomic features.
SMI_ENCODER = "unimol"
SMI_FEAT_DIR = "ligands_unimol"
PROT_FEAT_DIR = "proteins"
SMI_FEAT_DIM = 512
PROT_FEAT_DIM = 480
EMBED_DIM = 256
NUM_HEADS = 4
TEMPERATURE = 1.0

# Fold files used for training (verified: test order of every test_predictions.csv
# matches cv_folds_setting{s}_stratified.json)
FOLD_FILE = "cv_folds_setting{setting}_stratified.json"

# ESM-2 tokenisation: [CLS] r_1 ... r_L [EOS], L <= 1022 (max_length=1024)
ESM_MAX_RESIDUES = 1022

# Structure-based labels
SITE_CUTOFF = 5.0      # Å, binding site
RANGE_CUTOFF = 8.0     # Å, binding range
CONTACT_CUTOFF = 4.0   # Å, ligand atom in contact with protein

CASES_DIR = os.path.join(INTERPRET_DIR, "cases")
STRUCT_DIR = os.path.join(INTERPRET_DIR, "structures")

BATCH_SIZE = 64
SEED = 42


def device():
    import torch
    env = os.environ.get("PHYNET_DEVICE")
    if env:
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


def ckpt_path(dataset, setting, fold):
    return _resolve_5fold_file(dataset, setting, fold, "train", "best_model.pth")


def test_pred_path(dataset, setting, fold):
    return _resolve_5fold_file(dataset, setting, fold, "test", "test_predictions.csv")


def test_metrics_path(dataset, setting, fold):
    return _resolve_5fold_file(dataset, setting, fold, "test", "metrics.txt")


def ablation_ckpt_path(dataset, variant, fold, setting=1):
    r = os.path.join(PROJECT_ROOT, "results")
    rel = os.path.join(f"setting_{setting}", "train", f"fold_{fold}", "best_model.pth")
    bases = [
        RESULTS_ABLATION.get(dataset, {}).get(variant),
        os.path.join(r, "ab", dataset, f"result_{dataset}_{variant}", dataset),
        os.path.join(r, "ab", dataset, f"results_{dataset}_{variant}"),
        os.path.join(r, "ab", dataset, f"result_{dataset}_{variant}"),
        os.path.join(r, "ab", f"results_{dataset}_{variant}", dataset),
        os.path.join(r, f"results_{dataset}_{variant}", dataset),
    ]
    for b in bases:
        if b:
            p = os.path.join(b, rel)
            if os.path.exists(p):
                return p
    base = RESULTS_ABLATION.get(dataset, {}).get(variant)
    return os.path.join(base, rel) if base else None


def out(*parts, make_dirs=True):
    p = os.path.join(OUT_DIR, *parts)
    if make_dirs:
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p

