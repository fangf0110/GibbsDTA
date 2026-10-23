# Interpretability pipeline (GibbsDTA / PhyNet-DTA)

Code is written locally and executed on the Ubuntu server. Everything is relative to
the project root (the folder containing `model.py`, `data/`, `results/5fold/`).

## 1. Upload

```bash
# from the Mac (adjust user/host/path)
rsync -av --exclude outputs --exclude structures interpret/ user@server:/path/to/PhyNet-DTA/interpret/
```

The server must already have `model.py`, `data/{davis,kiba}/` (incl. `features/ligands_unimol`,
`features/proteins`, `folds/`) and `results/5fold/`, `results/ab/` with the same layout as locally.

## 2. Environment (once)

```bash
conda activate <training-env>              # the env that has torch
pip install -r interpret/requirements.txt  # does NOT touch torch
conda install -c conda-forge pymol-open-source   # optional, for 3D rendering in E2
```

## 3. Run

```bash
bash interpret/run_all.sh                    # all steps, log in interpret/outputs/logs/
bash interpret/run_all.sh --steps e0         # a single step
nohup bash interpret/run_all.sh > interp.log 2>&1 &   # detached
```

| step | script | needs | output (`interpret/outputs/…`) |
|---|---|---|---|
| unimol | `e0_unimol_check.py` | features | `e0/unimol_token_map.json` — **gate** |
| e0 | `e0_infer.py` | checkpoints | `e0/sanity_report.txt` — **gate**, `e0/predictions/*.csv` |
| labels | `e1_struct_labels.py` | internet (RCSB) | `e1/labels/*` |
| e1 | `e1_pocket_stats.py` | labels | `e1/dmff_zone_table.csv/.txt`, `group_stats.txt`, `case_metrics.csv`, `ligand_element_stats.csv`, figures |
| e2 | `e2_figures.py` | e1 | `e2/<case>/` RDKit IG/H/S ligand maps, PDB + `render.pml`, sequence tracks |
| e3 | `e3_thermo.py` | e0 | `e3/branch_summary_agg.csv`, descriptor heatmaps |
| e4 | `e4_attribution.py` | labels | `e4/ig_summary.txt`, `ig_metrics.csv` (64-step IG completeness & branch agreement) |
| e5 | `e5_selectivity.py` | e0 or 5fold | `e5/selectivity_summary.txt`, `selectivity_cases.csv`, `fig_selectivity_profiles.png` |

Rough cost: e0 is a full test-set pass for 2 datasets × 4 settings × 5 folds (minutes on GPU);
e1 runs ~10 checkpoints × 137 complexes with batched IG attribution + 1000 circular-shift permutations;
e4 runs 64-step IG completeness checks; e5 evaluates per-drug target selectivity across Davis and KIBA.

## 4. Download results

```bash
bash interpret/pack_outputs.sh               # -> interpret_outputs_YYYYmmdd_HHMM.tar.gz (no .npz)
```

## Conventions (verified)

* Checkpoints use **Uni-Mol** atomic features (`smi_proj` input 512), not ChemBERTa.
* Uni-Mol rows `0..N-1` = atoms in RDKit `Chem.AddHs` order (heavy first, then H); row `N` is an extra
  special token (element-classification accuracy 99.95 % vs 65 % for the alternative alignment).
* ESM-2 rows: `0` = CLS, `1..L` = residues (L ≤ 1022, longer sequences are truncated), `L+1` = EOS.
* Primary token importance (`--primary_imp ig_main`) uses **Integrated Gradients (`|IG(y_main)|`)** (alongside
  `|IG(y_phy)|`, `maxpool` channel winner count, and `hybrid`), which faithfully captures spatial pocket
  and chemical group sensitivity through the trained checkpoints without requiring model retraining.
* Test sets: `data/<ds>/folds/cv_folds_setting{s}_stratified.json` (`test` key) — matches the stored
  `test_predictions.csv` row by row for every dataset/setting/fold.
* Cases: `interpret/cases/cases.yaml` (137 Davis complexes: X-ray ≤ 3 Å, pKd ≥ 6; `tier: core` = 71
  balanced pairs with ≤ 4 per drug and ≤ 2 per kinase). Main-figure cases:
  4G5J EGFR–afatinib, 7N9G & 4XEY ABL1–dasatinib, 1T46 KIT–imatinib, 1UWH BRAF–sorafenib.
  Regenerate with `interpret/tools/curate_cases.py` + `interpret/tools/build_cases_yaml.py`.


