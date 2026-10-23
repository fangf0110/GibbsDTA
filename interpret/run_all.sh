#!/usr/bin/env bash
# Interpretability pipeline launcher (run from anywhere; works relative to the project root).
#
#   bash interpret/run_all.sh                       # all steps
#   bash interpret/run_all.sh --steps e0,e3         # selected steps
#   PY=/path/to/python bash interpret/run_all.sh    # choose interpreter
#   PHYNET_DEVICE=cpu bash interpret/run_all.sh     # force CPU
#
# Steps (in order): unimol, e0, labels, e1, e2, e3, e4, e5
#   unimol  Uni-Mol row <-> RDKit atom alignment check        (gate)
#   e0      re-inference of all checkpoints + sanity report    (gate)
#   labels  structure download, sequence alignment, distance labels
#   e1      pocket attribution statistics + DMFF-DTA zone tables
#   e2      case figures (RDKit attribution & H/S maps, PyMOL 3D, sequence tracks)
#   e3      thermodynamic branch analysis
#   e4      64-step integrated-gradients completeness & branch agreement
#   e5      per-drug cross-target selectivity analysis (DeepDTAGen benchmark)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
STEPS="unimol,e0,labels,e1,e2,e3,e4,e5"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

OUT="${PHYNET_OUT:-$ROOT/interpret/outputs}"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "root=$ROOT python=$($PY -c 'import sys;print(sys.executable)') steps=$STEPS" | tee "$LOG"

run() {
  local name="$1"; shift
  echo -e "\n========== [$name] $(date '+%F %T') ==========" | tee -a "$LOG"
  local t0=$SECONDS
  if ! "$PY" -u -m "$@" 2>&1 | tee -a "$LOG"; then
    echo "[$name] FAILED - see $LOG" | tee -a "$LOG"
    exit 1
  fi
  echo "[$name] done in $((SECONDS - t0))s" | tee -a "$LOG"
}

IFS=',' read -ra S <<< "$STEPS"
for step in "${S[@]}"; do
  case "$step" in
    unimol) run unimol interpret.e0_unimol_check ;;
    e0)     run e0     interpret.e0_infer ;;
    labels) run labels interpret.e1_struct_labels ;;
    e1)     run e1     interpret.e1_pocket_stats ;;
    e2)     run e2     interpret.e2_figures ;;
    e3)     run e3     interpret.e3_thermo ;;
    e4)     run e4     interpret.e4_attribution ;;
    e5)     run e5     interpret.e5_selectivity ;;
    *) echo "unknown step: $step" | tee -a "$LOG"; exit 2 ;;
  esac
done

echo -e "\nAll requested steps finished. Pack results with: bash interpret/pack_outputs.sh" | tee -a "$LOG"

