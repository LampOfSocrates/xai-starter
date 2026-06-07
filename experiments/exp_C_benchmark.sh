#!/usr/bin/env bash
#
# Experiment C - solubility benchmark suite at scale (capstone_l2).
# Thin wrapper over run_experiment.sh: fixes the notebook + exp-id, exposes the
# scientific knobs as flags with sensible defaults, forwards the rest.
#
# Examples:
#   experiments/exp_C_benchmark.sh                       # defaults (small smoke)
#   experiments/exp_C_benchmark.sh --n-train 5000 --plm facebook/esm2_t12_35M_UR50D \
#       --seeds '[0,1,2,3,4]' --campaign overnight-2026-06-08
#
set -euo pipefail
cd "$(dirname "$0")/.."

EXP_ID="C-sol-benchmark"
NOTEBOOK="capstones/capstone_l2_benchmark_suite.ipynb"

N_TRAIN=300
N_VAL=100
N_TEST=100
PLM="facebook/esm2_t6_8M_UR50D"
SEEDS="[0,1,2]"
MAX_EPOCHS=30
CAMPAIGN="${CAMPAIGN:-adhoc}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-train)    N_TRAIN="$2"; shift 2 ;;
    --n-val)      N_VAL="$2"; shift 2 ;;
    --n-test)     N_TEST="$2"; shift 2 ;;
    --plm)        PLM="$2"; shift 2 ;;
    --seeds)      SEEDS="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --campaign)   CAMPAIGN="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

experiments/run_experiment.sh \
  --exp-id "$EXP_ID" \
  --notebook "$NOTEBOOK" \
  --campaign "$CAMPAIGN" \
  -P "N_TRAIN=$N_TRAIN" \
  -P "N_VAL=$N_VAL" \
  -P "N_TEST=$N_TEST" \
  -P "PLM_NAME=$PLM" \
  -P "SEEDS=$SEEDS" \
  -P "MAX_EPOCHS=$MAX_EPOCHS"
