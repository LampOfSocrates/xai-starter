#!/usr/bin/env bash
#
# One overnight batch: several experiments run SEQUENTIALLY, each one passing
# the resource gate first. Tag the whole batch with a single campaign id so the
# report groups everything at the tag level.
#
#   experiments/batch_overnight.sh overnight-2026-06-08
#
set -euo pipefail
cd "$(dirname "$0")/.."

CAMPAIGN="${1:-overnight-$(date -u +%Y-%m-%d)}"
echo "=== campaign: $CAMPAIGN ==="

# Experiment C across two pLM sizes, 5 seeds, more data. Each invocation gates
# before it starts, so the batch is polite to any foreground GPU work. Each run
# carries a --reason explaining the question it answers (logged as an MLflow tag).
experiments/exp_C_benchmark.sh --campaign "$CAMPAIGN" \
  --n-train 5000 --n-val 1000 --n-test 1000 --seeds '[0,1,2,3,4]' \
  --plm facebook/esm2_t6_8M_UR50D \
  --reason "Baseline ESM-2 8M: honest 4-model solubility ranking at N=5000, 5 seeds"

experiments/exp_C_benchmark.sh --campaign "$CAMPAIGN" \
  --n-train 5000 --n-val 1000 --n-test 1000 --seeds '[0,1,2,3,4]' \
  --plm facebook/esm2_t12_35M_UR50D \
  --reason "Does ESM-2 35M beat 8M on solubility at N=5000 once seeds give error bars?"

# Add more experiments here as their wrappers land (exp_A_*, exp_B_*, ...).

echo "=== campaign $CAMPAIGN complete -- generate the report with: ==="
echo "    .venv/Scripts/python.exe mlflow_report.py $CAMPAIGN   # (to build)"
