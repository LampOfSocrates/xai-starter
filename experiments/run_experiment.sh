#!/usr/bin/env bash
#
# Generic experiment driver: gate -> mint ids -> papermill -> MLflow.
#
# Every scientific parameter is passed through to the notebook via papermill
# (-P key=value, repeatable). Campaign / reproducibility ids are injected as
# MLflow tags through the environment, so the notebook's mu.run(...) calls pick
# them up unchanged. Reruns with identical -P values share a config_hash but
# get a fresh run_uid, which is what lets you compare reproducibility.
#
# Usage:
#   experiments/run_experiment.sh \
#       --exp-id   A-sol-size-method \
#       --notebook capstones/capstone_l2_benchmark_suite.ipynb \
#       --campaign overnight-2026-06-08 \
#       --reason   "Does ESM-2 35M beat 8M on solubility once N=5000?" \
#       -P N_TRAIN=5000 -P PLM_NAME=facebook/esm2_t12_35M_UR50D -P 'SEEDS=[0,1,2,3,4]'
#
# --reason is REQUIRED: every run must carry a plain-English statement of why it
# is being run. It is logged as the MLflow tag `reason` so the report can show,
# per experiment, the question each run set out to answer.
#
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

PY=".venv/Scripts/python.exe"
KERNEL="${EXP_KERNEL:-xai-starter}"
EXP_ID=""
NOTEBOOK=""
CAMPAIGN="${CAMPAIGN:-adhoc}"
REASON=""
PARAMS=()                         # key=value scientific params for papermill

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-id)   EXP_ID="$2"; shift 2 ;;
    --notebook) NOTEBOOK="$2"; shift 2 ;;
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --reason)   REASON="$2"; shift 2 ;;
    -P)         PARAMS+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$EXP_ID"   ]] || { echo "--exp-id required" >&2; exit 2; }
[[ -n "$NOTEBOOK" ]] || { echo "--notebook required" >&2; exit 2; }
# Require a real plain-English reason: non-empty and more than one word, so a
# placeholder like "test" or "x" does not satisfy the requirement.
REASON_TRIMMED="$(echo "$REASON" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
if [[ ${#REASON_TRIMMED} -lt 10 || "$REASON_TRIMMED" != *" "* ]]; then
  echo "--reason required: a plain-English description (>=10 chars, multiple words) of WHY this experiment is run." >&2
  echo "  e.g. --reason \"Check whether ESM-2 35M beats 8M on solubility at N=5000\"" >&2
  exit 2
fi

# --- 0. Validate before spending any GPU time ------------------------------
[[ -x "$PY" ]] || { echo "venv python not found at $PY (run from repo root)" >&2; exit 3; }
[[ -f "$NOTEBOOK" ]] || { echo "notebook not found: $NOTEBOOK" >&2; exit 3; }
if ! "$PY" -m jupyter kernelspec list 2>/dev/null | grep -qiw "$KERNEL"; then
  echo "kernel '$KERNEL' not registered. Register the venv kernel with:" >&2
  echo "  $PY -m ipykernel install --user --name $KERNEL" >&2
  exit 3
fi

# --- 1. Resource gate (blocks until CPU and GPU are <= threshold) ----------
"$PY" experiments/lib/gate.py

# --- 2. Mint ids -----------------------------------------------------------
RUN_UID="$("$PY" experiments/lib/meta.py uid "$EXP_ID")"
CONFIG_HASH="$("$PY" experiments/lib/meta.py hash "${PARAMS[@]}")"
OUTDIR="experiments/runs/$EXP_ID/$RUN_UID"
mkdir -p "$OUTDIR"

echo "[run] exp_id=$EXP_ID campaign=$CAMPAIGN"
echo "[run] run_uid=$RUN_UID config_hash=$CONFIG_HASH"
echo "[run] reason: $REASON_TRIMMED"
echo "[run] params: ${PARAMS[*]:-(notebook defaults)}"

# --- 3. Inject ids as MLflow tags via env (mlflow_utils reads MLFLOW_TAG_*) -
export MLFLOW_TAG_campaign="$CAMPAIGN"
export MLFLOW_TAG_exp_id="$EXP_ID"
export MLFLOW_TAG_run_uid="$RUN_UID"
export MLFLOW_TAG_config_hash="$CONFIG_HASH"
export MLFLOW_TAG_reason="$REASON_TRIMMED"
# conda's stale SSL bundle breaks venv downloads; pin the venv certifi bundle.
export SSL_CERT_FILE="$("$PY" -c 'import certifi; print(certifi.where())')"
export HF_HUB_DISABLE_PROGRESS_BARS=1

# --- 4. Write a TYPED param file (papermill -p coerces to string) ----------
"$PY" experiments/lib/mkparams.py "$OUTDIR/params.yaml" "${PARAMS[@]:-}"

# Persist the resolved manifest next to the executed notebook.
{
  echo "exp_id: $EXP_ID"
  echo "campaign: $CAMPAIGN"
  echo "run_uid: $RUN_UID"
  echo "config_hash: $CONFIG_HASH"
  echo "reason: \"$REASON_TRIMMED\""
  echo "notebook: $NOTEBOOK"
  echo "params_file: params.yaml"
} > "$OUTDIR/manifest.yaml"

"$PY" -m papermill "$NOTEBOOK" "$OUTDIR/output.ipynb" \
  --kernel "$KERNEL" --log-output -f "$OUTDIR/params.yaml"

echo "[run] done -> $OUTDIR/output.ipynb"
echo "[run] MLflow: tags.run_uid = '$RUN_UID'  |  tags.config_hash = '$CONFIG_HASH'"
