#!/usr/bin/env bash
#
# Long-running, resource-gated GPU campaign over the GPU-friendly pLM lessons.
#
# Each lesson runs ONCE at its largest practical setting — fine-tune lessons at
# high EPOCHS, inference lessons at a bigger backbone — through the standard
# driver (experiments/run_experiment.sh), which gates on CPU/GPU, papermills the
# notebook, and stamps every MLflow run with campaign / reproducibility tags.
#
# Resilience: a lesson that fails is retried a few times (exponential backoff),
# then SKIPPED so the batch always finishes. Nothing here aborts the campaign.
# Safe to launch and walk away; safe to re-run — each run accumulates in MLflow
# and shows up in every lesson's "Run comparison" chart.
#
#   experiments/run_plm_gpu.sh [campaign-name]
#
# Tunables (env vars): MAX_RETRIES (3), RETRY_BACKOFF (60s, grows per attempt),
# and per-tier model sizes below (FT_MODEL / BIG_MODEL / V_MODEL). Dial the
# models down if you hit out-of-memory.
#
set -uo pipefail                     # NOT -e: we survive a failing lesson
cd "$(dirname "$0")/.."              # repo root

CAMPAIGN="${1:-plm-gpu-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_BACKOFF="${RETRY_BACKOFF:-60}"

# Model tiers. Full fine-tunes stay at 150M (650M full-FT tends to OOM on a
# single consumer GPU); LoRA and the inference lessons go all the way to 650M.
FT_MODEL="${FT_MODEL:-facebook/esm2_t30_150M_UR50D}"     # full fine-tune
BIG_MODEL="${BIG_MODEL:-facebook/esm2_t33_650M_UR50D}"   # LoRA / inference
V_MODEL="${V_MODEL:-facebook/esm1v_t33_650M_UR90S_1}"    # variant effect (ESM-1v)

FAILED=()

run_one () {                          # nb  exp_id  reason  [-P ...]
  local nb="$1" exp="$2" reason="$3"; shift 3
  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo ""
    echo "=== [$exp] attempt $attempt/$MAX_RETRIES :: $nb ==="
    if experiments/run_experiment.sh \
         --exp-id "$exp" --notebook "$nb" --campaign "$CAMPAIGN" \
         --reason "$reason" "$@"; then
      echo "=== [$exp] OK ==="
      return 0
    fi
    echo "!!! [$exp] failed (attempt $attempt/$MAX_RETRIES)" >&2
    (( attempt < MAX_RETRIES )) && sleep $(( RETRY_BACKOFF * attempt ))
    (( attempt++ ))
  done
  echo "!!! [$exp] giving up after $MAX_RETRIES attempts — skipping" >&2
  FAILED+=("$exp")
  return 1
}

echo "=== campaign: $CAMPAIGN ==="
echo "=== full-FT model: $FT_MODEL | big model: $BIG_MODEL | variant model: $V_MODEL ==="

# --- Fine-tune lessons: longer epochs + a larger backbone --------------------
run_one plm/plm_l3_finetune_classification.ipynb   plm-l3-solubility \
  "Max-epoch 150M solubility fine-tune for the run-comparison history" \
  -P EPOCHS=50 -P MODEL_NAME="$FT_MODEL" -P N_TRAIN=5000

run_one plm/plm_l3b_multiclass_localization.ipynb  plm-l3b-localization \
  "Max-epoch 150M subcellular-localization fine-tune (weighted-F1)" \
  -P EPOCHS=50 -P MODEL_NAME="$FT_MODEL" -P N_TRAIN=5000

run_one plm/plm_l4_token_classification.ipynb      plm-l4-secstruct \
  "Max-epoch 150M secondary-structure token-classification" \
  -P EPOCHS=40 -P MODEL_NAME="$FT_MODEL"

run_one plm/plm_l7_lora_peft.ipynb                 plm-l7-lora \
  "Max-epoch LoRA on ESM-2 650M (PEFT lets the big model fit)" \
  -P EPOCHS=50 -P MODEL_NAME="$BIG_MODEL" -P N_TRAIN=5000

# --- Grid + inference lessons: bigger backbone (no epoch knob) ---------------
run_one plm/plm_l5_model_comparison.ipynb          plm-l5-modelcmp \
  "Model x pooling grid extended through the 150M backbone, more data" \
  -P 'MODELS=[["facebook/esm2_t6_8M_UR50D","ESM-2 8M"],["facebook/esm2_t12_35M_UR50D","ESM-2 35M"],["facebook/esm2_t30_150M_UR50D","ESM-2 150M"]]' \
  -P N_TRAIN=2000 -P N_TEST=500

run_one plm/plm_l2_zero_shot_variants.ipynb        plm-l2-variant \
  "Zero-shot ProteinGym scoring with ESM-1v 650M (purpose-built for variants)" \
  -P MODEL_NAME="$V_MODEL"

run_one plm/plm_l6_attention_contacts.ipynb        plm-l6-contacts \
  "Attention-contact precision@L with ESM-2 650M (deeper heads track contacts)" \
  -P MODEL_NAME="$BIG_MODEL"

run_one plm/plm_l10_inverse_folding.ipynb          plm-l10-invfold \
  "Native-sequence recovery with ESM-2 650M" \
  -P MODEL_NAME="$BIG_MODEL"

echo ""
echo "=== campaign $CAMPAIGN complete ==="
if (( ${#FAILED[@]} )); then
  echo "skipped after retries: ${FAILED[*]}" >&2
else
  echo "all lessons ran."
fi
echo "report:  .venv/Scripts/python.exe mlflow_report.py --campaign $CAMPAIGN"
echo "browse:  .venv/Scripts/python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db"
