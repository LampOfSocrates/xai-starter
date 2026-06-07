#!/usr/bin/env bash
#
# Execute every lesson notebook top-to-bottom, in dependency order.
#
# Runs each notebook with `jupyter nbconvert --execute`, writing the executed
# copy (with outputs) back in place. Order follows README.md.
#   - Times each notebook; prints the time AND appends it to a log file.
#   - Points every HuggingFace cache at ONE shared directory so datasets and
#     models are downloaded once by the first notebook and reused by the rest.
#   - Stops on the first failure unless --continue-on-error is given.
#
# Usage:
#   ./run_all_notebooks.sh
#   ./run_all_notebooks.sh --continue-on-error --timeout 3600
#   ./run_all_notebooks.sh --tracks gnn,capstones
#   ./run_all_notebooks.sh --cache-dir /data/hf_cache
set -u

# ---- Defaults --------------------------------------------------------------
TRACKS="plm,gnn,capstones,ig"
TIMEOUT=1800
CACHE_DIR="${HOME}/.cache/huggingface"
CONTINUE_ON_ERROR=0

# ---- Parse args ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tracks)            TRACKS="$2"; shift 2 ;;
        --timeout)           TIMEOUT="$2"; shift 2 ;;
        --cache-dir)         CACHE_DIR="$2"; shift 2 ;;
        --continue-on-error) CONTINUE_ON_ERROR=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Pin execution to the project venv ------------------------------------
# Bare `jupyter` resolves via PATH and can launch the wrong interpreter (one
# without torch/transformers). Always drive nbconvert through the venv python.
if [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
    VENV_PY="$ROOT/.venv/Scripts/python.exe"   # Windows (git-bash)
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    VENV_PY="$ROOT/.venv/bin/python"           # Linux / macOS
else
    echo "venv python not found under $ROOT/.venv (see README to create it)." >&2
    exit 1
fi

# An activated conda env exports SSL_CERT_FILE pointing at conda's CA bundle.
# Our interpreter is the venv (not conda), so that path is wrong and httpx
# (huggingface downloads) fails with FileNotFoundError. Force the venv certifi.
SSL_CERT_FILE="$("$VENV_PY" -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
unset SSL_CERT_DIR

# ---- Shared cache: download datasets & models ONCE, reuse everywhere -------
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export HF_HUB_DISABLE_TELEMETRY=1

# ---- Log file (timestamped) -----------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$ROOT/run_all_${STAMP}.log"

log() {
    local line="[$(date +%H:%M:%S)] $*"
    echo "$line"
    echo "$line" >> "$LOG_FILE"
}

# ---- Notebooks per track, in README dependency order ----------------------
plm_nbs=(
    plm/plm_l1_embeddings_probe.ipynb
    plm/plm_l2_zero_shot_variants.ipynb
    plm/plm_l3_finetune_classification.ipynb
    plm/plm_l4_token_classification.ipynb
    plm/plm_l5_model_comparison.ipynb
    plm/plm_l6_attention_contacts.ipynb
    plm/plm_l7_lora_peft.ipynb
    plm/plm_l8_structure_aware.ipynb
    plm/plm_l9_embedding_retrieval.ipynb
    plm/plm_l10_inverse_folding.ipynb
    plm/plm_l11_calibration.ipynb
)
gnn_nbs=(
    gnn/gnn_l1_graphs_from_proteins.ipynb
    gnn/gnn_l2_node_classification.ipynb
    gnn/gnn_l3_graph_classification.ipynb
    gnn/gnn_l4_plm_plus_gnn.ipynb
    gnn/gnn_l5_equivariant_gnn.ipynb
    gnn/gnn_l6_real_structures.ipynb
    gnn/gnn_l7_edge_features.ipynb
    gnn/gnn_l8_contact_prediction.ipynb
    gnn/gnn_l9_knn_graphs.ipynb
    gnn/gnn_l10_oversmoothing.ipynb
    gnn/gnn_l11_interaction_graphs.ipynb
)
capstones_nbs=(
    capstones/capstone_l1_end_to_end_plm_gnn.ipynb
    capstones/capstone_l2_benchmark_suite.ipynb
)
ig_nbs=(
    ig/ig_l1_simple.ipynb
    ig/ig_l2_tiny_network.ipynb
)

# ---- Build the queue from the requested tracks ----------------------------
queue=()
IFS=',' read -ra track_list <<< "$TRACKS"
for t in "${track_list[@]}"; do
    case "$t" in
        plm)       queue+=("${plm_nbs[@]}") ;;
        gnn)       queue+=("${gnn_nbs[@]}") ;;
        capstones) queue+=("${capstones_nbs[@]}") ;;
        ig)        queue+=("${ig_nbs[@]}") ;;
        *) echo "Unknown track: $t" >&2; exit 2 ;;
    esac
done

log "Running ${#queue[@]} notebook(s): tracks = $TRACKS"
log "Shared HF cache: $CACHE_DIR  (datasets & models downloaded once, reused)"
log "Log file: $LOG_FILE"

# ---- Run ------------------------------------------------------------------
declare -a summary
run_start=$(date +%s)
i=0
failed=0
for nb in "${queue[@]}"; do
    i=$((i + 1))
    path="$ROOT/$nb"
    if [[ ! -f "$path" ]]; then
        log "[$i/${#queue[@]}] MISSING  $nb"
        summary+=("missing      0.0s  $nb")
        continue
    fi

    log "[$i/${#queue[@]}] RUN      $nb"
    start=$(date +%s.%N)
    "$VENV_PY" -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout="$TIMEOUT" \
        --ExecutePreprocessor.kernel_name=python3 \
        "$path"
    rc=$?
    end=$(date +%s.%N)
    secs=$(awk "BEGIN {printf \"%.1f\", $end - $start}")

    if [[ $rc -eq 0 ]]; then
        log "    OK       $nb  (${secs} s)"
        summary+=("ok      ${secs}s  $nb")
    else
        log "    FAILED   $nb  (${secs} s)"
        summary+=("failed  ${secs}s  $nb")
        failed=1
        if [[ $CONTINUE_ON_ERROR -eq 0 ]]; then
            log "Aborting (use --continue-on-error to keep going)."
            break
        fi
    fi
done
run_end=$(date +%s)

log ""
log "===== Summary (total $((run_end - run_start)) s) ====="
for row in "${summary[@]}"; do
    log "  $row"
done

[[ $failed -eq 1 ]] && exit 1 || exit 0
