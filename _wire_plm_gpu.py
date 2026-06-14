"""Wire the GPU-friendly pLM lessons for the long-running GPU campaign.

Idempotent, re-runnable (mirrors `_wire_gnn.py`). For each target notebook it:

  1. Lifts ``EPOCHS`` into the ``parameters``-tagged cell so papermill can
     override it (the fine-tune lessons hard-coded EPOCHS inside the training
     cell, which clobbered any override). Only l3 / l3b / l4 need this; l7
     already declares EPOCHS in its parameters cell.
  2. Appends a "run comparison" cell (tagged ``run-comparison``) that plots this
     lesson's MLflow history via ``mu.plot_run_comparison(...)``. The three
     inference lessons (l2 / l6 / l10) never logged to MLflow, so their appended
     cell first logs the lesson's headline metric, then plots the comparison.

Run from the repo root:  .venv/Scripts/python.exe _wire_plm_gpu.py
"""
import os
import re

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
PLM = os.path.join(HERE, "plm")
MARKER = "run-comparison"

# Fine-tune lessons whose EPOCHS must move into the parameters cell, with the
# default to declare there (matches the value they hard-coded before).
EPOCH_LIFT = {
    "plm_l3_finetune_classification.ipynb": 3,
    "plm_l3b_multiclass_localization.ipynb": 3,
    "plm_l4_token_classification.ipynb": 5,
}

# Lessons that already log to MLflow -> the appended cell only needs to chart.
COMPARE_ONLY = {
    "plm_l3_finetune_classification.ipynb": "plm-solubility",
    "plm_l3b_multiclass_localization.ipynb": "plm-localization",
    "plm_l4_token_classification.ipynb": "plm-secondary-structure",
    "plm_l5_model_comparison.ipynb": "plm-model-comparison",
    "plm_l7_lora_peft.ipynb": "plm-solubility",
}

# Inference lessons that don't log yet -> the appended cell logs, then charts.
# Each snippet reuses module-level functions/constants already defined above it.
LOG_THEN_COMPARE = {
    "plm_l2_zero_shot_variants.ipynb": '''\
# --- Run comparison -----------------------------------------------------------
# Log this run's zero-shot Spearman (per scheme) to the shared MLflow store,
# then chart how it compares to every prior run of this lesson. Re-run with a
# bigger model (ESM-1v 650M) to watch the correlation climb across runs.
import mlflow_utils as mu
mlflow = mu.setup_mlflow()

_schemes = {"spearman_masked": s_masked, "spearman_wt": s_wt, "spearman_pll": s_pll}
_metrics = {k: r for k, sc in _schemes.items()
            for r, _ in [_spearman(sc)] if r == r}      # drop NaN schemes
if _metrics:
    _metrics["spearman_best"] = max(_metrics.values())
    with mu.run("plm-variant-effect", f"l2_{MODEL_NAME.split('/')[-1]}",
                params={"model": MODEL_NAME, "assay": str(assay), "real_data": bool(real)},
                tags={"lesson": "plm_l2"}):
        mlflow.log_metrics(_metrics)
    print("logged:", {k: round(v, 3) for k, v in _metrics.items()})

mu.plot_run_comparison("plm-variant-effect")
''',
    "plm_l6_attention_contacts.ipynb": '''\
# --- Run comparison -----------------------------------------------------------
# main() keeps its metrics local, so recompute the headline contact-recovery
# numbers self-containedly (reusing this lesson's functions), log them, then
# chart this run against prior ones. Re-run with a bigger model to compare.
import mlflow_utils as mu
mlflow = mu.setup_mlflow()

_device = "cuda" if torch.cuda.is_available() else "cpu"
_seq, _ca = make_synthetic_structure()
_L = len(_seq)
_true = compute_contact_map(_ca)
_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
_mdl = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(_device).eval()
_attn = get_attention_maps(_seq, _mdl, _tok, _device)
_scored = sorted(
    (precision_at_L(_attn[l, h], _true, _L), auc_score(_attn[l, h], _true, _L), l, h)
    for l in range(_attn.shape[0]) for h in range(_attn.shape[1])
)
_best = _scored[-1]
_avg = np.stack([_attn[l, h] for *_rest, l, h in _scored[-TOP_K_HEADS:]]).mean(0)
_metrics = {
    "best_prec_at_l": float(_best[0]), "best_auc": float(_best[1]),
    "avg_prec_at_l": float(precision_at_L(_avg, _true, _L)),
    "avg_auc": float(auc_score(_avg, _true, _L)),
}
with mu.run("plm-contacts", f"l6_{MODEL_NAME.split('/')[-1]}",
            params={"model": MODEL_NAME, "seq_len": _L, "top_k_heads": TOP_K_HEADS},
            tags={"lesson": "plm_l6"}):
    mlflow.log_metrics(_metrics)
print("logged:", {k: round(v, 3) for k, v in _metrics.items()})

mu.plot_run_comparison("plm-contacts")
''',
    "plm_l10_inverse_folding.ipynb": '''\
# --- Run comparison -----------------------------------------------------------
# Log native-sequence recovery to the shared MLflow store, then chart this run
# against prior ones. Self-contained recompute (main() keeps its metrics local).
import mlflow_utils as mu
mlflow = mu.setup_mlflow()

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
_mdl = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(_device).eval()
_rec, _ = native_sequence_recovery(WILD_TYPE, _mdl, _tok, _device)
with mu.run("plm-inverse-folding", f"l10_{MODEL_NAME.split('/')[-1]}",
            params={"model": MODEL_NAME, "seq_len": len(WILD_TYPE)},
            tags={"lesson": "plm_l10"}):
    mlflow.log_metrics({"recovery_pct": float(_rec)})
print(f"logged recovery_pct={_rec:.1f}")

mu.plot_run_comparison("plm-inverse-folding")
''',
}


def _params_cell(nb):
    for cell in nb.cells:
        if cell.cell_type == "code" and "parameters" in cell.get("metadata", {}).get("tags", []):
            return cell
    return None


def _has_marker(nb):
    return any(MARKER in c.get("metadata", {}).get("tags", []) for c in nb.cells)


def _compare_only_cell(experiment):
    return (
        "# --- Run comparison -----------------------------------------------------------\n"
        "# Every run of this lesson logs to the shared MLflow store. This chart shows\n"
        "# how the latest run compares to all prior ones (e.g. more epochs / a bigger\n"
        "# model). It accumulates as you re-run; a no-op on a fresh checkout.\n"
        "import mlflow_utils as mu\n"
        f"mu.plot_run_comparison({experiment!r})\n"
    )


def wire(fname):
    path = os.path.join(PLM, fname)
    nb = nbformat.read(path, as_version=4)
    changed = False

    # 1. Lift EPOCHS into the parameters cell + stop the training cell clobbering it.
    if fname in EPOCH_LIFT:
        pcell = _params_cell(nb)
        if pcell is not None and not re.search(r"^\s*EPOCHS\s*=", pcell.source, re.M):
            line = f"EPOCHS = {EPOCH_LIFT[fname]}"
            if "setup_mlflow" in pcell.source:
                pcell.source = re.sub(r"\n([^\n]*setup_mlflow[^\n]*)$",
                                      f"\n{line}\n\\1", pcell.source)
            else:
                pcell.source = pcell.source.rstrip() + "\n" + line + "\n"
            changed = True
        for cell in nb.cells:
            if cell.cell_type == "code" and re.search(r"EPOCHS,\s*LR,\s*BATCH,\s*WD\s*=", cell.source):
                cell.source = re.sub(r"EPOCHS,\s*LR,\s*BATCH,\s*WD\s*=\s*\d+,\s*",
                                     "LR, BATCH, WD = ", cell.source)
                changed = True

    # 2. Append the run-comparison cell (once).
    if not _has_marker(nb):
        if fname in LOG_THEN_COMPARE:
            src = LOG_THEN_COMPARE[fname]
        else:
            src = _compare_only_cell(COMPARE_ONLY[fname])
        cell = nbformat.v4.new_code_cell(src)
        cell.metadata["tags"] = [MARKER]
        # nbformat v4 cells carry no execution outputs until run.
        nb.cells.append(nbformat.v4.new_markdown_cell(
            "## Run comparison — this lesson's MLflow history\n\n"
            "How this run compares to every prior run of the same lesson "
            "(bigger model / more epochs accumulate here as you re-run). "
            "Generated from the shared local MLflow store."))
        nb.cells.append(cell)
        changed = True

    if changed:
        nbformat.write(nb, path)
    return changed


def main():
    targets = sorted(set(EPOCH_LIFT) | set(COMPARE_ONLY) | set(LOG_THEN_COMPARE))
    for fname in targets:
        did = wire(fname)
        print(f"[{'wired' if did else 'ok   '}] {fname}")


if __name__ == "__main__":
    main()
