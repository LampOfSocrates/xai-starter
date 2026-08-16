# LATEST.md

## What this is
A hands-on Jupyter-notebook curriculum on explainable AI (xAI) for protein deep learning — pLM (protein language model), GNN, and capstone lesson tracks — plus three parallel MSc-scale research tracks (`pinch`, `clasp`, `edits`) built on a shared `common/` library, investigating attention/Integrated-Gradients/GNNExplainer consensus methods for PPI binding-hotspot attribution.

## Where it runs
Local dev (Windows, `.venv`, pip-installable via `pyproject.toml`); heavier pLM notebooks run locally on GPU via the `experiments/` papermill runner. Lesson outputs are committed and published as a static Quarto site on GitHub Pages: https://lampofsocrates.github.io/xai-starter/ (CI only renders, never executes).

## Features
- Self-contained lesson notebooks across `plm/`, `gnn/`, `capstones/`, teaching pLM probing, zero-shot variant scoring, fine-tuning, LoRA/PEFT, attention/contacts, and GNN node/graph classification.
- Shared MLflow experiment-tracking harness (`mlflow_utils.py`, `experiments/`) with a papermill runner requiring a `--reason` tag per run.
- Three scaffolded research tracks (`pinch/`, `clasp/`, `edits/`), each 5 notebooks, running end-to-end on miniature datasets (barnase–barstar, BPTI, GB1).
- Quarto-built static site publishing all notebooks with saved outputs to GitHub Pages.

## Recently tried
- 2026-06-14: GPU campaign wiring for pLM (`run_plm_gpu.sh`, `_wire_plm_gpu.py`) for long-running GPU runs.
- 2026-06-14: Rewrote LoRA math in `plm_l7` with proper notation and citation (Hu et al. 2021).
- 2026-06-14: Fixed a Quarto rendering bug where markdown cells stored as a single string collapsed into one bold heading instead of rendering properly.
- 2026-06-14: Added `plm_l3b` multi-class localization notebook, ProteinGym DMS benchmark in `plm_l2`, DRY'd MLflow setup, made repo pip-installable.
- 2026-06-08 to 2026-06-14 (earlier): Reorganised research into `pinch`/`clasp`/`edits` tracks over shared `common/`; published lesson suite as Quarto site on GitHub Pages.

## Next
- attrib-PINCH: scale the SKEMPI v2.0 consensus benchmark (`pinch_l5`) beyond the barnase–barstar miniature to the full PINDER↔SKEMPI intersection (per README's "Week 3" plan; inferred still open).
- Run/monitor the GPU pLM campaign now wired up (`experiments/run_plm_gpu.sh`) and report results via `mlflow_report.py`.
- `clasp`/`edits` tracks are still miniature-only (BPTI/GB1) — likely next to scale similarly to `pinch` (inferred).
