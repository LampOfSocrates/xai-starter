# Experiments

Running log of MLflow experiment ideas for the pLM + GNN lesson stack. Each
overnight batch is tagged with a single **`campaign`** tag (e.g.
`campaign="overnight-2026-06-08"`) so a report can be generated at the tag
level. The teaching notebooks stay small; "more data" runs live in standalone
sweep scripts that import the notebook model/training code.

## Workflow

1. **Run** via the gated papermill driver (see `experiments/README.md`):
   `experiments/exp_C_benchmark.sh --campaign overnight-2026-06-08 --n-train 5000 ...`
   Each run is gated on CPU/GPU and tagged with `campaign` / `exp_id` /
   `config_hash` / `run_uid` automatically (no notebook changes needed).
2. **Report** with `mlflow_report.py --campaign <campaign-id>` →
   `reports/report_<campaign-id>.md` (per-experiment config tables + best config +
   one accuracy-chart per experiment + each run's `reason`). Add `--interpret` to
   have the `claude` CLI read the JSON summary and write an honest Interpretation
   section. Also emits `reports/summary_<campaign-id>.json` + `reports/figs_<campaign-id>/`.
3. **Tag levels** available for grouping in a report:
   - `campaign` — the overnight batch (top-level report scope)
   - `exp_id` — logical experiment (e.g. `C-sol-benchmark`)
   - `config_hash` — identical params → identical hash (reproducibility groups)
   - `run_uid` — unique per execution (distinguishes reruns)
   - `lesson` — which lesson the run came from (already logged)
   - `model`, `method`, `seed`, `features` — per-run axes for sub-tables

## Conventions

- **Datasets** — default classification task is `proteinea/solubility`
  (binary soluble/insoluble). "Full data" = use the whole train split instead
  of the current 200–500 cap. SS task is `agemagician/NetSurfP-SS3`.
- **Seeds** — report mean ± std over seeds; a result inside ±1 std of the
  majority baseline is "no signal", not "small signal".
- **Compute** — RTX 5070 Ti, 16 GB. Batch size 4–8 for ≤150M; full
  fine-tuning of 650M is the one VRAM-risky item (use grad checkpointing).
- **Resource gate** — overnight runners reuse `_resource_check.py` (wait if
  CPU/GPU > 70%) so they never collide with foreground work.

---

## Experiment catalog

Status legend: ⬜ not started · 🟡 scripted · 🟢 run · ✅ reported

### A — Solubility: model size × adaptation method  ⬜
- **Source**: extends `plm_l1` (probe), `plm_l3` (full FT), `plm_l7` (LoRA)
- **Sweep**: ESM-2 {8M, 35M, 150M, 650M} × {frozen probe, LoRA r=8, full fine-tune}, full train split, 3 seeds → 36 runs
- **Experiment / tags**: `plm-solubility`, tags `model`, `method`, `seed`
- **Question it answers**: *Is a bigger protein language model worth the compute on solubility, and does fine-tuning beat a cheap frozen-embedding probe?* Plots accuracy/F1 against model size for each adaptation method, exposing (a) the size at which returns flatten, (b) whether LoRA recovers most of full-FT's gain at a fraction of the trainable params, and (c) whether any of it clears the majority baseline once N is large. This is the headline "does scale help?" result and the honest counterweight to the current tiny-N notebooks where the probe sometimes ties fine-tuning.
- **Expected time**: ~2–4 h (8M/35M/150M fast; 650M full-FT is the long pole, ~1–2 h alone)
- **Risk**: 650M full-FT VRAM — fall back to LoRA-only for 650M if OOM.

### B — LoRA rank sweep  ⬜
- **Source**: `plm_l7`
- **Sweep**: r ∈ {2, 4, 8, 16, 32}, fixed alpha/r ratio, full train split, 5 seeds → 25 runs
- **Experiment / tags**: `plm-lora-rank`, tags `lora_r`, `seed`
- **Question it answers**: *How small can the LoRA adapter get before accuracy degrades?* Produces the accuracy-vs-trainable-parameters curve and identifies the rank where the curve knees — the practical "use this rank" recommendation. Also shows seed variance at each rank, so you can tell a real rank effect from noise.
- **Expected time**: ~1 h (cheap; only adapters train)

### C — Benchmark suite at scale  ⬜
- **Source**: `capstones/capstone_l2` (already a parameterised 4-model suite)
- **Sweep**: N_train ≈ 5–10k, 5 seeds, add ESM-2 35M & 150M feature variants → ~30 nested runs
- **Experiment / tags**: `plm-solubility-benchmark`, tags `model`, `seed`
- **Question it answers**: *With enough data and seeds, what is the honest ranking of {majority baseline, ESM-2+LogReg, GNN one-hot, GNN+ESM-2 feats} — and are the gaps real?* The current suite runs at N=~100 where rankings are noise. At scale with 5 seeds it gives mean ± std bars, so you can state which models genuinely separate and by how much. Directly answers the capstone's own "is the GNN+pLM bridge worth it?" question.
- **Expected time**: ~1–2 h

### D — Secondary structure: Q3 vs Q8, size scaling  ⬜
- **Source**: `plm_l4`
- **Sweep**: full NetSurfP-SS3, {Q3, Q8} label sets × ESM-2 {8M, 35M}, more epochs → ~4 runs (+ per-step Trainer curves)
- **Experiment / tags**: `plm-secondary-structure`, tags `model`, `task`
- **Question it answers**: *How much does per-residue secondary-structure accuracy improve with model size, and how much harder is 8-state (Q8) than 3-state (Q3)?* Gives the Q3→Q8 difficulty gap and the 8M→35M size lift on a token-level task, complementing the sequence-level solubility story. Trainer auto-logs per-step loss/accuracy so you also get learning curves per run.
- **Expected time**: ~1 h

### E — GNN graph construction  ⬜
- **Source**: `gnn_l9`
- **Sweep**: kNN k ∈ {3, 5, 10, 20}, more proteins, {sequence-window, embedding-kNN, random, (optional) contact} → ~16 runs
- **Experiment / tags**: `gnn-knn-graphs`, tags `graph_type`, `knn_k`
- **Question it answers**: *Does the graph topology you impose on pLM embeddings matter, and is a learned embedding-kNN graph better than a naive sequence-window graph?* Ranks graph constructions on the same GAT and finds the k where embedding-kNN stops helping. The random-graph control quantifies how much of the accuracy is the graph vs. just the node features. Tells you whether investing in structure-aware graphs pays off when no real 3D structure is available.
- **Expected time**: ~30–60 min

### F — GNN oversmoothing depth sweep  ⬜
- **Source**: `gnn_l10`
- **Sweep**: depths {2, 4, 8, 16, 32}, more data, {vanilla GCN, residual GCN, GAT} → ~15 runs
- **Experiment / tags**: `gnn-oversmoothing`, tags `arch`, `depth`
- **Question it answers**: *At what depth does a GCN collapse from oversmoothing, and do residual connections / attention actually delay it?* Plots accuracy, mean pairwise cosine similarity (MPCS), and Dirichlet energy against depth for each architecture — the textbook oversmoothing collapse curve, but measured rather than asserted. Shows whether residual/GAT variants keep node representations distinct deeper than vanilla GCN.
- **Expected time**: ~1 h (trains many models; scales with #depths × #archs)

---

## Suggested overnight batches

| Batch | Contents | Total est. | Rationale |
|---|---|---|---|
| Batch 1 (headline) | A + C | ~4–6 h | The "does scale help solubility?" story with proper seeds/variance |
| Batch 2 (cheap sweeps) | B + E + F | ~3 h | Fast hyperparameter/topology curves, low VRAM |
| Batch 3 (token-level) | D | ~1 h | Secondary-structure scaling, separate dataset |

---

## Run log

| Date | Campaign tag | Experiments | Outcome | Report |
|---|---|---|---|---|
| _2026-06-07_ | _(pilot, untagged)_ | smoke + 14 wired notebooks at default N | 14/14 logged; gnn_l4 < gnn_l3 at tiny N (expected) | — |
| _2026-06-08_ | `gnn-smoke-2026-06-08` | gnn_l3 + gnn_l9, tiny N (pipeline smoke) | both logged; no signal at N=20–40 (expected); validates run→report→interpret | `reports/report_gnn-smoke-2026-06-08.md` |
| | | | | |
