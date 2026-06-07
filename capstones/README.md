# Capstones

Cross-cutting lessons that combine the pLM and GNN tracks. Do these after the
core lessons in [../plm/](../plm/README.md) and [../gnn/](../gnn/README.md).
**Run the cells top to bottom** (*Run All*).

| # | Notebook | You'll learn |
|---|----------|--------------|
| 1 | [End-to-end pLM + GNN](capstone_l1_end_to_end_plm_gnn.ipynb) | Train ESM-2 **and** a GAT jointly — gradients flow into the pLM — and compare against the frozen-pLM baseline of `gnn_l4`. Differential learning rates, the memory/compute cost of unfreezing. |
| 2 | [Honest benchmarking](capstone_l2_benchmark_suite.ipynb) | Evaluate properly: train/val/test splits, early stopping on validation, multi-seed mean±std, a clean results table. Why a baseline and variance reporting matter. |

## Running

```bash
pip install -r requirements.txt
jupyter lab            # then open a capstone notebook and Run All
```

Both download DeepSol + ESM-2 on first run and keep `N` small so they finish on
CPU; a GPU is recommended for capstone 1 (it backprops through the pLM).

Generated from the scripts archived in
[../to_be_deleted/capstones/](../to_be_deleted/capstones/).
