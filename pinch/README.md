# `pinch/` — attrib-PINCH: cross-method xAI for PPI hotspots

Re-implements and extends the Sendin (2025) PINDER pipeline to test one
falsifiable hypothesis:

> The **consensus** of three explainability methods — mutual attention,
> Integrated Gradients on the GCN node embeddings, and GNNExplainer — isolates
> PPI-causal residues **better than any single method**, measured by agreement
> with alanine-scanning ΔΔG hotspots (ΔΔG ≥ 2 kcal/mol) in **SKEMPI v2.0**.

> **Main document (source of truth):**
> `G:\My Drive\ObsidianDB\vault\obsidian\xAI for DL in Proteins\00_xAI_for_DL_in_Proteins_Definitive_Guide.md`
> — project **#1 `attrib-PINCH`** (Part 6) and **Part 7** (why it is the pick +
> the first-three-weeks de-risking plan). See also `CLAUDE.md` here for the
> standing data-leakage constraint.

## Notebooks (run top to bottom; miniature = barnase–barstar `1BRS`)

| Notebook | What it does |
|---|---|
| [pinch_l1_pinder_graphs](pinch_l1_pinder_graphs.ipynb) | A PPI complex → two residue contact graphs + interface mask (4/6/8/10 Å). |
| [pinch_l2_struct2graph_train](pinch_l2_struct2graph_train.ipynb) | Struct2Graph (shared-weight GCN + mutual attention); train the miniature PPI classifier. |
| [pinch_l3_ig_node_embeddings](pinch_l3_ig_node_embeddings.ipynb) | IG on the post-GCN node embeddings via Captum; completeness + the saturation caveat. |
| [pinch_l4_attention_gnnexplainer](pinch_l4_attention_gnnexplainer.ipynb) | Mutual-attention and GNNExplainer saliencies; do they agree? |
| [pinch_l5_skempi_consensus_benchmark](pinch_l5_skempi_consensus_benchmark.ipynb) | SKEMPI v2.0 hotspots; consensus vs single methods by AUROC — the hypothesis test. |

## Engine & harness

- [`pinch_common.py`](pinch_common.py) — the tested engine all five notebooks
  import. Builds on the shared [`common/`](../common/) primitives; adds
  `Struct2Graph`, the miniature PINDER stand-in dataset, the SKEMPI ground truth,
  and `compute_all_saliencies` / `benchmark_hotspots` / `run_experiment`.
- [`_build_pinch_notebooks.py`](_build_pinch_notebooks.py) — regenerates the five
  notebooks from the engine. [`_smoke_pinch.py`](_smoke_pinch.py) — fast
  end-to-end pipeline check.
- [`run_pinch.py`](run_pinch.py) / [`report_pinch.py`](report_pinch.py) — the
  parameterised runner + tag-level MLflow reporter (scale past the miniature
  without editing notebooks). See the repo README "Parameterised runs" section.

## Datasets beyond the lessons' HF sets

| Name | What it gives | URL |
|---|---|---|
| **PINDER** | 2.3 M PPIs with interface cluster IDs (training set) | github.com/pinder-org/pinder |
| **SKEMPI v2.0** | ~7,000 mutation-induced ΔΔG values (ground truth) | life.bsc.es/pid/skempi2 |
| **PyMOL** | 3D visualisation of saliency vs hotspots | `pip install pymol-open-source` |

## New work to scale up

PINDER loader + per-interface ablation graphs; train on PINDER's 31-cluster holo
subset; ESM-2 embeddings as node features; aggregate AUROC across all
PINDER↔SKEMPI complexes with CIs. Wire these through `pinch_common.run_experiment`
(`--dataset pinder` is reserved and currently raises) — not by editing notebooks.
