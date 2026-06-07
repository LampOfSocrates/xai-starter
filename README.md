# Protein ML — Hands-On Lessons

Lesson tracks: **pLMs** (Protein Language Models), **GNNs** (Graph Neural
Networks), **IG** (Integrated Gradients — model attribution), and two
cross-cutting **capstones**. Each track lives in its own folder (`plm/`, `gnn/`,
`ig/`, `capstones/`) with its own README index.

Every lesson is a self-contained **Jupyter notebook** (`*.ipynb`): explanation in
markdown, runnable code below, ending with a `Things to experiment with:` block.
**Run the cells top to bottom** (*Run All*) — they build on each other.

> The original `*.py` scripts each notebook was generated from are archived under
> [to_be_deleted/](to_be_deleted/). The notebooks are now canonical; the scripts
> are kept only for reference. Regenerate a notebook from a script with
> `python _py_to_notebook.py <path/to/script.py>`.

The pLM and GNN tracks intersect in `gnn/gnn_l4` (ESM-2 embeddings as GNN node
features) and again in the capstones (joint end-to-end training).

> **This repo is also the lab bench for a research project.** The reading list,
> paper notes, and project plan live in the Obsidian research vault at
> `G:\My Drive\ObsidianDB\vault\obsidian\xAI for DL in Proteins` (the **research
> docs** — treat them as the source of truth for the science). The active
> direction is **Idea 4 — cross-method xAI for PPI classification, validated
> against SKEMPI v2.0 alanine-scanning hotspots**. See
> [Research track — Idea 4](#research-track--cross-method-xai-for-ppi-idea-4) below.

## Setup

```powershell
# from this directory
.\.venv\Scripts\Activate.ps1     # if not already activated
pip install -r requirements.txt
```

The first lesson you run will download the chosen model (~30 MB for ESM-2 8M)
and the dataset. Both cache locally — subsequent runs are fast.

## pLM track — [plm/README.md](plm/README.md)

| Notebook | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [plm_l1_embeddings_probe](plm/plm_l1_embeddings_probe.ipynb) | Frozen pLM as a feature extractor + sklearn classifier on top. Cheapest way to use a pLM. | ~2-5 min | No |
| [plm_l2_zero_shot_variants](plm/plm_l2_zero_shot_variants.ipynb) | Score mutations using the masked-LM head. NO training required. | ~1 min | No |
| [plm_l3_finetune_classification](plm/plm_l3_finetune_classification.ipynb) | Fine-tune end-to-end for sequence classification with HF Trainer. | ~10-30 min | Helpful |
| [plm_l4_token_classification](plm/plm_l4_token_classification.ipynb) | Per-residue prediction (e.g. secondary structure). Label-to-token alignment. | ~15-30 min | Helpful |
| [plm_l5_model_comparison](plm/plm_l5_model_comparison.ipynb) | Grid of {models} × {pooling strategies}. Outputs CSV. | ~10-20 min | Helpful |
| [plm_l6_attention_contacts](plm/plm_l6_attention_contacts.ipynb) | ESM attention heads recover residue contacts, unsupervised (Rao 2020). | ~2-5 min | No |
| [plm_l7_lora_peft](plm/plm_l7_lora_peft.ipynb) | Parameter-efficient fine-tuning with LoRA — train <1% of params. | ~10-20 min | Helpful |
| [plm_l8_structure_aware](plm/plm_l8_structure_aware.ipynb) | Structure-aware pLMs (ProstT5 / SaProt) and the 3Di structural alphabet. | ~5-10 min | Helpful |
| [plm_l9_embedding_retrieval](plm/plm_l9_embedding_retrieval.ipynb) | Embedding-space geometry (PCA/t-SNE) + nearest-neighbour homology search. | ~3-5 min | No |
| [plm_l10_inverse_folding](plm/plm_l10_inverse_folding.ipynb) | Generative design via masked-LM sampling; native-sequence recovery. | ~3-5 min | No |
| [plm_l11_calibration](plm/plm_l11_calibration.ipynb) | Reliability diagrams, ECE, and temperature scaling. | ~3-5 min | No |

Consolidated tutorial: [plm_tutorial.ipynb](plm/plm_tutorial.ipynb).

## GNN track — [gnn/README.md](gnn/README.md)

| Notebook | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [gnn_l1_graphs_from_proteins](gnn/gnn_l1_graphs_from_proteins.ipynb) | Represent a protein as a PyG Data object: sequence vs contact graph. Visualises both. **+ interactive PDB explorer.** | <1 min | No |
| [gnn_l2_node_classification](gnn/gnn_l2_node_classification.ipynb) | A 2-layer GCN for per-residue prediction, head-to-head vs an MLP baseline. | ~2 min | No |
| [gnn_l3_graph_classification](gnn/gnn_l3_graph_classification.ipynb) | Whole-protein prediction with GAT + global pool, on DeepSol. | ~5-15 min | Helpful |
| [gnn_l4_plm_plus_gnn](gnn/gnn_l4_plm_plus_gnn.ipynb) | The bridge: ESM-2 embeddings as node features for the GNN of `gnn_l3`. | ~10-25 min | Helpful |
| [gnn_l5_equivariant_gnn](gnn/gnn_l5_equivariant_gnn.ipynb) | Build a minimal EGNN from scratch; numerically verify equivariance. | <1 min | No |
| [gnn_l6_real_structures](gnn/gnn_l6_real_structures.ipynb) | Real PDB structures + contact maps (no more synthetic helix). | ~1-2 min | No |
| [gnn_l7_edge_features](gnn/gnn_l7_edge_features.ipynb) | Edge features & geometry (distances, directions) with NNConv. | ~3-5 min | No |
| [gnn_l8_contact_prediction](gnn/gnn_l8_contact_prediction.ipynb) | Edge-level task: predict residue contacts; Precision@L. | ~3-5 min | No |
| [gnn_l9_knn_graphs](gnn/gnn_l9_knn_graphs.ipynb) | k-NN / learned graphs over ESM-2 embeddings when no structure is available. | ~10-20 min | Helpful |
| [gnn_l10_oversmoothing](gnn/gnn_l10_oversmoothing.ipynb) | Oversmoothing vs depth; residual / jumping-knowledge fixes. | ~3-5 min | No |
| [gnn_l11_interaction_graphs](gnn/gnn_l11_interaction_graphs.ipynb) | Heterogeneous / bipartite graphs; protein–drug link prediction. | ~2-4 min | No |

Consolidated tutorial: [gnn_tutorial.ipynb](gnn/gnn_tutorial.ipynb).

## Capstones — [capstones/README.md](capstones/README.md)

| Notebook | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [capstone_l1_end_to_end_plm_gnn](capstones/capstone_l1_end_to_end_plm_gnn.ipynb) | Train ESM-2 **and** a GNN jointly (gradients into the pLM) vs the frozen baseline. | ~15-40 min | Recommended |
| [capstone_l2_benchmark_suite](capstones/capstone_l2_benchmark_suite.ipynb) | Honest evaluation: train/val/test splits, early stopping, multi-seed mean±std. | ~10-25 min | Helpful |

## IG track (Integrated Gradients)

Attribution / explainability: "which inputs drove the model's output?"

| Notebook | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [ig_l1_simple](ig/ig_l1_simple.ipynb) | IG from scratch on a tiny known function: the path integral of gradients, the completeness axiom, why IG beats a plain gradient. | <1 sec | No |
| [ig_l2_tiny_network](ig/ig_l2_tiny_network.ipynb) | IG on a small trained network. | <1 min | No |

## Research track — Cross-method xAI for PPI (Idea 4)

The lessons above are the building blocks; this is what they build toward.

**Research docs (source of truth):** `G:\My Drive\ObsidianDB\vault\obsidian\xAI for DL in Proteins`

| Doc | What it gives you |
|---|---|
| `00_SUMMARY_xAI_for_DL_in_Proteins.md` | The student's guide: IG, protein representations, the 7 papers, 6 project ideas, and **Part 7 — why Idea 4 is the pick**. Read this first. |
| `June 2026 Paper Creation process.md` | The concrete first-three-weeks de-risking plan + task taxonomy. |
| `03_Sendin_2025_..._PPI_Geometric_DL_Explainability.md` | The pipeline we re-implement: Struct2Graph (GCN + mutual attention) on PINDER; the 0.97→0.78 F1 interface-ablation result (shortcut learning). |
| `02_Sundararajan_..._Integrated_Gradients_...md` | IG axioms + recipe (the `ig/` lessons implement this). |
| `04_Fazel_2025_...md` | Nine attribution methods benchmarked on pLMs — no single method wins (the motivation for *consensus*). |
| `01_Hunklinger_Ferruz_2025_...`, `05_Chakraborty...`, `06_Sun2025...`, `07_Johnston_2023...` | Field map, applied templates, and engineering context. |

### The hypothesis

> Combining **three** xAI methods — mutual attention, Integrated Gradients on the
> GCN node embeddings, and GNNExplainer on the graph — isolates PPI-causal
> residues **better than any single method**, as measured by agreement with
> alanine-scanning ΔΔG hotspots (ΔΔG ≥ 2 kcal/mol) in **SKEMPI v2.0**.

Sharp and falsifiable. Every paper in the folder flags this exact missing
experiment (Sendin's own future-work list; Hunklinger & Ferruz's call for
cross-method validation; Fazel's "no single method wins").

### Pipeline → which lessons cover each piece

| Stage | Lesson(s) to study first | New work for the project |
|---|---|---|
| Protein → residue graph (C-α nodes, distance-threshold edges) | [gnn_l1](gnn/gnn_l1_graphs_from_proteins.ipynb), [gnn_l6](gnn/gnn_l6_real_structures.ipynb), [gnn_l7](gnn/gnn_l7_edge_features.ipynb) | PINDER loader; 9.5 Å edge threshold; per-interface 4/6/8/10 Å ablation graphs |
| GCN encoder + mutual attention over a protein **pair** | [gnn_l3](gnn/gnn_l3_graph_classification.ipynb), [gnn_l11](gnn/gnn_l11_interaction_graphs.ipynb) (pair/bipartite), [plm_l6](plm/plm_l6_attention_contacts.ipynb) (attention as signal) | Struct2Graph shared-weight twin-GCN + mutual-attention head; multi-class Cluster-ID output |
| Optional pLM node features | [gnn_l4](gnn/gnn_l4_plm_plus_gnn.ipynb), [capstone_l1](capstones/capstone_l1_end_to_end_plm_gnn.ipynb) | ESM-2 embeddings as node features (Sendin future-work #2) |
| **xAI method 1 — attention** | [plm_l6](plm/plm_l6_attention_contacts.ipynb) | Extract + project mutual-attention coefficients |
| **xAI method 2 — Integrated Gradients** | [ig_l1](ig/ig_l1_simple.ipynb), [ig_l2](ig/ig_l2_tiny_network.ipynb) | IG on the post-GCN node-embedding matrix via Captum's `to_captum_model` (avoids attributing through raw message passing) |
| **xAI method 3 — GNNExplainer** | GNN track (PyG `GNNExplainer`) | Run PyG's built-in explainer on the trained GCN |
| Consensus + benchmark | [capstone_l2](capstones/capstone_l2_benchmark_suite.ipynb) (honest eval) | Rank-aggregate the three saliencies; correlate consensus residues vs SKEMPI ΔΔG |
| 3D visualisation | — | PyMOL renders of high-saliency residues vs measured hotspots |

### Datasets this track needs (beyond the lessons' HF sets)

| Name | What it gives | URL |
|---|---|---|
| **PINDER** | 2.3 M PPIs with interface Cluster IDs (the training set) | github.com/pinder-org/pinder |
| **SKEMPI v2.0** | ~7,000 mutation-induced ΔΔG values across 345 PPI complexes (the ground truth) | life.bsc.es/pid/skempi2 |
| **PyMOL** | 3D visualisation of saliency vs hotspots | `pip install pymol-open-source` |

`captum` and `torch_geometric` are already in [requirements.txt](requirements.txt).

### De-risking plan (first three weeks, from the research docs)

1. **Week 1** — Clone PINDER, replicate Sendin's 31-cluster training run, confirm ~0.97 F1 reproduces.
2. **Week 2** — IG on the GCN node embeddings (Captum `to_captum_model`); GNNExplainer via PyG.
3. **Week 3** — Pull SKEMPI v2.0, intersect with PINDER complexes, first plot of *consensus high-saliency residues vs measured ΔΔG ≥ 2 kcal/mol*.

> Go/no-go: if by week 3 IG produces sensible per-residue attributions on a single
> complex, the project is on track. If not, fall back to **Idea 1** (BioLiP2 +
> ESM-2 ligand-binding-site faithfulness benchmark) — same shape, flatter compute curve.

## Models you can swap in (pLMs)

ESM-2 (Meta AI), all on Hugging Face. The number after `t` is the layer count:

| Model ID | Params | Embed dim | Notes |
|---|---|---|---|
| `facebook/esm2_t6_8M_UR50D` | 8M | 320 | Default. CPU-friendly. |
| `facebook/esm2_t12_35M_UR50D` | 35M | 480 | Better, still CPU-OK. |
| `facebook/esm2_t30_150M_UR50D` | 150M | 640 | GPU recommended. |
| `facebook/esm2_t33_650M_UR50D` | 650M | 1280 | GPU required. |
| `facebook/esm1v_t33_650M_UR90S_1` | 650M | 1280 | Specialised for variant effect (`plm_l2`). |

Other families:
- **ProtBERT** — `Rostlab/prot_bert` (tokenizer expects spaces between amino acids).
- **ProtT5** — `Rostlab/prot_t5_xl_uniref50` (encoder-decoder; use only the encoder for embeddings).
- **Ankh** — `ElnaggarLab/ankh-base` / `ankh-large`.

## GNN building blocks

All from `torch_geometric.nn`. Every lesson uses one of these:

| Layer | What it does |
|---|---|
| `GCNConv` | Kipf & Welling 2017 — normalised mean of neighbours. The default GNN. |
| `GATConv` | Veličković 2018 — learns attention weights per edge (multi-head). |
| `SAGEConv` | Hamilton 2017 — concatenates [self, mean(neighbours)] then projects. Inductive. |
| `GINConv` | Xu 2019 — provably more expressive on graph isomorphism. |
| Custom EGNN | Satorras 2021 — equivariant to 3D rotations/translations (`gnn_l5`). |

Pooling for graph-level tasks: `global_mean_pool`, `global_max_pool`, `global_add_pool`.

## Datasets — where to get them

### Hugging Face (just `load_dataset(...)`)

| Name | Task | Type | Used in |
|---|---|---|---|
| `zhanglab/DeepSol` | Solubility | Binary classification | plm_l1, plm_l3, plm_l5, gnn_l3, gnn_l4 |
| `proteinea/solubility` | Solubility | Binary classification | Alt. for above |
| `proteinea/localization` | Subcellular localization | Multi-class (10) | Try in plm_l3 |
| `proteinea/fluorescence` | GFP fluorescence | Regression | Try in plm_l3 |
| `proteinea/stability` | Thermal stability | Regression | Try in plm_l3 |
| `proteinea/secondary_structure` | SS3 / SS8 | Token classification | plm_l4 |

### Bigger benchmark suites

- **ProteinGym** — variant-effect benchmarks. https://proteingym.org/
- **FLIP** — protein engineering benchmarks (GB1, AAV, meltome, etc.). https://github.com/J-SNACKKB/FLIP
- **TAPE** — older but classic. SS, contact prediction, fluorescence. https://github.com/songlab-cal/tape
- **PEER** — broader pLM benchmark. https://github.com/DeepGraphLearning/PEER_Benchmark

### For 3D structure

- **PDB** — https://www.rcsb.org/ — download structures directly.
- **AlphaFold DB** — https://alphafold.ebi.ac.uk/ — predicted structures for ~all of UniProt.
- **biopython** — `pip install biopython` for parsing PDBs into Python objects.

### Roll your own

- **UniProt** — search and download FASTA: https://www.uniprot.org/

## Quick start

```powershell
pip install -r requirements.txt
jupyter lab          # then open any lesson notebook and Run All
```

Gentlest starting points (no GPU, minimal downloads):
- [gnn_l1_graphs_from_proteins](gnn/gnn_l1_graphs_from_proteins.ipynb) — builds + visualises a graph; ends with an interactive PDB explorer.
- [ig_l1_simple](ig/ig_l1_simple.ipynb) — IG math from scratch, no downloads.
- [plm_l1_embeddings_probe](plm/plm_l1_embeddings_probe.ipynb) — pLM + sklearn.

If a dataset throws a column-name error, run the early cells — every lesson
prints the dataset's features and tells you which constants to adjust.

> `plm/plm_demo.ipynb` is the original starter and overlaps with `plm_l3`.
> Safe to skip once you're comfortable with the lessons.
