# Protein ML — Hands-On Lessons

Two parallel lesson tracks: **pLMs** (Protein Language Models) and **GNNs**
(Graph Neural Networks). Each lesson is a self-contained Python script.
Heavy comments throughout. Every script ends with a `Things to experiment
with:` block to keep you going.

The two tracks intersect in `gnn_l4`, where ESM-2 embeddings become node
features for a GNN.

## Setup

```powershell
# from this directory
.\.venv\Scripts\Activate.ps1     # if not already activated
pip install -r requirements.txt
```

The first lesson you run will download the chosen model (~30 MB for ESM-2 8M)
and the dataset. Both cache locally — subsequent runs are fast.

## pLM track

| File | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [plm_l1_embeddings_probe.py](plm_l1_embeddings_probe.py) | Frozen pLM as a feature extractor + sklearn classifier on top. Cheapest way to use a pLM. | ~2-5 min | No |
| [plm_l2_zero_shot_variants.py](plm_l2_zero_shot_variants.py) | Score mutations using the masked-LM head. NO training required. | ~1 min | No |
| [plm_l3_finetune_classification.py](plm_l3_finetune_classification.py) | Fine-tune end-to-end for sequence classification with HF Trainer. | ~10-30 min | Helpful |
| [plm_l4_token_classification.py](plm_l4_token_classification.py) | Per-residue prediction (e.g. secondary structure). Label-to-token alignment. | ~15-30 min | Helpful |
| [plm_l5_model_comparison.py](plm_l5_model_comparison.py) | Grid of {models} × {pooling strategies}. Outputs CSV. | ~10-20 min | Helpful for big models |

Notebook: [plm_tutorial.ipynb](plm_tutorial.ipynb).

## GNN track

| File | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [gnn_l1_graphs_from_proteins.py](gnn_l1_graphs_from_proteins.py) | Represent a protein as a PyG Data object: sequence graph vs contact graph. Visualises both. | <1 min | No |
| [gnn_l2_node_classification.py](gnn_l2_node_classification.py) | A 2-layer GCN for per-residue prediction. Compared head-to-head against an MLP baseline. | ~2 min | No |
| [gnn_l3_graph_classification.py](gnn_l3_graph_classification.py) | Whole-protein prediction with GAT + global pool, on DeepSol. | ~5-15 min | Helpful |
| [gnn_l4_plm_plus_gnn.py](gnn_l4_plm_plus_gnn.py) | The bridge: ESM-2 embeddings as node features for the GNN of `gnn_l3`. | ~10-25 min | Helpful |
| [gnn_l5_equivariant_gnn.py](gnn_l5_equivariant_gnn.py) | Build a minimal EGNN from scratch. Numerically verify rotation/translation equivariance. | <1 min | No |

Notebook: [gnn_tutorial.ipynb](gnn_tutorial.ipynb).

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
python plm_l1_embeddings_probe.py    # easiest: pLM + sklearn
python gnn_l1_graphs_from_proteins.py   # easiest: just builds + visualises a graph
```

If a dataset throws a column-name error, open the lesson — every script
prints the dataset's features and tells you which constants to adjust.

> `plm_demo.py` is the original starter and overlaps with `plm_l3`.
> Safe to delete once you're comfortable with the lessons.
