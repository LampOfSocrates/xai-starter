# Protein ML — Hands-On Lessons

Teaching tracks: **pLMs** (Protein Language Models), **GNNs** (Graph Neural
Networks), and two cross-cutting **capstones** — plus **Integrated Gradients**
foundations and the shared research code in [`common/`](common/). Each lives in
its own folder (`plm/`, `gnn/`, `capstones/`, `common/`) with its own README index.

On top of those sit three parallel **research tracks**, each in its own folder
and importing `common/`: [`pinch/`](pinch/) (attrib-PINCH), [`clasp/`](clasp/)
(contact-CLASP) and [`edits/`](edits/) (attrib-EDITS). See
[Research tracks](#research-tracks--three-parallel-projects) below.

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
> direction is **`attrib-PINCH` — cross-method xAI for PPI classification,
> validated against SKEMPI v2.0 alanine-scanning hotspots**. See
> [Research track — attrib-PINCH](#research-track--cross-method-xai-for-ppi-attrib-pinch) below.

## Browse the lessons online

The whole suite is published as a searchable static website (every notebook with
its **saved outputs** — plots, tables, metrics), so you can read the completed
lessons without installing anything:

**https://lampofsocrates.github.io/xai-starter/**

The site is built by [Quarto](https://quarto.org) and deployed to GitHub Pages by
[`.github/workflows/publish.yml`](.github/workflows/publish.yml) on every push
that touches a notebook. Re-execution is **disabled** (`_quarto.yml` →
`execute.enabled: false`): notebooks are run locally on GPU (via the
[`experiments/`](experiments/) papermill runner) and committed with their
outputs, and CI only renders those outputs — no models or GPU in CI.

### View the executed notebooks after cloning

You don't need to run any cells — the outputs are committed. Three ways to read
them locally, cheapest first:

1. **Open a notebook directly** in VS Code / JupyterLab / `nbviewer` — every
   `*.ipynb` already contains its plots and tables.
2. **Build the whole site once** and open it in a browser:
   ```powershell
   # one-time: install the Quarto CLI -> https://quarto.org/docs/get-started/
   #   winget install Posit.Quarto
   $env:QUARTO_PYTHON = ".\.venv\Scripts\python.exe"   # so Quarto can read notebook outputs
   quarto render                                       # writes the site to _site\
   start _site\index.html                              # open the homepage
   ```
3. **Live preview** (auto-reloads as you edit), good for browsing the suite:
   ```powershell
   $env:QUARTO_PYTHON = ".\.venv\Scripts\python.exe"
   quarto preview            # serves at http://localhost:<port> and opens a browser
   ```

`quarto render`/`preview` only **render** the saved outputs (execution is off in
`_quarto.yml`), so they need just the Quarto CLI plus `nbformat` from the
`.venv` — no GPU, torch, or model downloads. The generated `_site/` and
`.quarto/` folders are git-ignored.

> One-time repo setting (maintainer): **Settings → Pages → Build and deployment → Source = GitHub Actions.**

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

## Integrated Gradients — foundations ([common/](common/))

Attribution / explainability: "which inputs drove the model's output?" These two
primers are prerequisites for all three research tracks, so they live in the
shared [`common/`](common/) package next to the code every track reuses.

| Notebook | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [ig_l1_simple](common/ig_l1_simple.ipynb) | IG from scratch on a tiny known function: the path integral of gradients, the completeness axiom, why IG beats a plain gradient. | <1 sec | No |
| [ig_l2_tiny_network](common/ig_l2_tiny_network.ipynb) | IG on a small trained network. | <1 min | No |

## Research tracks — three parallel projects

Three MSc-scale research directions from the Definitive Guide, each in its own
folder with its own README, all importing the shared [`common/`](common/) code:

| Track | Folder | Question | Status |
|---|---|---|---|
| **`attrib-PINCH`** | [pinch/](pinch/) | Does the *consensus* of attention + IG + GNNExplainer beat any single method at finding PPI binding hotspots? | scaffolded — 5 notebooks, runs end-to-end on a miniature |
| **`contact-CLASP`** | [clasp/](clasp/) | Does ESM attention recover contacts like DCA, and does IG fusion help as Neff/L drops? | planned — see [clasp/README.md](clasp/README.md) |
| **`attrib-EDITS`** | [edits/](edits/) | What is the smallest realistic edit that flips a stability/fitness predictor — and does it match real ΔΔG? | planned — see [edits/README.md](edits/README.md) |

### attrib-PINCH notebooks ([pinch/](pinch/))

| Notebook | What it teaches | CPU runtime | GPU? |
|---|---|---|---|
| [pinch_l1_pinder_graphs](pinch/pinch_l1_pinder_graphs.ipynb) | A PPI complex (barnase–barstar, `1BRS`) → two residue contact graphs + interface mask. | ~1 min | No |
| [pinch_l2_struct2graph_train](pinch/pinch_l2_struct2graph_train.ipynb) | Struct2Graph (shared-weight GCN + mutual attention); train the miniature multi-class PPI classifier. | ~1 min | Helpful |
| [pinch_l3_ig_node_embeddings](pinch/pinch_l3_ig_node_embeddings.ipynb) | Integrated Gradients on the post-GCN node embeddings via Captum; completeness + the saturation caveat. | ~1 min | No |
| [pinch_l4_attention_gnnexplainer](pinch/pinch_l4_attention_gnnexplainer.ipynb) | Mutual-attention and GNNExplainer saliencies; do they agree? | ~1 min | No |
| [pinch_l5_skempi_consensus_benchmark](pinch/pinch_l5_skempi_consensus_benchmark.ipynb) | SKEMPI v2.0 hotspots; consensus vs single methods by AUROC — the hypothesis test. | ~1 min | No |

The five `pinch_l1`–`pinch_l5` notebooks share one tested engine,
[`pinch/pinch_common.py`](pinch/pinch_common.py) (which builds on the shared
[`common/`](common/) primitives), and are generated by
[`pinch/_build_pinch_notebooks.py`](pinch/_build_pinch_notebooks.py).
`pinch/_smoke_pinch.py` is a fast end-to-end check of the whole pipeline.

## Research track — Cross-method xAI for PPI (attrib-PINCH)

The `pinch_l1`–`pinch_l5` notebooks above are the runnable, de-risked scaffold for
this research direction; the rest of this section is the plan they implement.

**Research docs (source of truth):** `G:\My Drive\ObsidianDB\vault\obsidian\xAI for DL in Proteins`

| Doc | What it gives you |
|---|---|
| `00_xAI_for_DL_in_Proteins_Definitive_Guide.md` | The student's guide: IG, protein representations, the 7 papers, 17 project ideas, and **Part 7 — why `attrib-PINCH` is the pick**. Read this first. |
| `June 2026 Paper Creation process.md` | The concrete first-three-weeks de-risking plan + task taxonomy. |
| `03_Sendin_2025_..._PPI_Geometric_DL_Explainability.md` | The pipeline we re-implement: Struct2Graph (GCN + mutual attention) on PINDER; the 0.97→0.78 F1 interface-ablation result (shortcut learning). |
| `02_Sundararajan_..._Integrated_Gradients_...md` | IG axioms + recipe (the IG-foundation lessons implement this). |
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

### Pipeline → lessons to study, and the notebook that implements it

The **Implemented in** column points at the runnable scaffold (`pinch_l1`–`pinch_l5`,
on barnase–barstar as a miniature stand-in for PINDER). **New work** is what
remains to turn the scaffold into the full experiment.

| Stage | Lesson(s) to study first | Implemented in | New work to scale up |
|---|---|---|---|
| Protein → residue graph (C-α nodes, distance-threshold edges) | [gnn_l1](gnn/gnn_l1_graphs_from_proteins.ipynb), [gnn_l6](gnn/gnn_l6_real_structures.ipynb), [gnn_l7](gnn/gnn_l7_edge_features.ipynb) | **[pinch_l1](pinch/pinch_l1_pinder_graphs.ipynb)** | PINDER loader; per-interface 4/6/8/10 Å ablation graphs |
| GCN encoder + mutual attention over a protein **pair** | [gnn_l3](gnn/gnn_l3_graph_classification.ipynb), [gnn_l11](gnn/gnn_l11_interaction_graphs.ipynb), [plm_l6](plm/plm_l6_attention_contacts.ipynb) | **[pinch_l2](pinch/pinch_l2_struct2graph_train.ipynb)** | Train on PINDER's 31-cluster holo subset, not 5 complexes |
| Optional pLM node features | [gnn_l4](gnn/gnn_l4_plm_plus_gnn.ipynb), [capstone_l1](capstones/capstone_l1_end_to_end_plm_gnn.ipynb) | (hook in `pinch_common`) | ESM-2 embeddings as node features (Sendin future-work #2) |
| **xAI method 1 — attention** | [plm_l6](plm/plm_l6_attention_contacts.ipynb) | **[pinch_l4](pinch/pinch_l4_attention_gnnexplainer.ipynb)** | — |
| **xAI method 2 — Integrated Gradients** | [ig_l1](common/ig_l1_simple.ipynb), [ig_l2](common/ig_l2_tiny_network.ipynb) | **[pinch_l3](pinch/pinch_l3_ig_node_embeddings.ipynb)** | Converged IG — needs a regularised (non-overfit) model; see pinch_l3's saturation note |
| **xAI method 3 — GNNExplainer** | GNN track (PyG `GNNExplainer`) | **[pinch_l4](pinch/pinch_l4_attention_gnnexplainer.ipynb)** | — |
| Consensus + benchmark | [capstone_l2](capstones/capstone_l2_benchmark_suite.ipynb) (honest eval) | **[pinch_l5](pinch/pinch_l5_skempi_consensus_benchmark.ipynb)** | Aggregate AUROC across all PINDER↔SKEMPI complexes with CIs |
| 3D visualisation | — | (`pos` kept on every graph) | PyMOL renders of high-saliency residues vs measured hotspots |

### Datasets this track needs (beyond the lessons' HF sets)

| Name | What it gives | URL |
|---|---|---|
| **PINDER** | 2.3 M PPIs with interface Cluster IDs (the training set) | github.com/pinder-org/pinder |
| **SKEMPI v2.0** | ~7,000 mutation-induced ΔΔG values across 345 PPI complexes (the ground truth) | life.bsc.es/pid/skempi2 |
| **PyMOL** | 3D visualisation of saliency vs hotspots | `pip install pymol-open-source` |

`captum` and `torch_geometric` are already in [requirements.txt](requirements.txt).

### De-risking plan (first three weeks, from the research docs)

1. **Week 1** — Clone PINDER, replicate Sendin's 31-cluster training run, confirm ~0.97 F1 reproduces. *(Scaffold: `pinch_l1`/`pinch_l2` do this on a 5-complex miniature; swap in PINDER.)*
2. **Week 2** — IG on the GCN node embeddings (Captum); GNNExplainer via PyG. *(Done in `pinch_l3`/`pinch_l4`.)*
3. **Week 3** — Pull SKEMPI v2.0, intersect with PINDER complexes, first plot of *consensus high-saliency residues vs measured ΔΔG ≥ 2 kcal/mol*. *(Done in `pinch_l5` on barnase–barstar.)*

> Go/no-go: if by week 3 IG produces sensible per-residue attributions on a single
> complex, the project is on track. If not, fall back to **`attrib-FAITH`** (BioLiP2 +
> ESM-2 ligand-binding-site faithfulness benchmark) — same shape, flatter compute curve.

**Status:** the week-1→3 pipeline runs end to end on the miniature dataset
(`pinch/_smoke_pinch.py`, ~6 s on GPU). Two honest findings already surfaced and are
documented in the notebooks: (1) IG's completeness delta stays large regardless
of step count because the demo model is overfit/saturated — a *model* problem,
not a step problem. The `pinch_l3` appendix sweeps dropout/weight-decay and shows
there is **no fix at this scale**: light regularization still memorizes, heavy
regularization collapses the model to chance (a degenerate delta≈0) — so *scale*,
not hyperparameters, is the real lever (see `pinch_l3`); (2) on a single toy complex the consensus does
**not** reliably beat single methods — settling that needs PINDER at scale (see
`pinch_l5`). Both are expected and on-theme; the real verdict needs the full dataset.

### Running the notebooks responsibly

Compute-heavy runs are gated by [`_resource_check.py`](_resource_check.py): it
exits 0 only when CPU **and** GPU are both ≤ 70%, so you can guard a run with it:

```powershell
.\.venv\Scripts\python.exe _resource_check.py        # GO (exit 0) or WAIT (exit 1)
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=600 pinch\pinch_l2_struct2graph_train.ipynb
```

### Parameterised runs + tag-level reports (MLflow)

To sweep hyperparameters / scale up and compare runs, use the harness instead of
editing notebooks. Each run trains + benchmarks one config and logs params and
per-method hotspot AUROC to the repo's local MLflow store. **Every run self-gates
on `_resource_check.py`** (waits and re-checks if the box is busy — safe to launch
overnight). Group runs with `--tag`, then render one report per tag.

```powershell
# one configuration
.\.venv\Scripts\python.exe pinch\run_pinch.py --dropout 0.3 --wd 1e-2 --tag my-experiment

# preset overfit -> regularised -> collapse sweep, all under one tag
.\.venv\Scripts\python.exe pinch\run_pinch.py --sweep --tag reg-sweep

# a multi-seed grid (overnight): repeat with different seeds under one tag
foreach ($s in 0..4) {
  .\.venv\Scripts\python.exe pinch\run_pinch.py --dropout 0.3 --wd 1e-3 --seed $s --tag seed-robustness
}

# build the tag-level report (markdown + chart under pinch/reports/<tag>.md)
.\.venv\Scripts\python.exe pinch\report_pinch.py --tag reg-sweep

# browse everything
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`run_pinch.py` knobs: `--dropout --wd --augment --epochs --lr --ig-steps
--gnnx-epochs --threshold --seed --hidden --dataset {demo,pinder}`. `--dataset
pinder` is reserved for the real-data path (option *b*) and currently raises by
design. The pipeline metric source of truth is `pinch_common.run_experiment(...)`.

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
- [ig_l1_simple](common/ig_l1_simple.ipynb) — IG math from scratch, no downloads.
- [plm_l1_embeddings_probe](plm/plm_l1_embeddings_probe.ipynb) — pLM + sklearn.

If a dataset throws a column-name error, run the early cells — every lesson
prints the dataset's features and tells you which constants to adjust.

> `plm/plm_demo.ipynb` is the original starter and overlaps with `plm_l3`.
> Safe to skip once you're comfortable with the lessons.
