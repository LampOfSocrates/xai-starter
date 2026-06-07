# GNN for Proteins — Lesson Notebooks

A hands-on path from "what is a graph?" to equivariant 3D models and real
structures. Each lesson is an interactive notebook (`gnn_l*.ipynb`) with the
explanations broken out as markdown. **Run the cells top to bottom** (*Run All*).

## Core lessons (1–5)

| # | Notebook | You'll learn | Task |
|---|----------|--------------|------|
| 1 | [Graphs from proteins](gnn_l1_graphs_from_proteins.ipynb) | Nodes/edges/features, PyG `Data`, `edge_index`; sequence vs. contact graphs. Ends with an **interactive PDB explorer** (ipywidgets). | Build + visualise graphs |
| 2 | [Node classification](gnn_l2_node_classification.ipynb) | Message passing, 2-layer **GCN**, why a GNN beats a per-node MLP | Per-residue "hydrophobic core" |
| 3 | [Graph classification](gnn_l3_graph_classification.ipynb) | Graph-level prediction, **GAT** attention, pooling | DeepSol solubility (one-hot) |
| 4 | [pLM + GNN](gnn_l4_plm_plus_gnn.ipynb) | **ESM-2 embeddings** as node features (same task/model as L3) | DeepSol solubility (pLM feats) |
| 5 | [Equivariant GNN](gnn_l5_equivariant_gnn.ipynb) | Equivariance vs. invariance, a minimal **EGNN** from scratch | Verify equivariance numerically |

## Advanced lessons (6–11)

| # | Notebook | You'll learn | Task |
|---|----------|--------------|------|
| 6 | [Real structures](gnn_l6_real_structures.ipynb) | Download + parse a real PDB; true contact maps (no synthetic helix) | Real contact graph + stats |
| 7 | [Edge features & geometry](gnn_l7_edge_features.ipynb) | Putting distance/direction on edges; **NNConv** vs plain GCN | Geometry-dependent graph label |
| 8 | [Contact prediction](gnn_l8_contact_prediction.ipynb) | Edge-level tasks; **Precision@L** | Predict which residue pairs touch |
| 9 | [k-NN & learned graphs](gnn_l9_knn_graphs.ipynb) | Building edges from ESM-2 embeddings when no structure exists | Compare 3 graph constructions |
| 10 | [Oversmoothing & depth](gnn_l10_oversmoothing.ipynb) | Why deep GNNs collapse; residual / jumping-knowledge fixes | Accuracy vs depth study |
| 11 | [Interaction graphs](gnn_l11_interaction_graphs.ipynb) | Heterogeneous / bipartite graphs (`HeteroData`) | Protein–drug link prediction |

Then continue to the [capstones](../capstones/README.md) (end-to-end pLM+GNN, honest benchmarking).

## Running

```bash
pip install -r requirements.txt
jupyter lab            # then open any gnn_l*.ipynb and Run All
```

- Self-contained (synthetic data, no downloads): **L1, L2, L5, L7, L8, L10, L11**.
- Download DeepSol and/or ESM-2 weights on first run: **L3, L4, L9**.
- **L1**'s interactive explorer needs `biopython` (`pip install biopython`) and
  `ipywidgets`; **L6** downloads a PDB file over HTTP.

## Regenerating the notebooks

The notebooks were generated from the lesson scripts now archived in
[../to_be_deleted/gnn/](../to_be_deleted/gnn/). To rebuild one after editing its
script:

```bash
python _py_to_notebook.py to_be_deleted/gnn/gnn_l2_node_classification.py
```

(The notebooks are canonical now — editing them directly is fine too. The
interactive cells in L1 were hand-added and are not in the source script.)
