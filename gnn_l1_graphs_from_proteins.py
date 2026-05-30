"""
GNN Lesson 1: Representing Proteins as Graphs
==============================================
What you'll learn
-----------------
- The fundamental data structure of a graph: nodes, edges, node features.
- How to represent a protein as a PyTorch Geometric `Data` object.
- Two common graph constructions for proteins:
    1. Sequence graph  -- edges between residues adjacent in the sequence.
    2. Contact graph   -- edges between residues close in 3D space.

Why proteins are graphs
-----------------------
A protein is a chain of amino acids that folds into a 3D structure. Two
residues that are FAR APART in the sequence can be CLOSE in 3D — and that
spatial proximity often determines function (active sites, binding pockets,
etc.). A graph captures this naturally:

    Nodes         = residues
    Node features = amino acid identity (one-hot), or pLM embeddings
    Edges         = sequential or spatial relationships
    Edge features = sequence distance, 3D distance, etc.

PyTorch Geometric's Data object
-------------------------------
Every graph in PyG is a `Data` object with these tensors:

    x          : (num_nodes, num_features)   -- node features
    edge_index : (2, num_edges)              -- pairs (source, target)
    edge_attr  : (num_edges, num_features)   -- optional edge features
    y          : labels (graph- or node-level)
    pos        : (num_nodes, 3)              -- 3D coordinates (optional)

`edge_index` is the unusual bit: it's a (2, num_edges) tensor, not a square
adjacency matrix. The first row holds source-node indices, the second row
holds target-node indices. This is much more memory-efficient for sparse graphs.
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt


# The 20 standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}


def one_hot_aa(sequence):
    """Convert a sequence string into a (L, 20) one-hot tensor.

    This is the simplest possible node feature. In later lessons we'll
    swap this for ESM-2 embeddings (which capture much more biology).
    """
    n = len(sequence)
    x = torch.zeros(n, len(AMINO_ACIDS))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            x[i, AA_TO_IDX[aa]] = 1.0
    return x


def synthetic_helix_coords(n_residues, rise=1.5, radius=2.3, turn_deg=100.0):
    """Generate plausible alpha-helix Cα coordinates.

    A real alpha-helix has rise ~1.5 Å per residue and ~100° rotation between
    successive residues. This is just for demonstration — real proteins fold
    into much more complex shapes. To use real coordinates, install biopython
    and parse a PDB file (see "Things to experiment with" at the bottom).
    """
    angles = np.deg2rad(turn_deg * np.arange(n_residues))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = rise * np.arange(n_residues)
    return np.stack([x, y, z], axis=1)  # shape (n_residues, 3)


def sequence_graph(sequence, window=2):
    """Each residue is connected to its `window` nearest neighbours in the
    sequence (in both directions, so the graph is undirected)."""
    n = len(sequence)
    edges = []
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                # Add both directions — PyG expects directed edges; an
                # undirected edge is just two directed edges.
                edges.append((i, i + d))
                edges.append((i + d, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=one_hot_aa(sequence), edge_index=edge_index)


def contact_graph(sequence, coords, threshold=8.0):
    """Edge between residues whose Cα-Cα distance is below `threshold` Å.

    8 Å is the standard "contact" cutoff in protein structure analysis.
    """
    # Pairwise distance matrix (n, n).
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)

    # Exclude self-edges (i==j) and pairs above the threshold.
    mask = (dist < threshold) & (dist > 0)
    src, dst = np.where(mask)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(dist[mask], dtype=torch.float32).unsqueeze(-1)

    return Data(
        x=one_hot_aa(sequence),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords, dtype=torch.float32),
    )


def visualise_graph(data, title, savepath):
    """Use networkx for layout + matplotlib for rendering."""
    g = nx.Graph()
    for i in range(data.num_nodes):
        g.add_node(i)
    # data.edge_index is (2, E); each column is a directed edge. We add
    # them as undirected — networkx deduplicates automatically.
    for s, t in data.edge_index.t().tolist():
        g.add_edge(s, t)

    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(g, seed=42)
    nx.draw(
        g, pos,
        node_size=80, with_labels=False,
        node_color="skyblue", edge_color="lightgrey",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("./results", exist_ok=True)

    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    print(f"Sequence (length {len(sequence)}): {sequence}")

    # ---- 1. Sequence graph ----------------------------------------------
    seq_data = sequence_graph(sequence, window=2)
    print("\n[Sequence graph]")
    print(f"  num_nodes        = {seq_data.num_nodes}")
    print(f"  num_edges        = {seq_data.num_edges}  (counts each undirected edge twice)")
    print(f"  avg degree       = {seq_data.num_edges / seq_data.num_nodes:.1f}")
    print(f"  x.shape          = {tuple(seq_data.x.shape)}")
    print(f"  edge_index.shape = {tuple(seq_data.edge_index.shape)}")

    # ---- 2. Contact graph from synthetic 3D coordinates -----------------
    coords = synthetic_helix_coords(len(sequence))
    cnt_data = contact_graph(sequence, coords, threshold=8.0)
    print("\n[Contact graph from synthetic helix coords]")
    print(f"  num_nodes  = {cnt_data.num_nodes}")
    print(f"  num_edges  = {cnt_data.num_edges}")
    print(f"  avg degree = {cnt_data.num_edges / cnt_data.num_nodes:.1f}")
    print(f"  edge_attr  = mean {cnt_data.edge_attr.mean():.2f} Å, max {cnt_data.edge_attr.max():.2f} Å")

    # ---- 3. Visualise ---------------------------------------------------
    visualise_graph(seq_data, "Sequence graph (window=2)", "./results/gnn_l1_sequence_graph.png")
    visualise_graph(cnt_data, "Contact graph (threshold=8 Å, synthetic helix)", "./results/gnn_l1_contact_graph.png")
    print("\nSaved figures to ./results/gnn_l1_*.png")

    print(
        """
Things to experiment with:
- Build the contact graph from a REAL PDB structure (install biopython):
    from Bio.PDB import PDBList, PDBParser
    PDBList().retrieve_pdb_file("1ubq", file_format="pdb")  # ubiquitin, 76 residues
    # parse to get Cα coords + residue sequence, then call contact_graph().
- Vary the contact threshold (4 Å = direct contact; 8 Å = standard; 12 Å = loose).
- Vary the sequence-graph window — 1 = only-adjacent; 5 = wider context.
- Replace one_hot_aa() with ESM-2 embeddings (cf. plm_l1) — that's lesson 4's punch.
"""
    )


if __name__ == "__main__":
    main()
