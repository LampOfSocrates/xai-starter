"""
gnn_common.py - shared building blocks for the GNN lesson notebooks
===================================================================
Lesson 1 (`gnn_l1`) builds these primitives from scratch so you can see how a
protein becomes a graph. From Lesson 2 onward the same helpers get re-used over
and over, so they live here in one documented place instead of being copied into
every notebook.

What's in here
--------------
Constants
    AMINO_ACIDS   - the 20 standard amino acids, in a fixed canonical order.
    AA_TO_IDX     - {amino-acid letter -> column index} for one-hot encoding.
    HYDROPHOBIC   - the hydrophobic amino acids (used by the Lesson 2 toy task).

Featurisation
    one_hot_aa(sequence)                      -> (L, 20) node-feature tensor.

Graph construction
    make_edge_index(n, window)                -> (2, E) sequence-window edges.
    sequence_graph(sequence, window)          -> PyG Data (nodes + sequence edges).
    contact_graph(sequence, coords, threshold)-> PyG Data (nodes + 3D contacts).

Geometry (for demos without a real structure)
    synthetic_helix_coords(n_residues, ...)   -> (n, 3) alpha-helix Ca coordinates.

Visualisation
    draw_arch_stack(ax, title, layers)        -> matplotlib block diagram of a
                                                 stacked model (input -> output).

Importing this module from a notebook
-------------------------------------
The notebooks live in `gnn/` next to this file. To import it regardless of
whether Jupyter was launched from the repo root or from `gnn/`:

    import os, sys
    _root = os.path.abspath("")
    for _cand in (_root, os.path.join(_root, "gnn"), os.path.dirname(_root)):
        if os.path.isfile(os.path.join(_cand, "gnn_common.py")):
            sys.path.insert(0, _cand); break
    from gnn_common import one_hot_aa, make_edge_index, draw_arch_stack
"""

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The 20 standard amino acids in a fixed order. Index `i` of this string is the
# one-hot column for that residue, so the order must never change between
# featurisation and model training.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Reverse lookup: amino-acid letter -> its column index in the one-hot vector.
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}

# The hydrophobic ("water-fearing") amino acids. These tend to pack into the
# protein's core; Lesson 2's toy label is built from this set.
HYDROPHOBIC = set("AVLIFMWY")


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------

def one_hot_aa(sequence):
    """Convert an amino-acid string into a one-hot node-feature tensor.

    Each residue becomes a length-20 vector that is all zeros except for a
    single 1 at the column for its amino acid (see ``AA_TO_IDX``). This is the
    simplest possible node feature; later lessons swap it for ESM-2 embeddings.

    Unknown characters (e.g. 'X', '-', or a non-standard residue) are encoded as
    an all-zero row rather than raising - convenient when parsing real
    sequences that contain odd letters.

    Args:
        sequence: a string of single-letter amino-acid codes, length L.

    Returns:
        A ``torch.FloatTensor`` of shape ``(L, 20)``.
    """
    n = len(sequence)
    x = torch.zeros(n, len(AMINO_ACIDS))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:                 # skip unknown residues -> zero row
            x[i, AA_TO_IDX[aa]] = 1.0
    return x


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def make_edge_index(n, window=3):
    """Build the ``edge_index`` of a sequence-window graph.

    Every residue is connected to the ``window`` residues on each side of it in
    the chain. PyTorch Geometric stores edges as a ``(2, E)`` tensor of
    ``[source; target]`` indices, and treats every edge as directed - so each
    undirected connection is added twice (i->j and j->i).

    Args:
        n:      number of residues (nodes).
        window: how many neighbours on each side to connect (1 = only adjacent).

    Returns:
        A ``torch.LongTensor`` of shape ``(2, E)`` suitable for any PyG layer.
    """
    edges = []
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                edges.append((i, i + d))    # forward edge
                edges.append((i + d, i))    # and its reverse (undirected)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def sequence_graph(sequence, window=2):
    """A protein graph whose edges follow the chain (sequence proximity).

    Nodes are residues with one-hot features; edges connect residues that are
    within ``window`` positions of each other in the sequence. Returns a ready
    to use PyG ``Data`` object.
    """
    return Data(
        x=one_hot_aa(sequence),
        edge_index=make_edge_index(len(sequence), window=window),
    )


def contact_graph(sequence, coords, threshold=8.0):
    """A protein graph whose edges follow 3D space (spatial proximity).

    An edge is added between any two residues whose C-alpha atoms are closer than
    ``threshold`` angstroms. 8 A is the standard "contact" cutoff in structural
    biology. This captures long-range interactions (active sites, packing) that
    a sequence graph cannot see.

    Args:
        sequence:  amino-acid string, length L (used for the node features).
        coords:    array-like of shape (L, 3) of C-alpha coordinates.
        threshold: contact distance cutoff in angstroms.

    Returns:
        A PyG ``Data`` object with node features ``x``, ``edge_index``, the
        contact distances as ``edge_attr``, and the coordinates as ``pos``.
    """
    coords = np.asarray(coords, dtype=float)

    # Pairwise C-alpha distance matrix, shape (L, L).
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)

    # Keep pairs under the threshold, excluding the diagonal (i == j).
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


# ---------------------------------------------------------------------------
# Geometry (synthetic, for demos without a real structure)
# ---------------------------------------------------------------------------

def synthetic_helix_coords(n_residues, rise=1.5, radius=2.3, turn_deg=100.0):
    """Generate plausible alpha-helix C-alpha coordinates.

    A real alpha-helix advances ~1.5 A along its axis and rotates ~100 degrees
    per residue. This is a stand-in for a real structure when you just want some
    geometry to build a contact graph from. For real coordinates, parse a PDB
    file (see ``gnn_l6``) instead.

    Returns:
        A NumPy array of shape ``(n_residues, 3)``.
    """
    angles = np.deg2rad(turn_deg * np.arange(n_residues))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = rise * np.arange(n_residues)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_arch_stack(ax, title, layers):
    """Draw a vertical block diagram of a neural-network architecture.

    Each layer is a rounded box; boxes are stacked from input (top) to output
    (bottom) with arrows between them, and an optional tensor-shape label to the
    right of each box. Used in Lesson 2 to contrast the GCN and the MLP.

    Args:
        ax:     a matplotlib Axes to draw into.
        title:  the title for this stack (e.g. the model name).
        layers: a list of ``(label, shape, facecolor)`` tuples, ordered input
                first. Use the pipe character ``|`` inside ``label`` to force a
                line break (kept out of source strings to avoid escaping).

    Example:
        layers = [
            ("Input", "(N, 20)", "#cfe8ff"),
            ("Linear(20 -> 2)|per node", "(N, 2)", "#ffffff"),
        ]
        draw_arch_stack(ax, "MLP", layers)
    """
    from matplotlib.patches import FancyBboxPatch  # local import: only needed here

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.1, 1.75)
    box_h, gap = 0.62, 0.42
    n = len(layers)
    ax.set_ylim(-(n * (box_h + gap)), box_h + 0.2)
    ax.axis("off")

    for i, (label, shape, color) in enumerate(layers):
        y = -i * (box_h + gap)
        ax.add_patch(FancyBboxPatch(
            (0.05, y), 0.9, box_h, boxstyle="round,pad=0.02",
            linewidth=1.6, edgecolor="#333", facecolor=color))
        ax.text(0.5, y + box_h / 2, label.replace("|", chr(10)),
                ha="center", va="center", fontsize=9)
        if shape:
            ax.text(1.02, y + box_h / 2, shape, ha="left", va="center",
                    fontsize=8.5, color="#555", family="monospace")
        if i < n - 1:  # arrow down to the next box
            ax.annotate("", xy=(0.5, y - gap + 0.02), xytext=(0.5, y - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
