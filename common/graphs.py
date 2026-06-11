"""Residue-graph construction — one-hot features, contact graphs, interface masks.

The geometric primitives shared across tracks: `pinch/` builds a contact graph
per chain, `clasp/` reuses the same Cβ-distance logic for its contact ground
truth, and the interface mask is the structural reference both PPI tracks lean
on. Lifted verbatim from the original combined research module.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from .config import AMINO_ACIDS, AA_TO_IDX


def one_hot_aa(sequence):
    """Amino-acid string -> (L, 20) one-hot node-feature tensor.

    Unknown characters become an all-zero row instead of raising."""
    x = torch.zeros(len(sequence), len(AMINO_ACIDS))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            x[i, AA_TO_IDX[aa]] = 1.0
    return x


def contact_graph(sequence, coords, threshold=8.0):
    """Build a single-chain residue contact graph as a PyG ``Data`` object.

    An edge connects any two residues whose C-alpha atoms are within
    ``threshold`` angstroms (8 A is the standard structural-biology contact
    cutoff). Node features are one-hot amino acids; ``pos`` keeps the coords so
    later notebooks can paint saliency back onto the 3D structure.
    """
    coords = np.asarray(coords, dtype=float)
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    mask = (dist < threshold) & (dist > 0)
    src, dst = np.where(mask)
    return Data(
        x=one_hot_aa(sequence),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(dist[mask], dtype=torch.float32).unsqueeze(-1),
        pos=torch.tensor(coords, dtype=torch.float32),
    )


def interface_mask(coords_a, coords_b, cutoff=8.0):
    """Flag residues at the binding interface between two chains.

    A residue in chain A is "interface" if its C-alpha lies within ``cutoff`` of
    any C-alpha in chain B, and vice versa. This is a C-alpha approximation of
    the all-atom 4 A interface Sendin used for the ablation; it is the structural
    reference we will (loosely) compare the explainability rankings against, with
    SKEMPI providing the experimental ground truth.

    Returns:
        ``(mask_a, mask_b)`` boolean ndarrays of length len(coords_a) / b.
    """
    coords_a = np.asarray(coords_a, dtype=float)
    coords_b = np.asarray(coords_b, dtype=float)
    d = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=-1)
    return d.min(axis=1) < cutoff, d.min(axis=0) < cutoff
