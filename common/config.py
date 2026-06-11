"""Shared constants and device helper for the research tracks.

These are the bits every track (`pinch/`, `clasp/`, `edits/`) needs no matter
what it is studying: the canonical amino-acid order for one-hot encoding, the
3-letter -> 1-letter residue map for parsing PDB files, and the device pick.
Lifted out of the original combined research module so all three tracks share one copy.
"""

import torch

# The 20 standard amino acids in a fixed order. Column `i` of a one-hot vector
# corresponds to AMINO_ACIDS[i]; the order must never change between
# featurisation and model training.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}

# Three-letter -> one-letter residue codes, for parsing PDB files.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def get_device():
    """Return 'cuda' when a GPU is visible, else 'cpu'. The graphs in these
    tracks are tiny, so CPU is fine; the same code moves to GPU unchanged."""
    return "cuda" if torch.cuda.is_available() else "cpu"
