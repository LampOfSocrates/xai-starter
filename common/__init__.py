"""Shared code for the protein-xAI research tracks.

The three research tracks — ``pinch/`` (attrib-PINCH), ``clasp/`` (contact-CLASP)
and ``edits/`` (attrib-EDITS) — all stand on the same primitives: the amino-acid
alphabet, PDB download/parse, residue-graph construction, and rank-normalisation.
Those live here so each track imports them instead of copying them.

Track-specific machinery (the Struct2Graph model, SKEMPI loaders, DCA, the
counterfactual search) stays in each track's own ``<track>_common.py``.

A notebook in a track folder reaches this package by putting the repo root on
``sys.path`` first::

    import os, sys
    ROOT = os.getcwd()
    while not os.path.isdir(os.path.join(ROOT, "common")):
        ROOT = os.path.dirname(ROOT)
    sys.path.insert(0, ROOT)
    from common import fetch_pdb, contact_graph, get_device
"""

from .config import AMINO_ACIDS, AA_TO_IDX, THREE_TO_ONE, get_device
from .structures import fetch_pdb, parse_chain
from .graphs import one_hot_aa, contact_graph, interface_mask
from .metrics import rank01

__all__ = [
    "AMINO_ACIDS", "AA_TO_IDX", "THREE_TO_ONE", "get_device",
    "fetch_pdb", "parse_chain",
    "one_hot_aa", "contact_graph", "interface_mask",
    "rank01",
]
