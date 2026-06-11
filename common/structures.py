"""Protein structure I/O — download and parse PDB files.

Shared by every track that needs real coordinates: `pinch/` (contact graphs of
a complex), `clasp/` (Cβ contacts of a single chain), `edits/` (structure-aware
predictors). Lifted verbatim from the original combined research module.
"""

import os

import numpy as np

from .config import THREE_TO_ONE


def fetch_pdb(pdb_id, cache_dir="./data"):
    """Download a PDB structure from RCSB (once) and return the local path.

    Uses ``certifi``'s CA bundle explicitly: a conda base environment often
    exports ``SSL_CERT_FILE`` pointing at a bundle this venv cannot read, which
    otherwise makes ``requests`` fail with an SSL error inside a venv.

    Args:
        pdb_id:    4-character PDB accession, e.g. ``"1brs"``.
        cache_dir: directory to cache the .pdb file in.

    Returns:
        Absolute path to the downloaded ``<pdb_id>.pdb`` file.
    """
    import certifi
    import requests

    pdb_id = pdb_id.lower()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if not os.path.isfile(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        resp = requests.get(url, timeout=60, verify=certifi.where())
        resp.raise_for_status()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
    return os.path.abspath(path)


def parse_chain(pdb_path, chain_id, model_id=0):
    """Extract one chain's sequence and C-alpha coordinates from a PDB file.

    Only standard amino-acid residues with a C-alpha atom are kept (waters,
    ligands, and non-standard residues are skipped). Insertion codes are
    ignored; the first conformer of each residue is used.

    Args:
        pdb_path: path to a .pdb file (see ``fetch_pdb``).
        chain_id: the chain identifier, e.g. ``"A"``.
        model_id: which model to read (0 = first; matters for NMR ensembles).

    Returns:
        A tuple ``(sequence, ca_coords, resnums)`` where
            sequence  : str of one-letter codes, length L.
            ca_coords : float ndarray of shape (L, 3) - C-alpha xyz.
            resnums   : list of L author residue numbers (ints) for cross-
                        referencing SKEMPI mutations later.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    chain = structure[model_id][chain_id]

    seq, coords, resnums = [], [], []
    for residue in chain:
        resname = residue.get_resname().strip().upper()
        if resname not in THREE_TO_ONE:          # skip HOH, ligands, modified
            continue
        if "CA" not in residue:                   # need a C-alpha to place a node
            continue
        seq.append(THREE_TO_ONE[resname])
        coords.append(residue["CA"].get_coord())
        resnums.append(residue.id[1])
    return "".join(seq), np.asarray(coords, dtype=float), resnums
