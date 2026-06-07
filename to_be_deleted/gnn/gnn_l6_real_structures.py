"""
GNN Lesson 6: Real Protein Structures & Contact Maps
=====================================================
What you'll learn
-----------------
- How to fetch and parse a real PDB file without any third-party biology
  library, using only Python's standard library and numpy.
- How to build a contact graph from real Cα coordinates and why it looks
  very different from the synthetic-helix graphs in Lesson 1.
- What a contact map is and how to read it — rows and columns are residues;
  a filled cell means the two residues are within 8 Å of each other.
- Why "long-range contacts" (sequence separation > 5) are the interesting
  ones: they are exactly what sequence-window graphs miss.

The intuition
-------------
In Lesson 1 we generated a synthetic alpha-helix. Every residue was ~equal
distance from its neighbours, so the contact graph was boring and regular.
Real proteins fold into complex 3D shapes: beta-sheets, turns, buried cores.
The contact map of a real protein is dense along the diagonal (sequential
neighbours) but also has characteristic off-diagonal patterns that encode
secondary-structure and long-range interactions.

Why this matters
----------------
A machine-learning model that can read a real contact map — or, better, one
that is given 3D coordinates and learns to reason over them — has access to
structural information that no sequence model can provide directly. This is
the core motivation for structure-aware GNNs (e.g. the GVP, SE(3) networks).
"""

import os
import urllib.request
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib
matplotlib.use("Agg")   # no display needed; must be set before pyplot import
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDB_ID = "1UBQ"                            # ubiquitin, 76 residues, tiny and stable
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
RESULTS_DIR = "./results"
PDB_CACHE = os.path.join(RESULTS_DIR, f"{PDB_ID}.pdb")
CONTACT_THRESHOLD = 8.0                    # Å — standard Cα-Cα contact cutoff
LONG_RANGE_SEQSEP = 5                      # residue pairs separated by > this count

# Three-letter → one-letter amino acid code mapping (standard 20 AAs)
AA3_TO_AA1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}


# ---------------------------------------------------------------------------
# PDB fetching
# ---------------------------------------------------------------------------

def fetch_pdb(url, cache_path):
    """Download a PDB file from RCSB and cache it locally.

    On subsequent runs the cached file is used directly, avoiding
    unnecessary network traffic. Returns the raw text of the file.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r") as fh:
            return fh.read()
    print(f"  Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=15) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    with open(cache_path, "w") as fh:
        fh.write(text)
    print(f"  Cached to {cache_path}")
    return text


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_ca_atoms(pdb_text, chain="A"):
    """Extract Cα (alpha-carbon) atom records for one chain from PDB text.

    PDB ATOM records have fixed-width columns defined by the PDB format spec:
      columns  1- 6   record type ("ATOM  ")
      columns 13-16   atom name (CA for alpha-carbon)
      columns 22      chain ID
      columns 23-26   residue sequence number
      columns 18-20   residue name (three-letter code)
      columns 31-38   x coordinate
      columns 39-46   y coordinate
      columns 47-54   z coordinate

    Returns (sequence_string, coords_array) where coords_array is (L, 3).
    If a residue appears more than once (e.g. alternate conformers), we keep
    only the first occurrence.
    """
    seen_resnums = set()
    residues = []   # list of (resnum, one_letter, x, y, z)

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain_id = line[21].strip()
        if chain_id != chain:
            continue
        resnum = int(line[22:26].strip())
        if resnum in seen_resnums:
            continue    # skip alternate conformers
        seen_resnums.add(resnum)

        resname = line[17:20].strip()
        aa1 = AA3_TO_AA1.get(resname, "X")   # X = unknown / non-standard
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        residues.append((resnum, aa1, x, y, z))

    # Sort by residue number (PDB files are usually ordered, but be safe)
    residues.sort(key=lambda r: r[0])
    sequence = "".join(r[1] for r in residues)
    coords = np.array([[r[2], r[3], r[4]] for r in residues], dtype=np.float32)
    return sequence, coords


# ---------------------------------------------------------------------------
# Synthetic fallback (from Lesson 1) used if network is unavailable
# ---------------------------------------------------------------------------

def synthetic_helix_coords(n_residues, rise=1.5, radius=2.3, turn_deg=100.0):
    """Generate plausible alpha-helix Cα coordinates as a fallback."""
    angles = np.deg2rad(turn_deg * np.arange(n_residues))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = rise * np.arange(n_residues)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def one_hot_aa(sequence):
    """Convert a sequence string to a (L, 20) one-hot node-feature tensor."""
    n = len(sequence)
    x = torch.zeros(n, len(AMINO_ACIDS))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            x[i, AA_TO_IDX[aa]] = 1.0
    return x


def contact_graph(sequence, coords, threshold=CONTACT_THRESHOLD):
    """Build a PyG Data object from Cα coordinates using a distance cutoff.

    Each edge (i, j) exists if the Cα-Cα distance is below `threshold` Å.
    Edge features are the actual distances (useful for downstream models).
    """
    n = len(sequence)
    # Compute full pairwise distance matrix — shape (n, n)
    diffs = coords[:, None, :] - coords[None, :, :]   # broadcast subtraction
    dist = np.linalg.norm(diffs, axis=-1)              # (n, n)

    mask = (dist < threshold) & (dist > 0)   # True where a contact exists
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
# Contact-map statistics
# ---------------------------------------------------------------------------

def contact_stats(data, threshold=CONTACT_THRESHOLD, long_range_sep=LONG_RANGE_SEQSEP):
    """Compute descriptive statistics for the contact graph.

    Long-range contacts are pairs (i, j) with |i - j| > long_range_sep.
    These are the contacts that cannot be captured by any fixed-window
    sequence graph: they require 3D structural information.

    Returns a dict of metrics.
    """
    n = data.num_nodes
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()

    # Each undirected edge is stored as two directed edges; count unique pairs.
    unique_contacts = data.num_edges // 2
    avg_degree = data.num_edges / n

    seq_sep = np.abs(src.astype(int) - dst.astype(int))
    long_range_mask = seq_sep > long_range_sep
    # Each long-range directed edge also has a reverse; divide by 2.
    n_long_range = long_range_mask.sum() // 2
    frac_long_range = n_long_range / unique_contacts if unique_contacts > 0 else 0.0

    return {
        "num_residues": n,
        "num_contacts": unique_contacts,
        "avg_degree": avg_degree,
        "n_long_range": int(n_long_range),
        "frac_long_range": frac_long_range,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_contact_map(coords, sequence, threshold, savepath):
    """Save a binary contact-map heatmap to disk.

    Rows and columns are residues. A white cell means the two residues are
    within `threshold` Å; dark cells are non-contacts. The prominent
    diagonal band is sequential neighbours (expected). Off-diagonal patches
    reveal beta-sheets and other long-range structure.
    """
    n = len(sequence)
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    contact_matrix = (dist < threshold).astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(contact_matrix, cmap="Blues", origin="upper", vmin=0, vmax=1)
    ax.set_title(f"{PDB_ID} contact map  (Cα-Cα < {threshold} Å)", fontsize=12)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Contact (1=yes)")
    plt.tight_layout()
    plt.savefig(savepath, dpi=100, bbox_inches="tight")
    plt.close()


def save_graph_drawing(data, sequence, savepath):
    """Save a networkx spring-layout drawing of the contact graph.

    Nodes are coloured by residue index (position in chain) to make the
    overall fold topology visible even in a 2D force-directed layout.
    """
    g = nx.Graph()
    for i in range(data.num_nodes):
        g.add_node(i)
    for s, t in data.edge_index.t().tolist():
        g.add_edge(s, t)

    node_colors = [i / data.num_nodes for i in range(data.num_nodes)]

    plt.figure(figsize=(8, 8))
    # Use Cα xy-projected positions as the layout for a more meaningful picture
    if data.pos is not None:
        pos_2d = {i: (data.pos[i, 0].item(), data.pos[i, 1].item())
                  for i in range(data.num_nodes)}
    else:
        pos_2d = nx.spring_layout(g, seed=42)

    nx.draw(
        g, pos_2d,
        node_size=60,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        edge_color="lightgrey",
        width=0.6,
        with_labels=False,
    )
    plt.title(
        f"{PDB_ID} contact graph  ({data.num_nodes} residues, "
        f"threshold={CONTACT_THRESHOLD} Å)\n"
        "node colour = residue index (N-term=purple, C-term=yellow)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(savepath, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 1. Fetch or fall back to synthetic helix -----------------------
    using_real = True
    try:
        pdb_text = fetch_pdb(PDB_URL, PDB_CACHE)
        sequence, coords = parse_ca_atoms(pdb_text, chain="A")
        print(f"Loaded {PDB_ID}  ({len(sequence)} residues from chain A)")
        print(f"Sequence: {sequence}")
    except Exception as exc:
        print(f"WARNING: Could not fetch PDB ({exc}). Falling back to synthetic helix.")
        using_real = False
        sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        coords = synthetic_helix_coords(len(sequence))
        print(f"Using synthetic helix  ({len(sequence)} residues)")

    # ---- 2. Build contact graph -----------------------------------------
    print(f"\nBuilding contact graph (threshold = {CONTACT_THRESHOLD} Å) ...")
    data = contact_graph(sequence, coords)

    # ---- 3. Print statistics --------------------------------------------
    stats = contact_stats(data)
    print(f"\n{'='*50}")
    print(f"  Structure:      {'REAL (' + PDB_ID + ')' if using_real else 'SYNTHETIC HELIX'}")
    print(f"  Residues:       {stats['num_residues']}")
    print(f"  Contacts:       {stats['num_contacts']}")
    print(f"  Avg degree:     {stats['avg_degree']:.2f}")
    print(f"  Long-range (|i-j| > {LONG_RANGE_SEQSEP}): "
          f"{stats['n_long_range']}  "
          f"({stats['frac_long_range']:.1%} of all contacts)")
    print(f"{'='*50}")
    print(
        f"\n  > {stats['frac_long_range']:.1%} of contacts are long-range.\n"
        "  These cross-chain connections are invisible to any fixed-window\n"
        "  sequence graph — only structure-based graphs capture them."
    )

    # ---- 4. Visualise ---------------------------------------------------
    map_path = os.path.join(RESULTS_DIR, "gnn_l6_contact_map.png")
    graph_path = os.path.join(RESULTS_DIR, "gnn_l6_contact_graph.png")

    print("\nSaving contact map  ...")
    save_contact_map(coords, sequence, CONTACT_THRESHOLD, map_path)

    print("Saving graph drawing ...")
    save_graph_drawing(data, sequence, graph_path)

    print(f"\nFigures saved to:\n  {map_path}\n  {graph_path}")

    print(
        """
Things to experiment with:
- Try a different PDB ID: change PDB_ID at the top of the file (e.g. "2HHB"
  for haemoglobin or "6LU7" for the SARS-CoV-2 main protease).
- Vary CONTACT_THRESHOLD: 4 Å ≈ covalent/van-der-Waals contact; 12 Å is loose.
  Watch how the fraction of long-range contacts changes.
- Download an AlphaFold structure from https://alphafold.ebi.ac.uk/ and point
  PDB_URL at the downloaded file path — the parser works on any standard PDB.
- Feed this real Data object directly into the GCN from Lesson 2 (gnn_l2)
  for a node-classification task (e.g. predict secondary structure type).
- Add backbone-dihedral (phi, psi) angles as extra node features: they require
  N, Cα, C atom records, which are already in the PDB file — extend the parser.
- Replace one-hot node features with ESM-2 per-residue embeddings (Lesson 4)
  and compare the learned representations on a real structure.
"""
    )


if __name__ == "__main__":
    main()
