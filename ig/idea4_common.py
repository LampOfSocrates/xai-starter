"""
idea4_common.py - shared building blocks for the "Idea 4" research notebooks
===========================================================================
These five notebooks (``ig_l3`` .. ``ig_l7``) re-implement and extend the
Sendin (2025) PINDER pipeline to test one falsifiable hypothesis:

    Does the CONSENSUS of three explainability methods - mutual attention,
    Integrated Gradients on the GCN node embeddings, and GNNExplainer - predict
    experimentally measured protein-protein binding hotspots (SKEMPI v2.0
    alanine-scanning Delta-Delta-G >= 2 kcal/mol) better than any single method?

``ig_l3`` builds the data primitives from scratch so you can see how a two-chain
complex becomes a pair of graphs. From ``ig_l4`` onward the same helpers get
re-used, so they live here in one documented place instead of being copied into
every notebook.

What's in here
--------------
Constants / config
    AMINO_ACIDS, AA_TO_IDX        - canonical 20-AA order for one-hot encoding.
    THREE_TO_ONE                  - 3-letter -> 1-letter residue codes.
    get_device()                  - 'cuda' if available else 'cpu'.

Data acquisition (real structures, small downloads)
    fetch_pdb(pdb_id, cache_dir)  -> local path to a downloaded .pdb file.
    parse_chain(path, chain_id)   -> (sequence, ca_coords, resnums) for one chain.

Graph construction
    one_hot_aa(sequence)          -> (L, 20) node features.
    contact_graph(seq, coords, threshold) -> PyG Data (nodes + 3D contacts).
    interface_mask(coords_a, coords_b, cutoff) -> (mask_a, mask_b) boolean arrays
                                     flagging residues at the binding interface.

Model
    Struct2Graph                  - shared-weight GCN encoder for the two chains,
                                    a mutual-attention block, and a classifier
                                    head. Exposes node embeddings and attention
                                    for the explainability notebooks.

Importing this module from a notebook
-------------------------------------
The notebooks live in ``ig/`` next to this file:

    import os, sys
    _root = os.path.abspath("")
    for _cand in (_root, os.path.join(_root, "ig"), os.path.dirname(_root)):
        if os.path.isfile(os.path.join(_cand, "idea4_common.py")):
            sys.path.insert(0, _cand); break
    from idea4_common import fetch_pdb, parse_chain, contact_graph, Struct2Graph
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------

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
    """Return 'cuda' when a GPU is visible, else 'cpu'. The graphs here are tiny,
    so CPU is fine; the same code moves to GPU unchanged for full PINDER."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data acquisition - real structures, small downloads
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model - a compact Struct2Graph (GCN + mutual attention) classifier
# ---------------------------------------------------------------------------

class Struct2Graph(nn.Module):
    """Shared-weight GCN encoder + mutual attention + classifier.

    Faithful in spirit to Sendin's adaptation of Struct2Graph (Baranwal &
    Mayank 2022): the two chains of a complex are encoded by the SAME GCN
    weights, a mutual-attention block couples their residues, the attended
    context vectors are pooled, concatenated, and classified.

    The forward pass returns the logits and stashes everything the
    explainability notebooks need on ``self._cache`` (node embeddings and the
    residue-by-residue attention matrix). It is deliberately small (hidden dim
    ~32, 2 GCN layers) so the whole pipeline runs on CPU.

    Args:
        in_dim:      node-feature width (20 for one-hot; swap in 320/1280 for
                     ESM-2 embeddings later - the multi-modal extension Sendin
                     flagged).
        hidden:      GCN / attention hidden width.
        num_classes: number of PPI classes (binary or the PINDER cluster count).
    """

    def __init__(self, in_dim=20, hidden=32, num_classes=2, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.hidden = hidden
        # Dropout is applied FUNCTIONALLY (in encode + before the classifier),
        # not as a module, so adding it leaves the state_dict keys unchanged and
        # weights saved by a dropout=0 model still load. Active only in train().
        self.dropout = dropout
        # Mutual-attention projections (query from one chain, key from the other).
        self.w_q = nn.Linear(hidden, hidden, bias=False)
        self.w_k = nn.Linear(hidden, hidden, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        self._cache = {}

    def encode(self, x, edge_index):
        """One chain's residues -> (L, hidden) node embeddings (post-GCN)."""
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def mutual_attention(self, h_a, h_b):
        """Symmetric mutual attention between two chains' node embeddings.

        Returns context vectors for each chain and the (L_a, L_b) attention
        matrix (rows sum to 1) used as the attention-saliency signal later.
        """
        scores = self.w_q(h_a) @ self.w_k(h_b).t() / (self.hidden ** 0.5)
        attn_ab = torch.softmax(scores, dim=1)        # each A-residue over B
        attn_ba = torch.softmax(scores, dim=0).t()    # each B-residue over A
        ctx_a = attn_ab @ h_b
        ctx_b = attn_ba @ h_a
        return ctx_a, ctx_b, attn_ab

    def head_from_embeddings(self, h_a, h_b, batch_a=None, batch_b=None):
        """Run mutual attention + pooling + classifier on node embeddings.

        Factored out so Integrated Gradients can attribute the class logit with
        respect to the post-GCN node embeddings (``h_a``, ``h_b``) - exactly the
        baseline-friendly attribution target recommended for GCNs."""
        ctx_a, ctx_b, attn_ab = self.mutual_attention(h_a, h_b)
        if batch_a is None:
            pooled_a = ctx_a.mean(dim=0, keepdim=True)
            pooled_b = ctx_b.mean(dim=0, keepdim=True)
        else:
            pooled_a = global_mean_pool(ctx_a, batch_a)
            pooled_b = global_mean_pool(ctx_b, batch_b)
        pooled = F.dropout(torch.cat([pooled_a, pooled_b], dim=-1),
                           p=self.dropout, training=self.training)
        logits = self.classifier(pooled)
        self._cache["attn_ab"] = attn_ab
        self._cache["h_a"], self._cache["h_b"] = h_a, h_b
        return logits

    def forward(self, data_a, data_b):
        """Classify a complex from its two per-chain graphs.

        Args:
            data_a, data_b: PyG ``Data`` graphs for the two chains.
        Returns:
            ``(1, num_classes)`` logits. Side effects: ``self._cache`` holds the
            node embeddings and attention for the explainability notebooks.
        """
        h_a = self.encode(data_a.x, data_a.edge_index)
        h_b = self.encode(data_b.x, data_b.edge_index)
        return self.head_from_embeddings(h_a, h_b)


# ---------------------------------------------------------------------------
# A tiny REAL dataset + training loop (the lightweight stand-in for PINDER)
# ---------------------------------------------------------------------------
#
# Sendin trained a 31-class PINDER *cluster* classifier (100 PPIs per cluster).
# PINDER is multi-GB, so as a runnable miniature we treat each of a handful of
# real, well-characterised two-chain complexes (all present in SKEMPI v2.0) as
# its own "cluster", and create several examples per cluster by light structural
# augmentation (random edge dropout) - the same shape of task at a scale that
# trains in seconds on CPU. The full PINDER loader is a gated extension in
# ``ig_l4``.

# (pdb_id, chain_a, chain_b, short_name). Chain A is the larger "receptor"
# (enzyme), chain B the smaller "ligand" (inhibitor) by convention here.
DEMO_COMPLEXES = [
    ("1brs", "A", "D", "barnase / barstar"),
    ("2ptc", "E", "I", "trypsin / BPTI"),
    ("1ppf", "E", "I", "elastase / OMTKY3"),
    ("1acb", "E", "I", "chymotrypsin / eglin-c"),
    ("1jtg", "A", "B", "beta-lactamase / BLIP"),
]


def _edge_dropout(data, p, generator):
    """Return a copy of ``data`` with a fraction ``p`` of its edges removed.

    Used only for augmentation: it perturbs the graph slightly so each cluster
    has several non-identical training examples, mimicking the structural
    variation of the many PPIs Sendin had per PINDER cluster."""
    if p <= 0 or data.edge_index.size(1) == 0:
        return data.clone()
    e = data.edge_index.size(1)
    keep = torch.rand(e, generator=generator) > p
    out = data.clone()
    out.edge_index = data.edge_index[:, keep]
    if data.edge_attr is not None:
        out.edge_attr = data.edge_attr[keep]
    return out


def build_demo_dataset(threshold=8.0, augment=6, edge_drop=0.10, seed=0,
                       cache_dir="./data"):
    """Assemble the miniature multi-class PPI-cluster dataset described above.

    Each complex in ``DEMO_COMPLEXES`` is one class. The pristine complex plus
    ``augment`` edge-dropout variants become that class's examples.

    Returns:
        ``samples`` - a list of dicts, each with keys:
            ``ga``, ``gb`` : the two PyG graphs (chain A, chain B),
            ``label``      : the cluster index (0 .. n_classes-1),
            ``name``       : "<pdb>:<A>-<B>",
            ``pdb``, ``chain_a``, ``chain_b`` : provenance,
            ``augmented``  : False for the pristine complex, True for variants.
    """
    g = torch.Generator().manual_seed(seed)
    samples = []
    for label, (pdb, ca, cb, _name) in enumerate(DEMO_COMPLEXES):
        path = fetch_pdb(pdb, cache_dir)
        sa, xa, _ = parse_chain(path, ca)
        sb, xb, _ = parse_chain(path, cb)
        ga, gb = contact_graph(sa, xa, threshold), contact_graph(sb, xb, threshold)
        base = {"label": label, "name": f"{pdb}:{ca}-{cb}",
                "pdb": pdb, "chain_a": ca, "chain_b": cb}
        samples.append({**base, "ga": ga, "gb": gb, "augmented": False})
        for _ in range(augment):
            samples.append({**base, "augmented": True,
                            "ga": _edge_dropout(ga, edge_drop, g),
                            "gb": _edge_dropout(gb, edge_drop, g)})
    return samples


def train_struct2graph(samples, hidden=32, epochs=80, lr=5e-3, seed=0,
                       device=None, verbose=True, dropout=0.0, weight_decay=0.0):
    """Train the multi-class Struct2Graph cluster classifier on the demo samples.

    Tiny by design - a few real complexes, a 2-layer GCN. With the defaults
    (``dropout=0``, ``weight_decay=0``) it essentially overfits at this scale;
    that is fine and even on-message (the thesis's headline lesson is shortcut
    learning). Pass ``dropout`` / ``weight_decay`` > 0 to regularise it instead -
    ``ig_l5`` uses this to show that a non-memorising model has a far smaller IG
    completeness delta (less gradient saturation). Returns the model (``eval``).
    """
    device = device or get_device()
    torch.manual_seed(seed)
    num_classes = len({s["label"] for s in samples})
    model = Struct2Graph(in_dim=20, hidden=hidden, num_classes=num_classes,
                         dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    samples = [dict(s) for s in samples]
    for s in samples:                                   # move graphs once
        s["ga"], s["gb"] = s["ga"].to(device), s["gb"].to(device)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(samples))
        total = 0.0
        for k in perm.tolist():
            s = samples[k]
            opt.zero_grad()
            logits = model(s["ga"], s["gb"])
            loss = F.cross_entropy(logits, torch.tensor([s["label"]], device=device))
            loss.backward()
            opt.step()
            total += loss.item()
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  loss {total / len(samples):.4f}")
    return model.eval()


def load_or_train_demo(weights_path, hidden=32, epochs=80, device=None,
                       cache_dir="./data"):
    """Load saved Struct2Graph weights, or train + save them if absent.

    Lets ``ig_l5`` / ``ig_l6`` / ``ig_l7`` each run top-to-bottom on their own:
    they reuse the model trained in ``ig_l4`` when present, otherwise quietly
    train a fresh one. Returns ``(model, samples)``.
    """
    device = device or get_device()
    samples = build_demo_dataset(cache_dir=cache_dir)
    num_classes = len(DEMO_COMPLEXES)
    model = Struct2Graph(in_dim=20, hidden=hidden, num_classes=num_classes).to(device)
    if weights_path and os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    else:
        model = train_struct2graph(samples, hidden=hidden, epochs=epochs,
                                   device=device, verbose=False)
        if weights_path:
            os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), weights_path)
    return model, samples


# ---------------------------------------------------------------------------
# SKEMPI v2.0 - experimental ground truth (alanine-scanning hotspots)
# ---------------------------------------------------------------------------

SKEMPI_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
RT_KCAL = 0.0019872041 * 298.15          # ~0.592 kcal/mol at 298 K


def fetch_skempi(cache_dir="./data"):
    """Download (once) and return the SKEMPI v2.0 table as a DataFrame.

    SKEMPI v2.0 holds ~7,000 mutation-induced changes in PPI binding affinity.
    The file is a ~1.6 MB semicolon-separated CSV.
    """
    import certifi
    import pandas as pd
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "skempi_v2.csv")
    if not os.path.isfile(path):
        resp = requests.get(SKEMPI_URL, timeout=180, verify=certifi.where())
        resp.raise_for_status()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
    return pd.read_csv(path, sep=";")


def skempi_ddg(df, pdb_id, alanine_only=True):
    """Per-mutation binding Delta-Delta-G for one complex, from affinities.

    ΔΔG = RT * ln(Kd_mut / Kd_wt). A positive value means the mutation *weakens*
    binding - the signature of a hotspot. SKEMPI's ``#Pdb`` field looks like
    ``1BRS_A_D``; mutation strings like ``KA25A`` read as
    wild-type **K**, chain **A**, residue **25**, mutant **A**.

    Args:
        df:           the SKEMPI DataFrame from ``fetch_skempi``.
        pdb_id:       4-char accession, e.g. ``"1brs"`` (case-insensitive).
        alanine_only: keep only single mutations to alanine (true alanine scan).

    Returns:
        A DataFrame with columns ``chain, wt, pos, mut, ddg`` (one row per
        usable single-point mutation).
    """
    import numpy as np
    import pandas as pd

    pdb_id = pdb_id.upper()
    sub = df[df["#Pdb"].str.upper().str.startswith(pdb_id)].copy()
    rows = []
    for _, r in sub.iterrows():
        mut = str(r["Mutation(s)_cleaned"])
        if "," in mut:                       # skip multi-point mutations
            continue
        kd_mut, kd_wt = r["Affinity_mut_parsed"], r["Affinity_wt_parsed"]
        if not (np.isfinite(kd_mut) and np.isfinite(kd_wt)) or kd_mut <= 0 or kd_wt <= 0:
            continue
        wt, chain, mutant = mut[0], mut[1], mut[-1]
        try:
            pos = int(mut[2:-1])
        except ValueError:
            continue
        if alanine_only and mutant != "A":
            continue
        rows.append({"chain": chain, "wt": wt, "pos": pos, "mut": mutant,
                     "ddg": RT_KCAL * np.log(kd_mut / kd_wt)})
    return pd.DataFrame(rows)


def map_ddg_to_nodes(ddg_df, chain_id, resnums, hotspot_thresh=2.0):
    """Align per-mutation ΔΔG onto a chain's graph node indices.

    Args:
        ddg_df:         output of ``skempi_ddg``.
        chain_id:       which chain to extract (matches SKEMPI's chain letter).
        resnums:        author residue numbers from ``parse_chain`` (graph order).
        hotspot_thresh: ΔΔG cutoff (kcal/mol) above which a residue is a hotspot.

    Returns:
        ``(ddg_per_node, hotspot_mask, n_measured)``:
            ddg_per_node : float ndarray length L, max ΔΔG measured at that
                           residue (NaN where no measurement exists).
            hotspot_mask : boolean ndarray length L, True where ddg >= threshold.
            n_measured   : how many residues had any SKEMPI measurement.
    """
    import numpy as np

    ddg_per_node = np.full(len(resnums), np.nan)
    # No usable rows for this complex (e.g. no single-point alanine mutations):
    # return all-unmeasured rather than KeyError on the missing 'chain' column.
    if ddg_df is None or len(ddg_df) == 0 or "chain" not in getattr(ddg_df, "columns", []):
        return ddg_per_node, np.zeros(len(resnums), dtype=bool), 0
    pos_to_idx = {p: i for i, p in enumerate(resnums)}
    for _, r in ddg_df[ddg_df["chain"] == chain_id].iterrows():
        idx = pos_to_idx.get(int(r["pos"]))
        if idx is None:
            continue
        cur = ddg_per_node[idx]
        ddg_per_node[idx] = r["ddg"] if np.isnan(cur) else max(cur, r["ddg"])
    hotspot_mask = np.nan_to_num(ddg_per_node, nan=-np.inf) >= hotspot_thresh
    n_measured = int(np.isfinite(ddg_per_node).sum())
    return ddg_per_node, hotspot_mask, n_measured


# ---------------------------------------------------------------------------
# The three explainability methods, in one place
# ---------------------------------------------------------------------------

def rank01(values):
    """Rank-normalise a 1-D array to [0, 1] (ties broken by order).

    Used to put attention, IG, and GNNExplainer - which live on completely
    different scales - onto a common footing before averaging into a consensus.
    """
    import numpy as np

    v = np.asarray(values, dtype=float)
    order = v.argsort()
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v))
    return ranks / max(len(v) - 1, 1)


def _gnnexplainer_saliency(model, x, edge_index, partner_h, epochs=60):
    """Per-node GNNExplainer importance for one chain, partner context frozen.

    GNNExplainer is built for a single graph in -> prediction out. A two-graph
    mutual-attention model has no single ``(x, edge_index)`` signature, so we
    wrap it: the partner chain's node embeddings are held fixed while
    GNNExplainer learns a soft mask over THIS chain's nodes. Documented design
    choice - the explanation is conditional on the partner."""
    import torch.nn as nn
    from torch_geometric.explain import Explainer, GNNExplainer

    class _Wrap(nn.Module):
        def __init__(self, m, h_partner):
            super().__init__()
            self.m, self.h_partner = m, h_partner

        def forward(self, x, edge_index):
            h = self.m.encode(x, edge_index)
            return self.m.head_from_embeddings(h, self.h_partner)

    expl = Explainer(
        _Wrap(model, partner_h),
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification",
                          task_level="graph", return_type="raw"),
    )
    e = expl(x, edge_index)
    return e.node_mask.squeeze().detach().cpu().numpy()


def compute_all_saliencies(model, ga, gd, target=None, ig_steps=48,
                           gnnx_epochs=60, device=None):
    """Run all three xAI methods on a complex and return per-residue saliencies.

    Returns a dict with, for each chain ('a' = receptor, 'd' = ligand):
        ``attention``, ``ig``, ``gnnexplainer`` : raw per-residue saliency,
        ``consensus``                            : mean of the three rank01'd,
    plus ``target`` (the class explained) and ``ig_delta`` (IG completeness
    convergence delta - should be small).
    """
    import numpy as np
    import torch
    from captum.attr import IntegratedGradients

    device = device or get_device()
    model = model.to(device).eval()
    ga, gd = ga.to(device), gd.to(device)

    logits = model(ga, gd)
    if target is None:
        target = int(logits.argmax(1).item())

    # 1) mutual attention: sum the attention each residue sends to the partner.
    attn = model._cache["attn_ab"].detach().cpu().numpy()     # (La, Ld)
    sal = {"a": {"attention": attn.sum(1)}, "d": {"attention": attn.sum(0)}}

    # 2) Integrated Gradients on the post-GCN node embeddings (Captum).
    h_a = model.encode(ga.x, ga.edge_index).detach()
    h_d = model.encode(gd.x, gd.edge_index).detach()
    ig = IntegratedGradients(lambda a, b: model.head_from_embeddings(a, b))
    (att_a, att_d), delta = ig.attribute(
        (h_a, h_d), baselines=(torch.zeros_like(h_a), torch.zeros_like(h_d)),
        target=target, n_steps=ig_steps, return_convergence_delta=True)
    sal["a"]["ig"] = att_a.abs().sum(1).cpu().numpy()
    sal["d"]["ig"] = att_d.abs().sum(1).cpu().numpy()

    # 3) GNNExplainer, each chain in turn with the partner context frozen.
    sal["a"]["gnnexplainer"] = _gnnexplainer_saliency(model, ga.x, ga.edge_index, h_d, gnnx_epochs)
    sal["d"]["gnnexplainer"] = _gnnexplainer_saliency(model, gd.x, gd.edge_index, h_a, gnnx_epochs)

    # consensus = mean of the three rank-normalised signals.
    for chain in ("a", "d"):
        stack = [rank01(sal[chain][k]) for k in ("attention", "ig", "gnnexplainer")]
        sal[chain]["consensus"] = np.mean(stack, axis=0)

    sal["target"] = target
    sal["ig_delta"] = float(np.abs(delta.detach().cpu().numpy()).max())
    return sal


# ---------------------------------------------------------------------------
# One configuration -> one row of metrics (the parameterised experiment)
# ---------------------------------------------------------------------------

def benchmark_hotspots(model, samples, skempi_df=None, hotspot_thresh=2.0,
                       ig_steps=200, gnnx_epochs=80, device=None,
                       cache_dir="./data"):
    """Pool every pristine demo complex with SKEMPI hotspots and score each xAI
    method's hotspot recovery by AUROC.

    For each complex we run all three saliencies, line them up against the
    SKEMPI ΔΔG hotspot mask on the residues SKEMPI actually measured, and pool
    across complexes so the AUROC is computed over as many labelled residues as
    the demo set offers (a miniature of the real PINDER↔SKEMPI benchmark).

    Returns a flat dict: ``auroc_<method>`` for each method (omitted if the
    pooled labels have no class balance), plus ``pooled_measured``,
    ``pooled_hotspots`` and ``ig_delta_max``.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if skempi_df is None:
        skempi_df = fetch_skempi(cache_dir)
    methods = ("attention", "ig", "gnnexplainer", "consensus")
    pooled = {m: [] for m in methods}
    y_all, ig_deltas = [], []

    for s in (s for s in samples if not s["augmented"]):
        sal = compute_all_saliencies(model, s["ga"], s["gb"], ig_steps=ig_steps,
                                     gnnx_epochs=gnnx_epochs, device=device)
        ig_deltas.append(sal["ig_delta"])
        path = fetch_pdb(s["pdb"], cache_dir)
        _, _, rn_a = parse_chain(path, s["chain_a"])
        _, _, rn_d = parse_chain(path, s["chain_b"])
        ddg = skempi_ddg(skempi_df, s["pdb"])
        ddg_a, hot_a, _ = map_ddg_to_nodes(ddg, s["chain_a"], rn_a, hotspot_thresh)
        ddg_d, hot_d, _ = map_ddg_to_nodes(ddg, s["chain_b"], rn_d, hotspot_thresh)
        for key, ddg_n, hot in (("a", ddg_a, hot_a), ("d", ddg_d, hot_d)):
            m = np.isfinite(ddg_n)
            if m.sum() == 0:
                continue
            y_all.append(hot[m].astype(int))
            for meth in methods:
                pooled[meth].append(np.asarray(sal[key][meth])[m])

    y = np.concatenate(y_all) if y_all else np.array([])
    out = {"pooled_measured": int(y.size),
           "pooled_hotspots": int(y.sum()) if y.size else 0,
           "ig_delta_max": float(max(ig_deltas)) if ig_deltas else float("nan")}
    if y.size and 0 < y.sum() < y.size:
        for meth in methods:
            out[f"auroc_{meth}"] = float(roc_auc_score(y, np.concatenate(pooled[meth])))
    return out


def run_experiment(dropout=0.0, weight_decay=0.0, hidden=32, epochs=80, lr=5e-3,
                   augment=6, seed=0, threshold=8.0, ig_steps=200, gnnx_epochs=80,
                   dataset="demo", device=None, cache_dir="./data"):
    """Train + benchmark ONE configuration end to end; return a flat metrics dict.

    This is the single source of truth behind ``ig/run_idea4.py`` and the
    parameterised notebooks: change the hyper-parameters, get one comparable row
    of metrics (train accuracy + per-method hotspot AUROC + IG delta).

    ``dataset='pinder'`` is reserved for the real-data path (option *b*), which is
    intentionally not wired here — it raises so a scale-up run can't silently
    fall back to the toy set.
    """
    import torch

    if dataset != "demo":
        raise NotImplementedError(
            f"dataset='{dataset}' is not wired up. The PINDER loader is option "
            "(b), on hold until agreed; use dataset='demo' for now.")
    device = device or get_device()
    samples = build_demo_dataset(threshold=threshold, augment=augment, seed=seed,
                                 cache_dir=cache_dir)
    model = train_struct2graph(samples, hidden=hidden, epochs=epochs, lr=lr,
                               seed=seed, device=device, verbose=False,
                               dropout=dropout, weight_decay=weight_decay)
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in samples:
            pred = int(model(s["ga"].to(device), s["gb"].to(device)).argmax(1))
            correct += (pred == s["label"])
    metrics = {"train_acc": correct / len(samples),
               "n_samples": len(samples),
               "n_classes": len({s["label"] for s in samples})}
    metrics.update(benchmark_hotspots(model, samples, ig_steps=ig_steps,
                                      gnnx_epochs=gnnx_epochs, device=device,
                                      cache_dir=cache_dir))
    return metrics
