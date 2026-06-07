"""
GNN Lesson 8: Contact Prediction (Edge-Level Tasks)
====================================================
What you'll learn
-----------------
- The third GNN task type: edge-level prediction (after node-level in L2,
  graph-level in L3).
- How to build a GNN encoder that reads a SEQUENCE graph but predicts
  STRUCTURAL contacts it cannot directly see.
- Pair scoring with a bilinear head: from node embeddings h_i, h_j to a
  contact probability.
- Standard evaluation metrics: Precision@L and AUC-ROC.

Contact maps: the problem
-------------------------
A "contact" between residues i and j means their CA atoms are within 8 Å
in 3D space AND they are more than 3 positions apart in sequence (to exclude
trivially bonded neighbours). Predicting the contact map from sequence alone
is a classical problem, historically solved by coevolution statistics (e.g.
DCA, CCMpred) and now dominated by deep learning (AlphaFold2's early
pairwise representation is essentially this).

Why a GNN?
----------
Each residue node aggregates information from its sequence neighbours. After
2-3 rounds of message passing each h_i captures local structural tendencies.
A pair scorer then looks at h_i and h_j together and predicts whether they
are spatially proximal — the signal the MLP cannot extract from individual
residue features.

Precision@L
-----------
The standard contact prediction metric: rank all long-range candidate pairs
by predicted probability, take the top-L pairs (L = protein length), and
report what fraction are true contacts. A random predictor scores around
the background contact rate (~0.05). A good shallow model reaches 0.2-0.4.
"""

import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_TRAIN_PROTEINS = 300        # synthetic mini-proteins for training
N_TEST_PROTEINS = 60
LENGTH_RANGE = (30, 60)       # residues; short enough to be CPU-fast
SEQ_WINDOW = 3                # edges in the input sequence graph: ±3 neighbours
CONTACT_THRESHOLD = 8.0       # Å, CA-CA distance cutoff
MIN_SEQ_SEP = 4               # ignore trivially bonded pairs (i,j with |i-j|<=3)
HIDDEN = 64                   # GNN hidden dim
GNN_LAYERS = 3                # depth of the encoder
EPOCHS = 25
LR = 1e-3
PAIRS_PER_PROTEIN = 40        # balanced pos+neg pairs sampled per protein per epoch
NEG_POS_RATIO = 3             # negatives per positive in training pairs
OUTPUT_DIR = "./results"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
N_AA = len(AMINO_ACIDS)


# ---------------------------------------------------------------------------
# Synthetic protein generator (parametric 3D coords)
# ---------------------------------------------------------------------------

def make_helix_coords(n_residues, rise=1.5, radius=2.3, omega=100.0):
    """Return (n, 3) CA coordinates for an ideal alpha-helix.

    Parameters follow standard helix geometry: rise per residue ~1.5 Å,
    radius ~2.3 Å, rotation ~100° per residue.
    """
    coords = []
    for i in range(n_residues):
        angle = math.radians(omega * i)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = rise * i
        coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def make_hairpin_coords(n_residues, strand_rise=3.5, hairpin_bend=30.0):
    """Return (n, 3) CA coordinates for a simple beta-hairpin.

    Two strands running antiparallel with a short loop in the middle.
    Strand rise per residue is ~3.5 Å for a beta-strand.
    """
    half = n_residues // 2
    coords = []
    for i in range(half):
        coords.append([0.0, 0.0, strand_rise * i])
    bend_origin = np.array([hairpin_bend, 0.0, strand_rise * (half - 1)])
    for i in range(n_residues - half):
        coords.append([
            bend_origin[0],
            0.0,
            bend_origin[2] - strand_rise * i,
        ])
    return np.array(coords, dtype=np.float32)


def make_synthetic_protein(rng, length_range=LENGTH_RANGE):
    """Generate one synthetic mini-protein with a random sequence and
    a helix or hairpin backbone.  Returns (sequence_str, coords_array)."""
    L = rng.randint(*length_range)
    seq = "".join(rng.choice(AMINO_ACIDS) for _ in range(L))
    if rng.random() < 0.5:
        coords = make_helix_coords(L)
    else:
        coords = make_hairpin_coords(L)
    # Add small Gaussian noise to make coordinates non-degenerate.
    coords += rng.standard_normal(coords.shape).astype(np.float32) * 0.3
    return seq, coords


def contact_matrix(coords, threshold=CONTACT_THRESHOLD, min_sep=MIN_SEQ_SEP):
    """Return boolean (n, n) matrix: True where CA-CA distance < threshold
    and sequence separation > min_sep."""
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]      # (n, n, 3)
    dist = np.sqrt((diff ** 2).sum(-1))                  # (n, n)
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return (dist < threshold) & (sep > min_sep)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def sequence_graph(seq):
    """Sequence-window graph: node features = one-hot AA; edges = ±SEQ_WINDOW."""
    n = len(seq)
    x = torch.zeros(n, N_AA)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            x[i, AA_TO_IDX[aa]] = 1.0

    edges = []
    for i in range(n):
        for d in range(1, SEQ_WINDOW + 1):
            if i + d < n:
                edges.append((i, i + d))
                edges.append((i + d, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def sample_pairs(contacts, rng, neg_pos_ratio=NEG_POS_RATIO):
    """Return balanced (i, j, label) pairs for training.

    Positives: all true contact pairs with |i-j| > MIN_SEQ_SEP.
    Negatives: randomly sampled non-contact pairs, capped at neg_pos_ratio * #pos.
    """
    n = contacts.shape[0]
    pos_pairs, neg_pairs = [], []
    for i in range(n):
        for j in range(i + MIN_SEQ_SEP + 1, n):
            if contacts[i, j]:
                pos_pairs.append((i, j, 1))
            else:
                neg_pairs.append((i, j, 0))
    rng.shuffle(neg_pairs)
    neg_pairs = neg_pairs[: neg_pos_ratio * max(len(pos_pairs), 1)]
    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)
    return all_pairs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GNNContactPredictor(nn.Module):
    """Sequence-graph GNN encoder + bilinear pair scorer.

    Encoder
    -------
    Stack of SAGEConv layers that propagate amino-acid identity information
    along sequence edges.  After GNN_LAYERS rounds each h_i has seen
    GNN_LAYERS * SEQ_WINDOW residues on either side.

    Pair scorer
    -----------
    For candidate pair (i, j): concatenate [h_i, h_j] -> MLP -> scalar logit.
    Because contacts are symmetric, we also add the reverse [h_j, h_i] and
    average — making the scorer invariant to (i,j) order.
    """

    def __init__(self, in_channels=N_AA, hidden=HIDDEN, n_layers=GNN_LAYERS):
        super().__init__()
        convs = [SAGEConv(in_channels, hidden)]
        for _ in range(n_layers - 1):
            convs.append(SAGEConv(hidden, hidden))
        self.convs = nn.ModuleList(convs)

        # Pair scoring MLP: input is [h_i || h_j], dim = 2 * hidden
        self.pair_scorer = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        """Run GNN encoder; returns node embeddings of shape (n, hidden)."""
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return h

    def score_pairs(self, h, pair_indices):
        """Score a list of (i, j) pairs.

        pair_indices: LongTensor of shape (P, 2)
        Returns: (P,) logits (pre-sigmoid).
        """
        hi = h[pair_indices[:, 0]]      # (P, hidden)
        hj = h[pair_indices[:, 1]]      # (P, hidden)
        ij = self.pair_scorer(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        ji = self.pair_scorer(torch.cat([hj, hi], dim=-1)).squeeze(-1)
        return (ij + ji) * 0.5          # symmetric average

    def forward(self, data, pair_indices):
        h = self.encode(data.x, data.edge_index)
        return self.score_pairs(h, pair_indices)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataset, opt, device, rng):
    """One full pass over the dataset; each protein contributes sampled pairs."""
    model.train()
    total_loss, n_batches = 0.0, 0
    # Shuffle protein order each epoch.
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for idx in indices:
        graph, contacts, _ = dataset[idx]
        pairs = sample_pairs(contacts, rng)
        if not pairs:
            continue
        graph = graph.to(device)
        pair_idx = torch.tensor([[p[0], p[1]] for p in pairs], dtype=torch.long).to(device)
        labels = torch.tensor([p[2] for p in pairs], dtype=torch.float).to(device)
        opt.zero_grad()
        logits = model(graph, pair_idx)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def precision_at_L(model, dataset, device):
    """Mean Precision@L over all proteins in dataset.

    For each protein with L residues: take the top-L scoring long-range
    pairs and return the fraction that are true contacts.
    """
    model.eval()
    precisions = []
    for graph, contacts, seq_len in dataset:
        n = seq_len
        # All long-range candidate pairs.
        cand = [(i, j) for i in range(n) for j in range(i + MIN_SEQ_SEP + 1, n)]
        if not cand:
            continue
        graph = graph.to(device)
        pair_idx = torch.tensor(cand, dtype=torch.long).to(device)
        logits = model(graph, pair_idx)
        probs = torch.sigmoid(logits).cpu().numpy()
        order = np.argsort(-probs)          # descending
        top_L = order[:n]                   # top-L predictions
        n_correct = sum(contacts[cand[k][0], cand[k][1]] for k in top_L)
        precisions.append(n_correct / len(top_L))
    return float(np.mean(precisions))


@torch.no_grad()
def mean_auc(model, dataset, device):
    """Mean per-protein AUC-ROC over all candidate long-range pairs."""
    model.eval()
    aucs = []
    for graph, contacts, seq_len in dataset:
        n = seq_len
        cand = [(i, j) for i in range(n) for j in range(i + MIN_SEQ_SEP + 1, n)]
        labels = np.array([int(contacts[i, j]) for i, j in cand])
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue    # AUC undefined if only one class present
        graph = graph.to(device)
        pair_idx = torch.tensor(cand, dtype=torch.long).to(device)
        probs = torch.sigmoid(model(graph, pair_idx)).cpu().numpy()
        aucs.append(roc_auc_score(labels, probs))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_contact_maps(model, graph, contacts, seq_len, save_path, device):
    """Save side-by-side true vs predicted contact map for one protein."""
    model.eval()
    n = seq_len
    cand = [(i, j) for i in range(n) for j in range(n) if abs(i - j) > MIN_SEQ_SEP]
    pair_idx = torch.tensor(cand, dtype=torch.long).to(device)
    graph = graph.to(device)
    probs = torch.sigmoid(model(graph, pair_idx)).cpu().numpy()

    pred_map = np.zeros((n, n))
    for (i, j), p in zip(cand, probs):
        pred_map[i, j] = p
        pred_map[j, i] = p     # symmetrise for display

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(contacts.astype(float), cmap="Blues", origin="upper")
    axes[0].set_title("True contact map")
    axes[0].set_xlabel("Residue j")
    axes[0].set_ylabel("Residue i")

    axes[1].imshow(pred_map, cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Predicted contact probabilities")
    axes[1].set_xlabel("Residue j")
    axes[1].set_ylabel("Residue i")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(n_proteins, seed):
    """Return list of (graph, contacts_bool_array, seq_len)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    dataset = []
    for _ in range(n_proteins):
        seq, coords = make_synthetic_protein(rng)
        contacts = contact_matrix(coords)
        graph = sequence_graph(seq)
        dataset.append((graph, contacts, len(seq)))
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nGenerating synthetic mini-proteins...")
    train_set = build_dataset(N_TRAIN_PROTEINS, seed=SEED)
    test_set = build_dataset(N_TEST_PROTEINS, seed=SEED + 1)

    # Dataset statistics
    contact_rates = []
    for _, contacts, n in train_set:
        n_cand = sum(1 for i in range(n) for j in range(i + MIN_SEQ_SEP + 1, n))
        n_pos = sum(
            1 for i in range(n) for j in range(i + MIN_SEQ_SEP + 1, n) if contacts[i, j]
        )
        if n_cand > 0:
            contact_rates.append(n_pos / n_cand)
    mean_contact_rate = float(np.mean(contact_rates))
    print(f"  Train proteins: {len(train_set)},  Test proteins: {len(test_set)}")
    print(f"  Mean long-range contact rate: {mean_contact_rate:.3f}")
    print(f"  Random-predictor Precision@L ~ {mean_contact_rate:.3f}")

    model = GNNContactPredictor(in_channels=N_AA, hidden=HIDDEN, n_layers=GNN_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_rng = random.Random(SEED)

    print("\nTraining contact predictor...")
    for ep in range(EPOCHS):
        loss = train_epoch(model, train_set, opt, device, train_rng)
        if (ep + 1) % 5 == 0:
            prec = precision_at_L(model, test_set, device)
            auc = mean_auc(model, test_set, device)
            print(f"  epoch {ep + 1:3d}  loss={loss:.4f}  Precision@L={prec:.3f}  AUC={auc:.3f}")

    print("\n--- Final evaluation on test set ---")
    final_prec = precision_at_L(model, test_set, device)
    final_auc = mean_auc(model, test_set, device)
    print(f"  Precision@L : {final_prec:.3f}  (random baseline ~ {mean_contact_rate:.3f})")
    print(f"  Mean AUC    : {final_auc:.3f}  (random baseline ~ 0.500)")
    lift = final_prec / max(mean_contact_rate, 1e-9)
    print(f"  Lift over random: {lift:.1f}x")

    # Visualise one test protein
    graph, contacts, seq_len = test_set[0]
    fig_path = os.path.join(OUTPUT_DIR, "gnn_l8_contact_maps.png")
    plot_contact_maps(model, graph, contacts, seq_len, fig_path, device)
    print(f"\nContact map figure saved to: {fig_path}")

    print(
        """
Things to experiment with:
- Swap one-hot node features for ESM-2 residue embeddings (from Lesson 4) —
  this is the single biggest lift and mirrors real-world practice.
- Separate Precision@L into short (6<sep<=12), medium (12<sep<=24), and
  long-range (sep>24) bins; long-range is the hardest and most informative.
- Replace the MLP pair scorer with a symmetric bilinear head:
  score(i,j) = h_i^T W h_j — fewer parameters, enforces symmetry by design.
- Add coevolution-style pair features: concatenate [h_i, h_j, |h_i - h_j|,
  h_i * h_j] to give the scorer more geometric signal.
- Train on real PDB structures (use Lesson 6's data loader) and evaluate
  on CASP targets to benchmark against classical methods.
- Increase GNN_LAYERS to 4-5: each extra layer extends the receptive field
  by SEQ_WINDOW residues on each side.
"""
    )


if __name__ == "__main__":
    main()
