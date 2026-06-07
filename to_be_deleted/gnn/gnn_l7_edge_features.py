"""
GNN Lesson 7: Edge Features & Geometry
=======================================
What you'll learn
-----------------
- Why plain GCN/GAT ignores edge attributes and when that's a problem.
- How to compute informative edge features: Euclidean distance, RBF distance
  bins, sequence separation, and unit direction vectors.
- How NNConv uses a small network to turn edge features into per-edge weight
  matrices, so edge geometry directly modulates message passing.
- That for geometry-dependent tasks, an edge-aware model consistently
  out-performs an identical node-only GCN.

Why edge features matter in proteins
-------------------------------------
Standard GCN message passing aggregates NEIGHBOUR NODE features. It knows an
edge exists but not HOW FAR APART the two residues are, HOW SEQUENTIALLY
DISTANT they are, or in WHICH DIRECTION the bond runs. For many protein signals
— packing density, long-range contacts, hydrogen-bond geometry — this geometric
information is exactly what separates structured folds. Ignoring it means the
model is blind to the very signal it needs.

NNConv: edge-conditioned convolution
-------------------------------------
NNConv (Gilmer et al. 2017, "Neural Message Passing for Quantum Chemistry")
computes per-edge weight matrices via a small MLP applied to edge_attr, then
uses them to transform neighbour features before aggregation:

    m_ij = MLP(edge_attr_ij) * h_j        # (hidden x in_channels) matrix mul
    h_i' = AGG_{j in N(i)} m_ij           # sum/mean aggregation

The weight matrix is DIFFERENT for every edge — the model can learn to
weight long-range vs short-range contacts differently.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SEED = 42
N_TRAIN = 300           # graphs in training set
N_TEST = 75             # graphs in test set
LENGTH_RANGE = (25, 45) # residues per chain (short for CPU speed)
CONTACT_CUTOFF = 8.0    # Å — CA-CA contact threshold
N_RBF = 8               # number of Gaussian RBF distance bins
RBF_CUTOFF = 10.0       # Å — max distance for RBF encoding
COMPACT_THRESHOLD = 5.0 # Å mean-contact-distance cutoff for the graph label
HIDDEN = 32             # hidden dimension in both models
EDGE_NN_HIDDEN = 16     # hidden size of NNConv's internal MLP
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 16
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
NODE_IN = len(AMINO_ACIDS)  # 20 one-hot features per residue


# ---------------------------------------------------------------------------
# Synthetic protein coordinate generators
# ---------------------------------------------------------------------------

def helix_coords(n, rise=1.5, radius=2.3, turn_deg=100.0, noise=0.1, rng=None):
    """Generate Cα coordinates for an idealized alpha-helix.

    Rise ~1.5 Å per residue and ~100° rotation per residue are standard
    helix parameters. Small Gaussian noise makes each sample unique.
    """
    if rng is None:
        rng = np.random.default_rng()
    angles = np.deg2rad(turn_deg * np.arange(n))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = rise * np.arange(n)
    coords = np.stack([x, y, z], axis=1)
    coords += rng.normal(0, noise, size=coords.shape)
    return coords


def hairpin_coords(n, strand_sep=5.0, rise=3.8, noise=0.1, rng=None):
    """Generate Cα coordinates for a simple beta-hairpin.

    Two antiparallel strands connected by a short loop. The strands are
    separated by `strand_sep` Å. Beta strands have ~3.8 Å rise per residue.
    Hairpins are more EXTENDED than helices — radius of gyration is larger,
    so the geometry-based label separates them cleanly.
    """
    if rng is None:
        rng = np.random.default_rng()
    half = n // 2
    # Strand 1: extends along +z
    z1 = rise * np.arange(half)
    x1 = np.zeros(half)
    # Strand 2: runs antiparallel back along -z, offset by strand_sep in x
    z2 = rise * (half - 1 - np.arange(n - half))
    x2 = np.full(n - half, strand_sep)
    x = np.concatenate([x1, x2])
    y = np.zeros(n)
    z = np.concatenate([z1, z2])
    coords = np.stack([x, y, z], axis=1)
    coords += rng.normal(0, noise, size=coords.shape)
    return coords


# ---------------------------------------------------------------------------
# Edge feature computation
# ---------------------------------------------------------------------------

def rbf_encode(distances, n_rbf=N_RBF, d_max=RBF_CUTOFF):
    """Encode scalar distances as a vector of Gaussian RBF values.

    RBF (radial basis functions) are smoother than a single distance value
    and help the model learn distance-dependent patterns in different ranges.
    Centers are evenly spaced in [0, d_max]; width is set to half the spacing.
    """
    centers = np.linspace(0.0, d_max, n_rbf)
    width = (d_max / (n_rbf - 1)) / 2.0  # half the spacing
    rbf = np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
    return rbf.astype(np.float32)  # (E, n_rbf)


def build_contact_graph(sequence, coords, threshold=CONTACT_CUTOFF):
    """Build a contact graph and compute edge features for every edge.

    Edge feature vector (per edge):
        [0]          : raw Euclidean distance (1 dim)
        [1 .. n_rbf] : RBF-encoded distance   (N_RBF dims)
        [n_rbf+1]    : normalised sequence separation |i-j| / L  (1 dim)
        [n_rbf+2..4] : unit direction vector (u_x, u_y, u_z)      (3 dims)

    Total: 1 + N_RBF + 1 + 3 = N_RBF + 5 dims.
    """
    n = len(sequence)
    diff = coords[:, None, :] - coords[None, :, :]   # (n, n, 3)
    dist = np.linalg.norm(diff, axis=-1)              # (n, n)

    mask = (dist < threshold) & (dist > 0)
    src, dst = np.where(mask)

    raw_dist = dist[src, dst]                         # (E,)
    # Unit direction: from src to dst
    direction = coords[dst] - coords[src]             # (E, 3)
    direction /= (raw_dist[:, None] + 1e-8)

    seq_sep = np.abs(src - dst).astype(np.float32) / max(n - 1, 1)  # normalised
    rbf = rbf_encode(raw_dist)                        # (E, N_RBF)

    edge_feat = np.concatenate([
        raw_dist[:, None],  # raw distance
        rbf,                # RBF bins
        seq_sep[:, None],   # sequence separation
        direction,          # unit vector
    ], axis=1)  # (E, N_RBF + 5)

    # Graph-level label: compact (1) if mean contact distance is below threshold
    label = int(raw_dist.mean() < COMPACT_THRESHOLD) if len(raw_dist) > 0 else 0

    # Node features: one-hot AA identity
    x = torch.zeros(n, NODE_IN)
    for i, aa in enumerate(sequence):
        x[i, AA_TO_IDX[aa]] = 1.0

    return Data(
        x=x,
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(edge_feat, dtype=torch.float32),
        y=torch.tensor([label], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_dataset(n_graphs, length_range=LENGTH_RANGE, seed=0):
    """Generate a balanced dataset of helix and hairpin graphs.

    Each graph is labelled by whether its mean contact distance is below
    COMPACT_THRESHOLD. Helices are naturally more compact than hairpins,
    so edge geometry predicts the label well; node identity (random AA) does not.
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    data_list = []
    for i in range(n_graphs):
        L = rng_py.randint(*length_range)
        seq = "".join(rng_py.choice(AMINO_ACIDS) for _ in range(L))
        # Alternate folds so the dataset is balanced
        if i % 2 == 0:
            coords = helix_coords(L, rng=rng_np)
        else:
            coords = hairpin_coords(L, rng=rng_np)
        data_list.append(build_contact_graph(seq, coords))
    return data_list


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PlainGCN(nn.Module):
    """Two-layer GCN that completely ignores edge_attr.

    GCNConv normalises and averages neighbour features — it has no mechanism
    to weight neighbours differently based on HOW they are connected. For
    geometry-dependent tasks this is a fundamental limitation.
    """

    def __init__(self, in_ch, hidden, num_classes=2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr intentionally unused — that is the point of this baseline
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = global_mean_pool(h, batch)
        return self.head(h)


class EdgeAwareGNN(nn.Module):
    """Two-layer NNConv model that uses edge_attr to modulate messages.

    NNConv applies a small MLP to each edge's feature vector to produce a
    (hidden x in_channels) weight matrix. This matrix transforms the source
    node's features before they are aggregated at the target. Different
    geometric contexts produce different weight matrices — the model learns
    to treat a close short-range contact differently from a distant long-range
    one.
    """

    def __init__(self, in_ch, hidden, edge_dim, nn_hidden=EDGE_NN_HIDDEN, num_classes=2):
        super().__init__()
        # NNConv needs a network mapping edge_dim -> (in_ch * hidden)
        edge_nn1 = nn.Sequential(
            nn.Linear(edge_dim, nn_hidden),
            nn.ReLU(),
            nn.Linear(nn_hidden, in_ch * hidden),
        )
        self.conv1 = NNConv(in_ch, hidden, edge_nn1, aggr="mean")

        edge_nn2 = nn.Sequential(
            nn.Linear(edge_dim, nn_hidden),
            nn.ReLU(),
            nn.Linear(nn_hidden, hidden * hidden),
        )
        self.conv2 = NNConv(hidden, hidden, edge_nn2, aggr="mean")
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = global_mean_pool(h, batch)
        return self.head(h)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.numel()
    return correct / total


def run_training(model, train_loader, test_loader, device, label):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        if ep % 10 == 0:
            acc = evaluate(model, test_loader, device)
            history.append((ep, loss, acc))
            print(f"  [{label}] epoch {ep:3d}  loss={loss:.3f}  test_acc={acc:.3f}")
    return evaluate(model, test_loader, device), history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("./results", exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 1. Dataset -------------------------------------------------------
    print("\nGenerating synthetic protein graphs (helix vs hairpin)...")
    train_set = make_dataset(N_TRAIN, seed=SEED)
    test_set = make_dataset(N_TEST, seed=SEED + 1)

    n_pos_train = sum(d.y.item() == 1 for d in train_set)
    edge_dim = train_set[0].edge_attr.shape[1]
    print(f"  Train: {len(train_set)} graphs | Test: {len(test_set)} graphs")
    print(f"  Compact (label=1) fraction in train: {n_pos_train / len(train_set):.2f}")
    print(f"  Edge feature dimension: {edge_dim}  "
          f"(1 raw dist + {N_RBF} RBF + 1 seq-sep + 3 direction)")
    sample = train_set[0]
    print(f"  Sample graph: {sample.num_nodes} nodes, {sample.num_edges} edges")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # ---- 2. Train plain GCN (no edge features) ----------------------------
    print("\n[1/2] Training PlainGCN (ignores edge geometry)...")
    gcn = PlainGCN(in_ch=NODE_IN, hidden=HIDDEN)
    gcn_acc, gcn_hist = run_training(gcn, train_loader, test_loader, device, "GCN")
    print(f"  => Final PlainGCN test accuracy: {gcn_acc:.3f}")

    # ---- 3. Train edge-aware NNConv model ---------------------------------
    print("\n[2/2] Training EdgeAwareGNN (NNConv uses edge_attr)...")
    edge_gnn = EdgeAwareGNN(in_ch=NODE_IN, hidden=HIDDEN, edge_dim=edge_dim)
    edge_acc, edge_hist = run_training(edge_gnn, train_loader, test_loader, device, "EdgeGNN")
    print(f"  => Final EdgeAwareGNN test accuracy: {edge_acc:.3f}")

    # ---- 4. Summary -------------------------------------------------------
    delta = edge_acc - gcn_acc
    print(f"\nDelta (EdgeAwareGNN - PlainGCN): {delta:+.3f}")
    if delta > 0.03:
        print("  Edge-aware model wins — geometry in edge_attr is doing real work.")
    elif delta > -0.03:
        print("  Models roughly tied — both learned the task or the signal is weak.")
    else:
        print("  GCN leading — try more epochs or check label balance.")

    # ---- 5. Plot accuracy curves -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    if gcn_hist:
        eps_g, _, accs_g = zip(*gcn_hist)
        ax.plot(eps_g, accs_g, marker="o", label="PlainGCN (no edge attr)")
    if edge_hist:
        eps_e, _, accs_e = zip(*edge_hist)
        ax.plot(eps_e, accs_e, marker="s", label="EdgeAwareGNN (NNConv)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title("GNN Lesson 7: Edge Features & Geometry")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savepath = "./results/gnn_l7_edge_features.png"
    fig.savefig(savepath, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved accuracy plot to {savepath}")

    print(
        """
Things to experiment with:
- Increase N_RBF (e.g. 16) for finer distance resolution — does test accuracy
  improve? At what point do you see diminishing returns?
- Add backbone dihedral edge features (phi/psi angles computed from 4 Cα
  positions). These are strong signals for helix vs strand classification.
- Swap NNConv for TransformerConv (torch_geometric.nn.TransformerConv) which
  accepts edge_dim directly and uses multi-head attention over edges.
- Replace synthetic coordinates with real PDB structures from Lesson 6 and
  predict secondary structure (helix/sheet/coil) as a 3-class problem.
- Turn the label into a regression target: predict the radius of gyration
  directly, switching from CrossEntropy to MSELoss.
"""
    )


if __name__ == "__main__":
    main()
