"""
GNN Lesson 10: Oversmoothing and Depth
=======================================
What you'll learn
-----------------
- Why stacking many GNN layers hurts rather than helps (oversmoothing).
- How to measure oversmoothing: mean pairwise cosine similarity of final
  node embeddings rises toward 1 as depth increases.
- How test accuracy degrades once depth exceeds the graph diameter.
- How residual/skip connections keep deep GCNs accurate.

The oversmoothing problem
-------------------------
Each GCN layer averages a node's features with its neighbours. After k layers
every node has seen its k-hop neighbourhood. Sounds good — until k is large
relative to the graph diameter: at that point ALL nodes have "seen" the whole
graph and their representations converge to a single shared vector. The model
loses the ability to distinguish them. This is oversmoothing.

Diagnosing and fixing it
------------------------
We measure it with MEAN PAIRWISE COSINE SIMILARITY (MPCS) of the final-layer
node embeddings: a perfectly smooth graph gives MPCS = 1.0. We also measure
DIRICHLET ENERGY (sum of squared differences across edges), which collapses to
0 under the same condition. Then we add RESIDUAL CONNECTIONS — each layer
adds its input back to its output — which lets gradients and raw signals
bypass the averaging operation and prevents full convergence.
"""

import os
import random

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

AMINO_ACIDS   = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC   = set("AVLIFMWY")
AA_TO_IDX     = {a: i for i, a in enumerate(AMINO_ACIDS)}

N_TRAIN       = 200          # synthetic graphs for training
N_TEST        = 50           # held-out graphs
SEQ_LEN_RANGE = (40, 120)    # residues per protein
WINDOW        = 3            # sequence-graph connectivity (±3)
HYDRO_THRESH  = 0.6          # fraction of hydrophobic neighbours needed for label=1

HIDDEN        = 32           # hidden dimension for all models
EPOCHS        = 30           # training epochs per model
LR            = 1e-2         # Adam learning rate
BATCH_SIZE    = 8

DEPTHS        = [1, 2, 3, 4, 5, 6, 7, 8]   # depths to benchmark

RESULTS_DIR   = "./results"


# ---------------------------------------------------------------------------
# Dataset helpers (from Lesson 2)
# ---------------------------------------------------------------------------

def one_hot_aa(sequence):
    n = len(sequence)
    x = torch.zeros(n, len(AMINO_ACIDS))
    for i, aa in enumerate(sequence):
        x[i, AA_TO_IDX[aa]] = 1.0
    return x


def make_edge_index(n, window=WINDOW):
    """Sequence-window graph: residue i connects to residues i±1 … i±window."""
    edges = []
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                edges.append((i, i + d))
                edges.append((i + d, i))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def label_residues(sequence, window=WINDOW, hydro_threshold=HYDRO_THRESH):
    """Label = 1 if the residue is hydrophobic AND its sequence-window
    neighbourhood exceeds `hydro_threshold` fraction of hydrophobic residues."""
    n = len(sequence)
    labels = []
    for i in range(n):
        if sequence[i] not in HYDROPHOBIC:
            labels.append(0)
            continue
        lo, hi = max(0, i - window), min(n, i + window + 1)
        neighbours = sequence[lo:i] + sequence[i + 1:hi]
        if not neighbours:
            labels.append(0)
            continue
        frac = sum(1 for aa in neighbours if aa in HYDROPHOBIC) / len(neighbours)
        labels.append(1 if frac >= hydro_threshold else 0)
    return torch.tensor(labels, dtype=torch.long)


def make_dataset(n_graphs=N_TRAIN, length_range=SEQ_LEN_RANGE, seed=0):
    rng = random.Random(seed)
    data_list = []
    for _ in range(n_graphs):
        L = rng.randint(*length_range)
        seq = "".join(rng.choice(AMINO_ACIDS) for _ in range(L))
        data_list.append(Data(
            x=one_hot_aa(seq),
            edge_index=make_edge_index(L),
            y=label_residues(seq),
        ))
    return data_list


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DeepGCN(torch.nn.Module):
    """Vanilla deep GCN with a configurable number of GCNConv layers.

    Each layer applies: h = ReLU(GCNConv(h))
    No skip connections — this is the model that oversmoothes.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, depth):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Output layer (only added if depth >= 2)
        if depth >= 2:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            # depth == 1: replace the input conv's output dim
            self.convs = torch.nn.ModuleList([GCNConv(in_channels, out_channels)])

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=0.1, training=self.training)
        h = self.convs[-1](h, edge_index)
        return h

    @torch.no_grad()
    def final_embeddings(self, x, edge_index):
        """Return the pre-logit hidden representation (second-to-last layer output)."""
        self.eval()
        h = x
        # Run all layers except the last classifier head
        layers = self.convs[:-1] if len(self.convs) > 1 else self.convs
        for conv in layers:
            h = F.relu(conv(h, edge_index))
        return h


class ResidualGCN(torch.nn.Module):
    """Deep GCN with residual (skip) connections.

    Each hidden layer computes: h = ReLU(GCNConv(h)) + h
    The addition lets raw signals flow through, preventing the averaging from
    washing out all variation. A linear projection aligns input dimension with
    the hidden dimension for the first layer.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, depth):
        super().__init__()
        # Project input features to hidden_channels once
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.convs = torch.nn.ModuleList()
        for _ in range(max(depth - 1, 1)):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h   # residual addition
            h = F.dropout(h, p=0.1, training=self.training)
        return self.classifier(h)

    @torch.no_grad()
    def final_embeddings(self, x, edge_index):
        self.eval()
        h = self.input_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h
        return h


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
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
        pred = model(batch.x, batch.edge_index).argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.numel()
    return correct / total


def train_model(model, train_loader, test_loader, device, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device)
    return evaluate(model, test_loader, device)


# ---------------------------------------------------------------------------
# Oversmoothing metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_pairwise_cosine_similarity(model, loader, device):
    """Average over graphs: mean cosine similarity of all node-embedding pairs.

    A score near 1.0 indicates oversmoothing — all nodes look alike.
    We subsample at most 64 nodes per graph to keep it tractable.
    """
    model.eval()
    scores = []
    for batch in loader:
        batch = batch.to(device)
        # Collect per-graph node ranges via batch.batch tensor
        embs = model.final_embeddings(batch.x, batch.edge_index)  # (N_total, H)
        batch_vec = batch.batch
        for g in batch_vec.unique():
            mask = batch_vec == g
            h = embs[mask]          # (n_nodes, H)
            if h.size(0) < 2:
                continue
            # Subsample to 64 nodes to cap quadratic cost
            if h.size(0) > 64:
                idx = torch.randperm(h.size(0), device=device)[:64]
                h = h[idx]
            h_norm = F.normalize(h, dim=-1)                     # unit vectors
            sim_matrix = h_norm @ h_norm.t()                    # (n, n) cosines
            # Mean of upper triangle (excluding diagonal)
            n = h.size(0)
            triu_mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
            scores.append(sim_matrix[triu_mask].mean().item())
    return sum(scores) / len(scores) if scores else 0.0


@torch.no_grad()
def dirichlet_energy(model, loader, device):
    """Sum of squared L2 differences across edges, normalised by edge count.

    Collapses toward 0 when all connected nodes have the same representation.
    """
    model.eval()
    energies = []
    for batch in loader:
        batch = batch.to(device)
        embs = model.final_embeddings(batch.x, batch.edge_index)
        src, dst = batch.edge_index
        diff = embs[src] - embs[dst]                          # (E, H)
        energy = (diff ** 2).sum(dim=-1).mean().item()        # mean squared diff
        energies.append(energy)
    return sum(energies) / len(energies) if energies else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nGenerating synthetic protein graphs...")
    train_set = make_dataset(n_graphs=N_TRAIN, seed=0)
    test_set  = make_dataset(n_graphs=N_TEST,  seed=1)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

    pos   = sum((d.y == 1).sum().item() for d in train_set)
    total = sum(d.y.numel() for d in train_set)
    print(f"  Train / test graphs : {len(train_set)} / {len(test_set)}")
    print(f"  Positive label rate : {pos / total:.3f}")
    print(f"  Majority-class baseline: {max(pos / total, 1 - pos / total):.3f}")

    # -----------------------------------------------------------------------
    # Benchmark vanilla GCN at each depth
    # -----------------------------------------------------------------------
    print(f"\n[1/2] Vanilla DeepGCN — depths {DEPTHS}")
    vanilla_acc  = []
    vanilla_mpcs = []
    vanilla_de   = []

    for depth in DEPTHS:
        model = DeepGCN(len(AMINO_ACIDS), HIDDEN, 2, depth).to(device)
        acc = train_model(model, train_loader, test_loader, device)
        mpcs = mean_pairwise_cosine_similarity(model, test_loader, device)
        de   = dirichlet_energy(model, test_loader, device)
        vanilla_acc.append(acc)
        vanilla_mpcs.append(mpcs)
        vanilla_de.append(de)
        print(f"  depth={depth}  acc={acc:.3f}  MPCS={mpcs:.3f}  DirEnergy={de:.4f}")

    # -----------------------------------------------------------------------
    # Benchmark residual GCN at each depth
    # -----------------------------------------------------------------------
    print(f"\n[2/2] ResidualGCN (skip connections) — depths {DEPTHS}")
    residual_acc  = []
    residual_mpcs = []

    for depth in DEPTHS:
        model = ResidualGCN(len(AMINO_ACIDS), HIDDEN, 2, depth).to(device)
        acc  = train_model(model, train_loader, test_loader, device)
        mpcs = mean_pairwise_cosine_similarity(model, test_loader, device)
        residual_acc.append(acc)
        residual_mpcs.append(mpcs)
        print(f"  depth={depth}  acc={acc:.3f}  MPCS={mpcs:.3f}")

    # -----------------------------------------------------------------------
    # Before / after comparison at deep end
    # -----------------------------------------------------------------------
    print("\nBefore/after summary at depth 6 and 8:")
    print(f"  {'Depth':<6}  {'Vanilla acc':<13}  {'Residual acc':<13}  {'Vanilla MPCS':<14}  {'Residual MPCS'}")
    for i, d in enumerate(DEPTHS):
        if d in (6, 8):
            print(f"  {d:<6}  {vanilla_acc[i]:<13.3f}  {residual_acc[i]:<13.3f}  "
                  f"{vanilla_mpcs[i]:<14.3f}  {residual_mpcs[i]:.3f}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- Plot 1: accuracy vs depth ---
    ax = axes[0]
    ax.plot(DEPTHS, vanilla_acc,  "o-", label="Vanilla GCN",   color="steelblue")
    ax.plot(DEPTHS, residual_acc, "s-", label="Residual GCN",  color="darkorange")
    ax.set_xlabel("Number of GCN layers (depth)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs depth")
    ax.legend()
    ax.set_xticks(DEPTHS)

    # --- Plot 2: MPCS vs depth ---
    ax = axes[1]
    ax.plot(DEPTHS, vanilla_mpcs,  "o-", label="Vanilla GCN",  color="steelblue")
    ax.plot(DEPTHS, residual_mpcs, "s-", label="Residual GCN", color="darkorange")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="Perfect smooth (MPCS=1)")
    ax.set_xlabel("Number of GCN layers (depth)")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("Oversmoothing (MPCS) vs depth")
    ax.legend()
    ax.set_xticks(DEPTHS)

    # --- Plot 3: Dirichlet energy vs depth ---
    ax = axes[2]
    ax.plot(DEPTHS, vanilla_de, "o-", color="steelblue", label="Vanilla GCN")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8, label="Perfect smooth (energy=0)")
    ax.set_xlabel("Number of GCN layers (depth)")
    ax.set_ylabel("Mean Dirichlet energy")
    ax.set_title("Dirichlet energy vs depth\n(lower = more smooth)")
    ax.legend()
    ax.set_xticks(DEPTHS)

    fig.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "gnn_l10_oversmoothing.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {fig_path}")

    print(
        """
Things to experiment with:
- Add PairNorm after each conv layer (normalises node norms to prevent collapse)
  — import PairNorm from torch_geometric.nn and insert between conv and ReLU.
- Try DropEdge: randomly drop graph edges during training to prevent over-aggregation.
  Use torch_geometric.utils.dropout_edge(edge_index, p=0.3, training=self.training).
- Use JumpingKnowledge (torch_geometric.nn.JumpingKnowledge) to concatenate
  representations from ALL layers before the classifier — gives the model a choice
  of "which depth" to use for each node.
- Compute Dirichlet energy EXPLICITLY at each layer (not just the final one) and
  plot the collapse curve layer-by-layer for a 6-layer network.
- Vary graph connectivity: increase WINDOW from 3 to 8 — denser graphs propagate
  information faster and oversmoooth at shallower depths.
- Replace GCNConv with GATConv (Graph Attention Network) — does attention help
  resist oversmoothing, or does it collapse too?
"""
    )


if __name__ == "__main__":
    main()
