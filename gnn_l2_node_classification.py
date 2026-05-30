"""
GNN Lesson 2: Node Classification with a GCN
==============================================
What you'll learn
-----------------
- The simplest GNN: Graph Convolutional Network (GCN) by Kipf & Welling (2017).
- "Message passing" — each node updates its features by aggregating from neighbours.
- Train a 2-layer GCN on a per-residue prediction task.
- Why a GNN can outperform a per-residue MLP for context-dependent tasks.

The toy task
------------
Predict whether each residue is a "hydrophobic core residue", defined as:
  1. The residue itself is hydrophobic (A, V, L, I, F, M, W, Y), AND
  2. Most of its sequence-window neighbours are also hydrophobic.

Condition (2) makes this CONTEXT-DEPENDENT: a per-residue model that ignores
neighbours physically cannot solve it. A GNN can. We'll train both and compare.

Real protein analogues
----------------------
Many real per-residue tasks are similarly context-dependent:
  - Solvent accessibility (a residue is "buried" if surrounded by other residues)
  - Interface residues (binding face of a complex)
  - Catalytic site prediction
"""

import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AVLIFMWY")
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}


def one_hot_aa(sequence):
    n = len(sequence)
    x = torch.zeros(n, len(AMINO_ACIDS))
    for i, aa in enumerate(sequence):
        x[i, AA_TO_IDX[aa]] = 1.0
    return x


def make_edge_index(n, window=3):
    """Sequence-window graph: each residue connects to ±window neighbours."""
    edges = []
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                edges.append((i, i + d))
                edges.append((i + d, i))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def label_residues(sequence, window=3, hydro_threshold=0.6):
    """Per-residue label: 1 if residue is hydrophobic AND > threshold of its
    sequence-window neighbours are hydrophobic, else 0."""
    n = len(sequence)
    labels = []
    for i in range(n):
        if sequence[i] not in HYDROPHOBIC:
            labels.append(0)
            continue
        # Look at neighbours within `window` residues, excluding self.
        lo, hi = max(0, i - window), min(n, i + window + 1)
        neighbours = sequence[lo:i] + sequence[i + 1 : hi]
        if not neighbours:
            labels.append(0)
            continue
        frac = sum(1 for aa in neighbours if aa in HYDROPHOBIC) / len(neighbours)
        labels.append(1 if frac >= hydro_threshold else 0)
    return torch.tensor(labels, dtype=torch.long)


def make_dataset(n_graphs=200, length_range=(40, 120), seed=0):
    """Generate `n_graphs` random protein-like sequences with labels."""
    rng = random.Random(seed)
    data_list = []
    for _ in range(n_graphs):
        L = rng.randint(*length_range)
        seq = "".join(rng.choice(AMINO_ACIDS) for _ in range(L))
        data_list.append(
            Data(
                x=one_hot_aa(seq),
                edge_index=make_edge_index(L, window=3),
                y=label_residues(seq, window=3),
            )
        )
    return data_list


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GCN(torch.nn.Module):
    """A bog-standard 2-layer GCN.

    The Kipf-Welling layer's update rule is roughly:
        h_i' = sigma( W * mean_{j in N(i) union {i}} (h_j) )
    i.e. each node's new feature vector is a weighted average of its own
    feature vector and those of its neighbours, fed through a linear layer
    and a non-linearity. Stack two of these and each node's output is
    informed by its 2-hop neighbourhood.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        return self.conv2(h, edge_index)


class MLP(torch.nn.Module):
    """Per-node MLP — completely ignores edges. Baseline.

    Same number of layers and hidden size as the GCN, so the only difference
    between the two models is whether they USE the graph structure.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):  # edge_index argument unused — that's the point
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_and_eval(model, train_loader, test_loader, device, epochs=20, lr=1e-2):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # PyG's DataLoader BATCHES multiple graphs into a single big graph
            # with a `batch.batch` tensor that says which graph each node is in.
            # Message passing respects the batch boundaries automatically — no
            # edges between separate graphs.
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (ep + 1) % 5 == 0:
            acc = evaluate(model, test_loader, device)
            print(f"  epoch {ep + 1:3d}  loss={total_loss / len(train_loader):.3f}  test_acc={acc:.3f}")

    return evaluate(model, test_loader, device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.numel()
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nGenerating synthetic protein graphs...")
    train_set = make_dataset(n_graphs=200, seed=0)
    test_set = make_dataset(n_graphs=50, seed=1)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8)

    pos = sum((d.y == 1).sum().item() for d in train_set)
    total_y = sum(d.y.numel() for d in train_set)
    pos_frac = pos / total_y
    print(f"  Train graphs: {len(train_set)}, Test graphs: {len(test_set)}")
    print(f"  Total residues (train): {total_y}")
    print(f"  Fraction of positive labels: {pos_frac:.3f}")
    print(f"  Always-predict-majority accuracy: {max(pos_frac, 1 - pos_frac):.3f}")

    print("\n[1/2] Training MLP baseline (no neighbour info)...")
    mlp_acc = train_and_eval(MLP(20, 32, 2), train_loader, test_loader, device)
    print(f"  Final MLP accuracy: {mlp_acc:.3f}")

    print("\n[2/2] Training GCN (uses neighbours via message passing)...")
    gcn_acc = train_and_eval(GCN(20, 32, 2), train_loader, test_loader, device)
    print(f"  Final GCN accuracy: {gcn_acc:.3f}")

    print(f"\nDelta (GCN - MLP): {gcn_acc - mlp_acc:+.3f}")
    print("If they're roughly tied, the task is too easy or the window is too small.")

    print(
        """
Things to experiment with:
- Add a third GCN layer — output now reflects 3-hop neighbourhoods.
- Drop GCN to 1 layer — should drop to MLP-level performance.
- Make the task harder: hydro_threshold=0.8, smaller window.
- Replace GCNConv with SAGEConv (GraphSAGE) or GINConv (GIN) — different
  aggregation schemes, similar API.
- Use ATTENTION via GATConv: lets the model learn which neighbours matter.
"""
    )


if __name__ == "__main__":
    main()
