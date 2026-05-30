"""
GNN Lesson 3: Graph Classification with GAT
============================================
What you'll learn
-----------------
- Graph-level prediction: ONE label per graph (per protein).
- Graph Attention Networks (GAT, Veličković et al. 2018): edges have LEARNED
  weights — neighbours contribute differently based on how relevant the model
  thinks they are. This is "attention", same idea as in transformers.
- Pooling: how to go from per-node features to one per-graph vector
  (global mean-pool / max-pool / sum-pool).

Compared to lesson 2 (node classification)
------------------------------------------
- Node classification: one prediction per node.
- Graph classification: one prediction per graph. Need a POOLING step
  to collapse all node features into a single graph-level vector.

Task
----
Solubility prediction (binary) on a small subset of DeepSol.
Each protein -> one graph. Each residue -> one node.
Edge structure: sequence-window graph.
Node features: one-hot amino acid identity (in lesson 4 we'll upgrade these
to ESM-2 embeddings).
"""

import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = "zhanglab/DeepSol"
N_TRAIN = 500
N_TEST = 100
WINDOW = 3                # sequence-graph window
MAX_LEN = 400             # truncate very long proteins for speed
HIDDEN = 64
HEADS = 4                 # GAT attention heads
EPOCHS = 15
BATCH_SIZE = 8
LR = 5e-4
OUTPUT_DIR = "./results"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}


def sequence_to_graph(sequence, label, window=WINDOW, max_len=MAX_LEN):
    """Convert a protein sequence + label to a PyG Data object."""
    sequence = sequence[:max_len]  # truncate
    n = len(sequence)

    x = torch.zeros(n, len(AMINO_ACIDS))
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            x[i, AA_TO_IDX[aa]] = 1.0

    edges = []
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                edges.append((i, i + d))
                edges.append((i + d, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GATGraphClassifier(torch.nn.Module):
    """Two-layer GAT + global mean pool + linear classifier.

    Why GAT?
    --------
    A GCN aggregates neighbours with FIXED weights (essentially a normalised
    sum). GAT learns ATTENTION coefficients per edge:
        alpha_ij = softmax_j( LeakyReLU(a^T [W h_i || W h_j]) )
        h_i'     = sigma( sum_j alpha_ij * W h_j )
    With multiple HEADS, the model learns different "attention patterns" in
    parallel and concatenates them — same trick as multi-head attention in
    transformers.

    The pooling step
    ----------------
    `global_mean_pool(h, batch)` averages all node features within each
    graph (as identified by the `batch` tensor) into a single per-graph vector.
    Alternatives: global_max_pool, global_add_pool, or a "set transformer".
    """

    def __init__(self, in_channels, hidden, out_channels, heads=4):
        super().__init__()
        # GAT layer 1: project in_channels -> hidden, with `heads` parallel
        # attention heads concatenated (output dim = hidden * heads).
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=0.2)
        # GAT layer 2: combine the heads down to `hidden` (averaging via concat=False).
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.2)
        # Final classifier on the pooled graph vector.
        self.classifier = torch.nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        # Pool: (num_nodes_total, hidden) -> (num_graphs_in_batch, hidden)
        g = global_mean_pool(h, batch)
        return self.classifier(g)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    return correct / total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    train_raw = raw["train"].select(range(N_TRAIN))
    test_raw = raw["test"].select(range(N_TEST))

    print("Converting sequences to graphs...")
    train_set = [sequence_to_graph(d["sequence"], d["label"]) for d in train_raw]
    test_set = [sequence_to_graph(d["sequence"], d["label"]) for d in test_raw]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # Class balance check
    train_pos = sum(int(d.y.item()) for d in train_set) / len(train_set)
    print(f"  train: {len(train_set)} graphs, pos_frac={train_pos:.3f}")
    print(f"  test:  {len(test_set)} graphs")
    print(f"  Always-predict-majority test accuracy: {max(train_pos, 1 - train_pos):.3f}")

    model = GATGraphClassifier(
        in_channels=len(AMINO_ACIDS), hidden=HIDDEN, out_channels=2, heads=HEADS
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    print("\nTraining...")
    for ep in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, opt, device)
        acc = evaluate(model, test_loader, device)
        print(f"  epoch {ep + 1:3d}  loss={loss:.3f}  test_acc={acc:.3f}")

    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal test accuracy: {final_acc:.3f}")

    print(
        """
Things to experiment with:
- Increase WINDOW to 5 or 7 — wider edges = larger receptive field.
- Replace GAT with GCNConv or SAGEConv — see how attention compares.
- Replace global_mean_pool with global_max_pool or global_add_pool.
- Add a third GAT layer (might overfit on this dataset size — try with dropout).
- For fairer evaluation, add early stopping on a validation split.
- Lesson 4 keeps everything else the same and swaps node features
  to ESM-2 embeddings — the typical accuracy bump is large.
"""
    )


if __name__ == "__main__":
    main()
