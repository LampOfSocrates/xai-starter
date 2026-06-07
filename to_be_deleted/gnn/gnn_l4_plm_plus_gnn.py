"""
GNN Lesson 4: pLM Embeddings as GNN Node Features
==================================================
What you'll learn
-----------------
- The "pLM + GNN" pattern: ESM-2 turns each residue into a high-dimensional
  vector that already encodes evolutionary/biophysical information; the GNN
  then reasons about how those residues interact through a graph.
- Why this combination is so powerful: pLMs encode per-residue context;
  GNNs encode pairwise interactions. They're complementary.

Compared to gnn_l3
------------------
Same task (DeepSol solubility, graph classification), same graph structure
(sequence-window), same model architecture. The ONLY thing that changes:
the node features.

   gnn_l3:  one-hot AA  (20-dim)        -> typically modest accuracy
   gnn_l4:  ESM-2 8M    (320-dim)       -> usually noticeably better

This is one of the most reliably-effective patterns in protein ML.

Compute note
------------
Extracting ESM-2 embeddings is the expensive step here. We do it ONCE per
sequence and cache them in the Data objects. The GNN training loop afterwards
is fast and CPU-friendly.
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLM_NAME = "facebook/esm2_t6_8M_UR50D"   # produces 320-dim embeddings
DATASET_NAME = "zhanglab/DeepSol"
N_TRAIN = 200       # smaller because ESM-2 forward is the bottleneck
N_TEST = 50
WINDOW = 3
MAX_LEN = 400
HIDDEN = 128
HEADS = 4
EPOCHS = 15
BATCH_SIZE = 4
LR = 5e-4
PLM_BATCH_SIZE = 4    # how many sequences to embed at once


# ---------------------------------------------------------------------------
# pLM embedding extraction
# ---------------------------------------------------------------------------

def embed_sequences(sequences, tokenizer, plm, device, batch_size=PLM_BATCH_SIZE):
    """Run ESM-2 over each sequence and return PER-RESIDUE embeddings.

    Note: we keep ALL residue embeddings (not pooled), because the GNN
    will use them as node features.

    Returns: list of tensors, one per sequence, each of shape (L, hidden_dim).
    """
    all_embs = []
    plm.eval()
    for i in range(0, len(sequences), batch_size):
        batch = [s[:MAX_LEN] for s in sequences[i : i + batch_size]]
        inputs = tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_LEN + 2,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = plm(**inputs).last_hidden_state    # (B, L+2, D)

        # The tokenizer adds <cls> at index 0 and <eos> at the end. We want
        # one embedding per amino acid, so we slice [1, 1+len(seq)] for each
        # sequence in the batch.
        for j, seq in enumerate(batch):
            emb = hidden[j, 1 : 1 + len(seq)].cpu()
            all_embs.append(emb)

        if (i // batch_size) % 10 == 0:
            print(f"  embedded {min(i + batch_size, len(sequences))}/{len(sequences)}")
    return all_embs


def make_graph(sequence, embedding, label, window=WINDOW):
    """Build a PyG Data object using ESM-2 embeddings as node features."""
    n = len(sequence)
    edges = []
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                edges.append((i, i + d))
                edges.append((i + d, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(
        x=embedding,                   # (n, plm_hidden_dim)
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Model — same shape as gnn_l3, but `in_channels` = pLM hidden dim
# ---------------------------------------------------------------------------

class GATGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.2)
        self.classifier = torch.nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        return self.classifier(global_mean_pool(h, batch))


# ---------------------------------------------------------------------------
# Train / eval (same as gnn_l3)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        opt.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        correct += (logits.argmax(dim=-1) == batch.y).sum().item()
        total += batch.num_graphs
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 1. Load pLM (frozen — we never update its weights) ------------
    print(f"Loading pLM: {PLM_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(PLM_NAME)
    plm = AutoModel.from_pretrained(PLM_NAME).to(device)
    plm_dim = plm.config.hidden_size
    print(f"  pLM hidden dim: {plm_dim}")

    # ---- 2. Load dataset ------------------------------------------------
    print(f"\nLoading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    train_raw = raw["train"].select(range(N_TRAIN))
    test_raw = raw["test"].select(range(N_TEST))

    train_seqs = [d["sequence"][:MAX_LEN] for d in train_raw]
    train_labels = [d["label"] for d in train_raw]
    test_seqs = [d["sequence"][:MAX_LEN] for d in test_raw]
    test_labels = [d["label"] for d in test_raw]

    # ---- 3. Extract embeddings (the expensive step) -------------------
    print("\nExtracting train embeddings...")
    train_embs = embed_sequences(train_seqs, tokenizer, plm, device)
    print("Extracting test embeddings...")
    test_embs = embed_sequences(test_seqs, tokenizer, plm, device)

    # Free the pLM — we don't need it for GNN training.
    del plm
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 4. Build graphs -----------------------------------------------
    print("\nBuilding graphs (with pLM embeddings as node features)...")
    train_set = [make_graph(s, e, y) for s, e, y in zip(train_seqs, train_embs, train_labels)]
    test_set = [make_graph(s, e, y) for s, e, y in zip(test_seqs, test_embs, test_labels)]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # ---- 5. Train the GNN ----------------------------------------------
    print("\nTraining GAT on top of pLM embeddings...")
    model = GATGraphClassifier(
        in_channels=plm_dim, hidden=HIDDEN, out_channels=2, heads=HEADS
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    for ep in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, opt, device)
        acc = evaluate(model, test_loader, device)
        print(f"  epoch {ep + 1:3d}  loss={loss:.3f}  test_acc={acc:.3f}")

    print(f"\nFinal test accuracy: {evaluate(model, test_loader, device):.3f}")
    print(
        "\nCompare with gnn_l3 (same architecture, but one-hot AA features). "
        "The accuracy lift here is the value of the pLM."
    )

    print(
        """
Things to experiment with:
- Use a bigger pLM: PLM_NAME = "facebook/esm2_t12_35M_UR50D"
- FINE-TUNE the pLM end-to-end with the GNN: remove `with torch.no_grad()`
  and put the pLM on the optimiser. Massive lift, much more compute.
- Replace the sequence-window graph with a learned k-NN graph on pLM
  embeddings — useful when no real 3D structure is available.
- Use this on a per-residue task instead of graph-level: see gnn_l2.
"""
    )


if __name__ == "__main__":
    main()
