"""
GNN Lesson 9: k-NN and Learned Graphs
======================================
What you'll learn
-----------------
- How to build a graph when you have NO 3D structure, using only ESM-2
  per-residue embeddings to define neighbourhood.
- The difference between sequence-window, embedding-kNN, and random graphs
  built over the same node features.
- How to benchmark three graph constructions on the same task so you can
  see which edges actually help the GNN.
- How to visualise which residue pairs a k-NN graph connects versus the
  simpler sequence-window baseline.

When do you need this?
----------------------
Most proteins in the wild have no experimentally determined structure. You
can still build a useful graph by asking: "which residues are SIMILAR in
embedding space?" Residues with similar ESM-2 embeddings often play similar
structural or functional roles, even if they are far apart in sequence. A
k-NN graph in that embedding space recovers some of the long-range contacts
a contact graph would give you — without ever needing 3D coordinates.

Experiment design
-----------------
We hold EVERYTHING constant except the edge set:
  Same dataset  : DeepSol solubility classification (graph-level, binary)
  Same features : ESM-2 8M per-residue embeddings, frozen, cached once
  Same model    : two-layer GAT + global_mean_pool + linear head

Three edge sets are tested per protein:
  (a) sequence-window  -- edges to the k nearest neighbours IN SEQUENCE
  (b) embedding-kNN    -- edges to the k nearest neighbours IN ESM-2 SPACE
  (c) random           -- random edges with the same count as embedding-kNN
                          (control: are ANY edges better than nothing?)

Expected ordering: embedding-kNN >= sequence-window > random.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, knn_graph
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLM_NAME = "facebook/esm2_t6_8M_UR50D"   # 320-dim per-residue embeddings
DATASET_NAME = "zhanglab/DeepSol"
N_TRAIN = 150        # ESM-2 forward is the bottleneck; keep this small
N_TEST = 50
MAX_LEN = 300        # truncate longer sequences to save memory
WINDOW = 5           # sequence-window half-width (matches kNN k below)
KNN_K = 10           # k for knn_graph in embedding space
HIDDEN = 128
HEADS = 4
EPOCHS = 20
BATCH_SIZE = 4
LR = 5e-4
PLM_BATCH_SIZE = 4   # sequences to embed per ESM-2 forward pass

VIZ_PROTEIN_IDX = 0  # index into test set for the residue-pair figure
VIZ_MAX_RESIDUES = 60  # truncate the visualised protein for readability


# ---------------------------------------------------------------------------
# pLM embedding extraction  (same as gnn_l4 — cached in-memory)
# ---------------------------------------------------------------------------

def embed_sequences(sequences, tokenizer, plm, device):
    """Run ESM-2 over each sequence; return per-residue embeddings.

    Returns a list of (L, D) float tensors, one per sequence.
    We slice away the <cls>/<eos> special tokens so length == len(sequence).
    """
    all_embs = []
    plm.eval()
    for i in range(0, len(sequences), PLM_BATCH_SIZE):
        batch = sequences[i : i + PLM_BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN + 2,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = plm(**inputs).last_hidden_state  # (B, L+2, D)
        for j, seq in enumerate(batch):
            emb = hidden[j, 1 : 1 + len(seq)].cpu()  # strip special tokens
            all_embs.append(emb)
        if (i // PLM_BATCH_SIZE) % 10 == 0:
            done = min(i + PLM_BATCH_SIZE, len(sequences))
            print(f"  embedded {done}/{len(sequences)}")
    return all_embs


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def make_sequence_graph(embedding, label, window=WINDOW):
    """Edges connect residues within `window` positions in the sequence."""
    n = len(embedding)
    edges = []
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                edges.append((i, i + d))
                edges.append((i + d, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(
        x=embedding,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )


def make_knn_graph(embedding, label, k=KNN_K):
    """Edges connect each residue to its k nearest neighbours in ESM-2 space.

    knn_graph() from torch_geometric.nn returns directed edges (each node
    gets exactly k incoming edges). We leave them directed; GAT handles it.
    """
    # knn_graph expects (N, F) float tensor and returns edge_index (2, N*k).
    edge_index = knn_graph(embedding, k=k, loop=False)
    return Data(
        x=embedding,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )


def make_random_graph(embedding, label, k=KNN_K):
    """Random graph with the same number of edges as the kNN graph (control).

    Each node gets exactly k random target edges, matching the degree of the
    kNN graph so the only difference is WHICH edges exist, not how many.
    """
    n = len(embedding)
    src = torch.arange(n).repeat_interleave(k)
    # Sample k random targets per node, excluding self-loops.
    tgt_list = []
    for i in range(n):
        pool = list(range(n))
        pool.remove(i)
        chosen = np.random.choice(pool, size=min(k, n - 1), replace=False)
        tgt_list.append(torch.tensor(chosen, dtype=torch.long))
    tgt = torch.cat(tgt_list)
    edge_index = torch.stack([src, tgt], dim=0)
    return Data(
        x=embedding,
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Model  (identical to gnn_l4)
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
# Train / eval helpers
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
        preds = model(batch.x, batch.edge_index, batch.batch).argmax(dim=-1)
        correct += (preds == batch.y).sum().item()
        total += batch.num_graphs
    return correct / total


def run_experiment(name, train_graphs, test_graphs, plm_dim, device):
    """Train a fresh GATGraphClassifier on `train_graphs`, return test acc."""
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    model = GATGraphClassifier(
        in_channels=plm_dim, hidden=HIDDEN, out_channels=2, heads=HEADS
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"\n  [{name}] training for {EPOCHS} epochs...")
    for ep in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, opt, device)
        if (ep + 1) % 5 == 0:
            acc = evaluate(model, test_loader, device)
            print(f"    epoch {ep + 1:3d}  loss={loss:.3f}  test_acc={acc:.3f}")

    final_acc = evaluate(model, test_loader, device)
    return final_acc


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_edge_comparison(embedding, savepath, max_res=VIZ_MAX_RESIDUES):
    """Plot residue-pair connectivity for sequence-window vs kNN on one protein.

    Each subplot shows a matrix where a filled cell (i, j) means an edge
    exists between residue i and residue j. The sequence-window matrix has a
    banded structure; the kNN matrix reveals which long-range pairs the
    model considers similar in embedding space.
    """
    n = min(len(embedding), max_res)
    emb = embedding[:n]

    # Sequence-window adjacency
    seq_adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for d in range(1, WINDOW + 1):
            if i + d < n:
                seq_adj[i, i + d] = 1
                seq_adj[i + d, i] = 1

    # kNN adjacency
    ei = knn_graph(emb, k=KNN_K, loop=False)
    knn_adj = np.zeros((n, n), dtype=np.float32)
    for s, t in ei.t().tolist():
        if s < n and t < n:
            knn_adj[s, t] = 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in zip(
        axes,
        [seq_adj, knn_adj],
        [f"Sequence-window (w={WINDOW})", f"Embedding k-NN (k={KNN_K})"],
    ):
        ax.imshow(mat, cmap="Blues", origin="upper", aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Residue index")
        ax.set_ylabel("Residue index")

    plt.suptitle(
        f"Edge patterns for one protein (first {n} residues)", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved edge-comparison figure to {savepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("./results", exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 1. Load pLM -------------------------------------------------------
    print(f"\nLoading pLM: {PLM_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(PLM_NAME)
    plm = AutoModel.from_pretrained(PLM_NAME).to(device)
    plm_dim = plm.config.hidden_size
    print(f"  hidden dim: {plm_dim}")

    # ---- 2. Load dataset ---------------------------------------------------
    print(f"\nLoading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    train_raw = raw["train"].select(range(N_TRAIN))
    test_raw = raw["test"].select(range(N_TEST))

    train_seqs = [d["sequence"][:MAX_LEN] for d in train_raw]
    train_labels = [d["label"] for d in train_raw]
    test_seqs = [d["sequence"][:MAX_LEN] for d in test_raw]
    test_labels = [d["label"] for d in test_raw]

    # ---- 3. Embed (the expensive step — done once, cached as tensors) ------
    print("\nExtracting train embeddings...")
    train_embs = embed_sequences(train_seqs, tokenizer, plm, device)
    print("Extracting test embeddings...")
    test_embs = embed_sequences(test_seqs, tokenizer, plm, device)

    del plm  # free memory; we don't need the pLM for GNN training
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 4. Build three graph datasets ------------------------------------
    print("\nBuilding three graph sets from the same embeddings...")

    train_seq  = [make_sequence_graph(e, y) for e, y in zip(train_embs, train_labels)]
    test_seq   = [make_sequence_graph(e, y) for e, y in zip(test_embs,  test_labels)]

    train_knn  = [make_knn_graph(e, y) for e, y in zip(train_embs, train_labels)]
    test_knn   = [make_knn_graph(e, y) for e, y in zip(test_embs,  test_labels)]

    train_rand = [make_random_graph(e, y) for e, y in zip(train_embs, train_labels)]
    test_rand  = [make_random_graph(e, y) for e, y in zip(test_embs,  test_labels)]

    avg_edges_knn = np.mean([g.num_edges for g in train_knn])
    avg_edges_seq = np.mean([g.num_edges for g in train_seq])
    print(f"  avg edges per graph — sequence-window: {avg_edges_seq:.0f},  kNN: {avg_edges_knn:.0f}")

    # ---- 5. Visualise edge patterns for one test protein ------------------
    viz_emb = test_embs[VIZ_PROTEIN_IDX]
    visualise_edge_comparison(
        viz_emb,
        savepath="./results/gnn_l9_edge_comparison.png",
    )

    # ---- 6. Train each variant and collect results ------------------------
    print("\nRunning experiments (same GAT, different edges)...")
    acc_seq  = run_experiment("sequence-window", train_seq,  test_seq,  plm_dim, device)
    acc_knn  = run_experiment("embedding-kNN",   train_knn,  test_knn,  plm_dim, device)
    acc_rand = run_experiment("random (control)", train_rand, test_rand, plm_dim, device)

    # ---- 7. Results table -------------------------------------------------
    print("\n" + "=" * 48)
    print(f"{'Graph type':<22}  {'Test accuracy':>13}")
    print("-" * 48)
    print(f"{'sequence-window':<22}  {acc_seq:>12.3f}")
    print(f"{'embedding-kNN':<22}  {acc_knn:>12.3f}")
    print(f"{'random (control)':<22}  {acc_rand:>12.3f}")
    print("=" * 48)

    best = max(
        [("sequence-window", acc_seq), ("embedding-kNN", acc_knn), ("random", acc_rand)],
        key=lambda t: t[1],
    )
    print(f"\nBest graph: {best[0]}  ({best[1]:.3f})")
    if acc_knn >= acc_seq:
        print("Embedding-kNN matched or beat sequence-window — long-range "
              "embedding similarity carries useful signal.")
    else:
        print("Sequence-window beat embedding-kNN here — worth varying k "
              "or the distance metric (see below).")

    print(
        """
Things to experiment with:
- Vary k (KNN_K): smaller k = sparser graph (try 3, 5, 20). Watch the
  crossover point where sparser graphs start hurting accuracy.
- Use cosine similarity instead of Euclidean: pass `cosine=True` to
  knn_graph() to see whether direction matters more than magnitude.
- Combine edges: take the UNION of sequence-window and kNN edges. Does
  more connectivity help, or does the noise cancel the signal?
- Build kNN over AlphaFold2 predicted Cα coordinates instead of embeddings
  (requires loading a .pdb/.cif), then compare to the embedding-kNN here.
- Try a learned graph generator (e.g. Gumbel-softmax top-k) that is
  trained end-to-end with the GNN so the graph itself is optimised.
- Compare to the true contact graph from Lesson 6 (if you have structures)
  — how closely does the embedding-kNN approximate the contact map?
"""
    )


if __name__ == "__main__":
    main()
