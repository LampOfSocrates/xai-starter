"""
GNN Lesson 11: Heterogeneous and Interaction Graphs
====================================================
What you'll learn
-----------------
- Heterogeneous graphs: graphs with MULTIPLE node and edge types, used to
  model real biological networks (protein-drug, PPI, protein-ligand complexes).
- HeteroData in PyG: how to store per-type node features and typed edge indices
  in a single graph object.
- HeteroConv: running different message-passing operators per relation type,
  then aggregating into a shared embedding space.
- Link prediction: framing "does this protein interact with this drug?" as a
  binary edge-existence task with negative sampling.
- Scoring a node pair via dot-product of learned embeddings after a
  heterogeneous GNN encoder.

Why heterogeneous graphs in biology?
-------------------------------------
A protein-drug interaction (DTI) network has two fundamentally different
entity types: proteins (described by sequence/structure features) and drugs
(described by chemical fingerprints). Treating them as the same kind of node
would discard that distinction. HeteroConv lets the model learn a SEPARATE
message-passing function for each relation type — the "protein -> drug"
messages can use completely different weights from "drug -> protein" messages.

Task and synthetic dataset
--------------------------
We construct a synthetic bipartite protein-drug graph. A LOW-RANK LATENT
FACTOR MODEL defines the ground-truth interactions: each protein and drug gets
a hidden embedding; a pair interacts when their dot-product exceeds a
threshold. This makes the task learnable from the features that are derived
from those same latent factors (with noise). In real work you would replace
the synthetic features with ESM-2 embeddings (proteins) and Morgan fingerprints
(drugs) and labels from BindingDB / DAVIS.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_PROTEINS = 120          # number of protein nodes
N_DRUGS = 80              # number of drug nodes
PROTEIN_FEAT_DIM = 64     # simulates a pooled composition / embedding vector
DRUG_FEAT_DIM = 128       # simulates a binary ECFP4-like fingerprint
LATENT_DIM = 16           # rank of the hidden interaction matrix
INTERACTION_THRESHOLD = 0.0   # dot-product cutoff for a positive pair
NOISE_STD = 0.3           # noise added to features (controls task hardness)
HIDDEN = 64               # GNN hidden dimension
EPOCHS = 80
LR = 1e-3
NEG_RATIO = 2             # negative samples per positive edge in training
TRAIN_FRAC = 0.7          # fraction of all pairs used for training
OUTPUT_DIR = "./results"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Synthetic dataset construction
# ---------------------------------------------------------------------------

def build_synthetic_dataset():
    """Build a protein-drug HeteroData graph with a learnable interaction rule.

    Ground-truth mechanism
    ----------------------
    Each protein p has a latent vector z_p in R^LATENT_DIM.
    Each drug d has a latent vector z_d in R^LATENT_DIM.
    The pair (p, d) interacts if  z_p . z_d > INTERACTION_THRESHOLD.

    Observed features are z + noise (projected to the declared feature dims),
    so the model CAN recover the signal but must learn to denoise.

    Returns
    -------
    data        : HeteroData ready for training
    all_pairs   : (N_proteins * N_drugs, 2) tensor of all (p, d) indices
    all_labels  : (N_proteins * N_drugs,) binary tensor
    train_mask  : boolean mask over all_pairs for training edges
    test_mask   : boolean mask over all_pairs for evaluation
    """
    # Latent factors (the hidden signal)
    z_protein = torch.randn(N_PROTEINS, LATENT_DIM)
    z_drug = torch.randn(N_DRUGS, LATENT_DIM)

    # Observed features: linear projection of latent factors + noise
    W_p = torch.randn(LATENT_DIM, PROTEIN_FEAT_DIM) / LATENT_DIM ** 0.5
    W_d = torch.randn(LATENT_DIM, DRUG_FEAT_DIM) / LATENT_DIM ** 0.5
    protein_feats = z_protein @ W_p + NOISE_STD * torch.randn(N_PROTEINS, PROTEIN_FEAT_DIM)
    drug_feats = z_drug @ W_d + NOISE_STD * torch.randn(N_DRUGS, DRUG_FEAT_DIM)

    # Ground-truth interaction matrix (N_PROTEINS x N_DRUGS)
    scores = z_protein @ z_drug.t()          # (N_PROTEINS, N_DRUGS)
    labels_matrix = (scores > INTERACTION_THRESHOLD).float()

    # Enumerate ALL pairs for evaluation
    pi, di = torch.meshgrid(
        torch.arange(N_PROTEINS),
        torch.arange(N_DRUGS),
        indexing="ij",
    )
    all_pairs = torch.stack([pi.reshape(-1), di.reshape(-1)], dim=1)  # (N*M, 2)
    all_labels = labels_matrix.reshape(-1)                              # (N*M,)

    # Train/test split (random over all pairs)
    n_total = all_pairs.shape[0]
    perm = torch.randperm(n_total)
    n_train = int(n_total * TRAIN_FRAC)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Build positive training edges (the graph structure seen by the GNN)
    # We expose only POSITIVE edges during message passing so the GNN can
    # propagate interaction signals; negative pairs are labelled at scoring time.
    train_pos_mask = train_mask & (all_labels == 1)
    pos_pairs = all_pairs[train_pos_mask]         # (n_pos, 2)
    p_src = pos_pairs[:, 0]
    d_dst = pos_pairs[:, 1]

    # HeteroData object
    data = HeteroData()
    data["protein"].x = protein_feats            # (N_PROTEINS, PROTEIN_FEAT_DIM)
    data["drug"].x = drug_feats                  # (N_DRUGS, DRUG_FEAT_DIM)
    # Forward relation: protein -> drug
    data["protein", "interacts", "drug"].edge_index = torch.stack([p_src, d_dst], dim=0)
    # Reverse relation: drug -> protein (so drugs can aggregate protein signals)
    data["drug", "rev_interacts", "protein"].edge_index = torch.stack([d_dst, p_src], dim=0)

    return data, all_pairs, all_labels, train_mask, test_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeteroInteractionGNN(nn.Module):
    """Two-layer heterogeneous GNN for protein-drug link prediction.

    Architecture
    ------------
    Layer 1:  HeteroConv with two SAGEConv operators — one per relation type.
              Each node type aggregates messages from its neighbours.
    Layer 2:  Another HeteroConv.
    Scoring:  The learned embeddings h_protein[p] and h_drug[d] are combined
              with a dot-product score (equivalent to a rank-1 bilinear form).
              Alternatives: MLP on [h_p || h_d] or full bilinear W.

    Why SAGEConv?
    -------------
    GraphSAGE concatenates the node's own features with the aggregated
    neighbour mean before projecting. It works well on bipartite graphs
    because the source and target nodes can have different feature dimensions
    (SAGEConv automatically handles the (in_channels_src, in_channels_dst)
    tuple form).
    """

    def __init__(self, protein_dim, drug_dim, hidden):
        super().__init__()
        # Layer 1 — separate SAGEConv per relation.
        # SAGEConv accepts (in_channels_src, in_channels_dst) for bipartite.
        self.conv1 = HeteroConv(
            {
                ("protein", "interacts", "drug"): SAGEConv((protein_dim, drug_dim), hidden),
                ("drug", "rev_interacts", "protein"): SAGEConv((drug_dim, protein_dim), hidden),
            },
            aggr="sum",
        )
        # Layer 2 — both sides now have the same `hidden` dimension.
        self.conv2 = HeteroConv(
            {
                ("protein", "interacts", "drug"): SAGEConv(hidden, hidden),
                ("drug", "rev_interacts", "protein"): SAGEConv(hidden, hidden),
            },
            aggr="sum",
        )

    def encode(self, data):
        """Return a dict of per-node-type embeddings."""
        x_dict = {ntype: data[ntype].x for ntype in ["protein", "drug"]}
        edge_index_dict = data.edge_index_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict   # {"protein": (N_PROTEINS, hidden), "drug": (N_DRUGS, hidden)}

    def score_pairs(self, x_dict, pairs):
        """Dot-product score for each (protein_idx, drug_idx) row in `pairs`."""
        h_p = x_dict["protein"][pairs[:, 0]]    # (B, hidden)
        h_d = x_dict["drug"][pairs[:, 1]]       # (B, hidden)
        return (h_p * h_d).sum(dim=-1)           # (B,) — raw logit


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_train_batch(all_pairs, all_labels, train_mask, neg_ratio=NEG_RATIO):
    """Return a balanced (positive + negatives) batch from the training split."""
    pos_idx = (train_mask & (all_labels == 1)).nonzero(as_tuple=True)[0]
    neg_idx = (train_mask & (all_labels == 0)).nonzero(as_tuple=True)[0]
    # Down-sample negatives to neg_ratio * n_pos
    n_neg = min(len(neg_idx), neg_ratio * len(pos_idx))
    neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_neg]]
    idx = torch.cat([pos_idx, neg_idx])
    return all_pairs[idx], all_labels[idx]


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train(model, data, all_pairs, all_labels, train_mask, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    model.train()
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        x_dict = model.encode(data)
        pairs_b, labels_b = sample_train_batch(all_pairs, all_labels, train_mask)
        pairs_b = pairs_b.to(device)
        labels_b = labels_b.to(device)
        logits = model.score_pairs(x_dict, pairs_b)
        loss = F.binary_cross_entropy_with_logits(logits, labels_b)
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  loss={loss.item():.4f}")


@torch.no_grad()
def evaluate(model, data, all_pairs, all_labels, test_mask):
    model.eval()
    x_dict = model.encode(data)
    pairs_t = all_pairs[test_mask]
    labels_t = all_labels[test_mask]
    logits = model.score_pairs(x_dict, pairs_t)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y = labels_t.cpu().numpy()
    auc = roc_auc_score(y, probs)
    acc = (preds == y).mean()
    return auc, acc


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- 1. Build synthetic bipartite graph ----------------------------
    print("\nBuilding synthetic protein-drug interaction dataset...")
    data, all_pairs, all_labels, train_mask, test_mask = build_synthetic_dataset()
    data = data.to(device)
    all_pairs = all_pairs.to(device)
    all_labels = all_labels.float().to(device)

    n_pos = int(all_labels.sum().item())
    n_total = len(all_labels)
    pos_rate = n_pos / n_total
    print(f"  Proteins : {N_PROTEINS}")
    print(f"  Drugs    : {N_DRUGS}")
    print(f"  All pairs: {n_total}  (pos={n_pos}, pos_rate={pos_rate:.3f})")
    print(f"  Train pairs: {int(train_mask.sum())}  |  Test pairs: {int(test_mask.sum())}")
    n_graph_edges = data["protein", "interacts", "drug"].edge_index.shape[1]
    print(f"  Graph edges (train pos only): {n_graph_edges}")

    # ---- 2. Build model ------------------------------------------------
    model = HeteroInteractionGNN(
        protein_dim=PROTEIN_FEAT_DIM,
        drug_dim=DRUG_FEAT_DIM,
        hidden=HIDDEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ---- 3. Train ------------------------------------------------------
    print("\nTraining heterogeneous GNN...")
    train(model, data, all_pairs, all_labels, train_mask, device)

    # ---- 4. Evaluate ---------------------------------------------------
    auc, acc = evaluate(model, data, all_pairs, all_labels, test_mask)
    majority_acc = max(pos_rate, 1.0 - pos_rate)
    print(f"\n--- Test results ---")
    print(f"  AUC      : {auc:.4f}  (random baseline = 0.50)")
    print(f"  Accuracy : {acc:.4f}  (majority baseline = {majority_acc:.3f})")

    print(
        """
Things to experiment with:
- Add a third node type ("disease") with ("protein","associated","disease") edges
  and chain the relations into meta-paths (protein -> drug -> disease).
- Replace synthetic features with real data: ESM-2 per-protein embeddings +
  RDKit Morgan fingerprints for drugs; labels from BindingDB or the DAVIS dataset.
- Swap SAGEConv for GATConv inside HeteroConv to get relation-aware attention weights.
- Try a cold-start split: reserve proteins (or drugs) not seen at train time;
  measure how badly AUC drops and whether richer features help.
- Replace dot-product scoring with a small MLP([h_p || h_d]) for a more
  expressive interaction function — compare AUC.
- Use torch_geometric.transforms.ToUndirected() and compare against the
  explicit reverse-relation construction used here.
"""
    )


if __name__ == "__main__":
    main()
