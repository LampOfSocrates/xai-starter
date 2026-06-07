"""
Capstone 2: Benchmarking Models the Honest Way
===============================================
What you'll learn
-----------------
- Why a single accuracy on a test set tells you almost nothing on its own.
- How to set up TRAIN / VALIDATION / TEST splits and why the test set must
  only be touched once, at the very end.
- How to compare multiple models fairly: same data, same seeds, same metric.
- Early stopping on the validation set instead of a fixed epoch count.
- Reporting mean ± std over seeds to show variance, not just a lucky number.

The evaluation problem
----------------------
Earlier lessons each reported one accuracy with no validation split, no
baseline, and no variance estimate. This capstone fixes all three.

Data hygiene note
-----------------
The gold standard is SEQUENCE-IDENTITY-BASED splitting via MMseqs2 or
CD-HIT: cluster at 30% identity and put whole clusters in train or test.
Random splitting (used here for speed) overstates performance when
near-duplicates land on both sides. See the FLIP benchmark paper.
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLM_NAME     = "facebook/esm2_t6_8M_UR50D"   # 320-dim, CPU-friendly
DATASET_NAME = "zhanglab/DeepSol"

N_TRAIN   = 300   # from the dataset's train split
N_VAL     = 100   # carved out of N_TRAIN; no model ever sees this during tuning
N_TEST    = 100   # from the dataset's test split — touched ONCE, at the end
MAX_LEN   = 300   # truncate long proteins for speed
WINDOW    = 3     # sequence-window graph connectivity

HIDDEN    = 64
HEADS     = 4
LR        = 5e-4
BATCH_SIZE = 8
PLM_BATCH  = 4

MAX_EPOCHS = 30
PATIENCE   = 5    # early stopping: halt if val acc stalls this many epochs

SEEDS      = [0, 1, 2]   # 3 seeds gives a rough variance estimate quickly
OUTPUT_DIR = "./results"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX   = {a: i for i, a in enumerate(AMINO_ACIDS)}


# ---------------------------------------------------------------------------
# ESM-2 embedding helpers
# ---------------------------------------------------------------------------

def embed_mean_pool(seqs, tok, plm, device):
    """(N,) protein sequences -> (N, D) numpy array of mean-pooled embeddings."""
    out = []
    plm.eval()
    for i in range(0, len(seqs), PLM_BATCH):
        batch = [s[:MAX_LEN] for s in seqs[i : i + PLM_BATCH]]
        inp = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN + 2)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            h = plm(**inp).last_hidden_state           # (B, L, D)
        mask = inp["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.vstack(out)


def embed_per_residue(seqs, tok, plm, device):
    """(N,) sequences -> list of N tensors, each (L_i, D)."""
    out = []
    plm.eval()
    for i in range(0, len(seqs), PLM_BATCH):
        batch = [s[:MAX_LEN] for s in seqs[i : i + PLM_BATCH]]
        inp = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN + 2)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            h = plm(**inp).last_hidden_state           # (B, L+2, D)
        for j, seq in enumerate(batch):
            out.append(h[j, 1 : 1 + len(seq)].cpu())  # strip <cls>/<eos>
    return out


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def seq_to_graph_onehot(seq, label):
    seq = seq[:MAX_LEN]
    n = len(seq)
    x = torch.zeros(n, len(AMINO_ACIDS))
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            x[i, AA_TO_IDX[aa]] = 1.0
    edges = [(i, i+d) for i in range(n) for d in range(1, WINDOW+1)
             if i+d < n]
    edges += [(b, a) for a, b in edges]
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=ei, y=torch.tensor([label], dtype=torch.long))


def seq_to_graph_plm(seq, emb, label):
    seq = seq[:MAX_LEN]
    n = len(seq)
    edges = [(i, i+d) for i in range(n) for d in range(1, WINDOW+1)
             if i+d < n]
    edges += [(b, a) for a, b in edges]
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=emb, edge_index=ei, y=torch.tensor([label], dtype=torch.long))


# ---------------------------------------------------------------------------
# GAT graph classifier (shared by Model 3 and Model 4)
# ---------------------------------------------------------------------------

class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels,        HIDDEN, heads=HEADS, dropout=0.2)
        self.gat2 = GATConv(HIDDEN * HEADS, HIDDEN, heads=1, concat=False, dropout=0.2)
        self.head = torch.nn.Linear(HIDDEN, 2)

    def forward(self, x, edge_index, batch):
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        return self.head(global_mean_pool(h, batch))


def _gnn_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            correct += (model(b.x, b.edge_index, b.batch).argmax(-1) == b.y).sum().item()
            total += b.num_graphs
    return correct / total


def train_gnn(graphs_tr, graphs_va, in_ch, device, seed):
    """Train with early stopping; returns (model_at_best_val, best_val_acc)."""
    torch.manual_seed(seed)
    model = GATClassifier(in_ch).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    tr_ld = DataLoader(graphs_tr, batch_size=BATCH_SIZE, shuffle=True)
    va_ld = DataLoader(graphs_va, batch_size=BATCH_SIZE)

    best_val, best_state, no_imp = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for b in tr_ld:
            b = b.to(device)
            opt.zero_grad()
            F.cross_entropy(model(b.x, b.edge_index, b.batch), b.y).backward()
            opt.step()
        va = _gnn_acc(model, va_ld, device)
        if va > best_val:
            best_val, best_state, no_imp = va, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_val


def _eval_gnn(model, graphs, device):
    loader = DataLoader(graphs, batch_size=BATCH_SIZE)
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            yt += b.y.cpu().tolist()
            yp += model(b.x, b.edge_index, b.batch).argmax(-1).cpu().tolist()
    return np.array(yt), np.array(yp)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Fixed split — done BEFORE any model is created.
    print(f"Loading {DATASET_NAME}...")
    raw = load_dataset(DATASET_NAME)
    pool = raw["train"].select(range(N_TRAIN + N_VAL))
    val_rows   = pool.select(range(N_VAL))
    train_rows = pool.select(range(N_VAL, N_VAL + N_TRAIN))
    test_rows  = raw["test"].select(range(N_TEST))

    tr_seqs, tr_y = train_rows["sequence"], np.array(train_rows["label"])
    va_seqs, va_y = val_rows["sequence"],   np.array(val_rows["label"])
    te_seqs, te_y = test_rows["sequence"],  np.array(test_rows["label"])

    majority_acc = max(te_y.mean(), 1 - te_y.mean())
    print(f"  train={len(tr_seqs)} val={len(va_seqs)} test={len(te_seqs)}")
    print(f"  train pos-frac={tr_y.mean():.3f}  majority test acc={majority_acc:.3f}")

    # 2. Extract ESM-2 embeddings ONCE; reuse for Models 2 and 4.
    print(f"\nLoading {PLM_NAME}...")
    tok = AutoTokenizer.from_pretrained(PLM_NAME)
    plm = AutoModel.from_pretrained(PLM_NAME).to(device)
    plm_dim = plm.config.hidden_size
    print("Embedding (mean-pool for LogReg)...")
    X_tr = embed_mean_pool(tr_seqs, tok, plm, device)
    X_va = embed_mean_pool(va_seqs, tok, plm, device)
    X_te = embed_mean_pool(te_seqs, tok, plm, device)
    print("Embedding (per-residue for GNN)...")
    emb_tr = embed_per_residue(tr_seqs, tok, plm, device)
    emb_va = embed_per_residue(va_seqs, tok, plm, device)
    emb_te = embed_per_residue(te_seqs, tok, plm, device)
    del plm
    if device == "cuda":
        torch.cuda.empty_cache()

    # 3. Build graphs (once; topology is fixed, reused across seeds).
    print("Building graphs...")
    g_tr_oh  = [seq_to_graph_onehot(s, y) for s, y in zip(tr_seqs, tr_y)]
    g_va_oh  = [seq_to_graph_onehot(s, y) for s, y in zip(va_seqs, va_y)]
    g_te_oh  = [seq_to_graph_onehot(s, y) for s, y in zip(te_seqs, te_y)]
    g_tr_plm = [seq_to_graph_plm(s, e, y) for s, e, y in zip(tr_seqs, emb_tr, tr_y)]
    g_va_plm = [seq_to_graph_plm(s, e, y) for s, e, y in zip(va_seqs, emb_va, va_y)]
    g_te_plm = [seq_to_graph_plm(s, e, y) for s, e, y in zip(te_seqs, emb_te, te_y)]

    # 4. Evaluate all models over multiple seeds.
    results = {"1-Majority baseline": [], "2-ESM2+LogReg": [],
               "3-GNN (one-hot)": [],     "4-GNN (ESM-2 feats)": []}
    maj_pred = np.full(len(te_y), int(tr_y.mean() >= 0.5))

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        np.random.seed(seed)

        results["1-Majority baseline"].append(
            (majority_acc, majority_acc, f1_score(te_y, maj_pred, zero_division=0))
        )

        print("  Model 2: ESM-2 + LogReg")
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_tr, tr_y)
        results["2-ESM2+LogReg"].append((
            accuracy_score(va_y, clf.predict(X_va)),
            accuracy_score(te_y, clf.predict(X_te)),
            f1_score(te_y, clf.predict(X_te), zero_division=0),
        ))

        print("  Model 3: GNN one-hot + early stop")
        m3, va3 = train_gnn(g_tr_oh, g_va_oh, len(AMINO_ACIDS), device, seed)
        yt3, yp3 = _eval_gnn(m3, g_te_oh, device)
        results["3-GNN (one-hot)"].append(
            (va3, accuracy_score(yt3, yp3), f1_score(yt3, yp3, zero_division=0))
        )

        print("  Model 4: GNN ESM-2 feats + early stop")
        m4, va4 = train_gnn(g_tr_plm, g_va_plm, plm_dim, device, seed)
        yt4, yp4 = _eval_gnn(m4, g_te_plm, device)
        results["4-GNN (ESM-2 feats)"].append(
            (va4, accuracy_score(yt4, yp4), f1_score(yt4, yp4, zero_division=0))
        )

    # 5. Print aligned results table.
    print("\n" + "=" * 66)
    print(f"{'Model':<24}  {'Val acc':>12}  {'Test acc':>12}  {'Test F1':>10}")
    print("-" * 66)
    bar_names, bar_means, bar_stds = [], [], []
    for name, runs in results.items():
        vm, vs = np.mean([r[0] for r in runs]), np.std([r[0] for r in runs])
        tm, ts = np.mean([r[1] for r in runs]), np.std([r[1] for r in runs])
        fm, fs = np.mean([r[2] for r in runs]), np.std([r[2] for r in runs])
        print(f"{name:<24}  {vm:.3f}±{vs:.3f}    {tm:.3f}±{ts:.3f}  {fm:.3f}±{fs:.3f}")
        bar_names.append(name.split("-", 1)[1])
        bar_means.append(tm); bar_stds.append(ts)
    print("=" * 66)

    # 6. Bar chart with error bars.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(bar_names)), bar_means, yerr=bar_stds, capsize=5,
           color="steelblue", alpha=0.8)
    ax.axhline(majority_acc, color="red", linestyle="--", linewidth=1,
               label=f"Majority baseline ({majority_acc:.2f})")
    ax.set_xticks(range(len(bar_names)))
    ax.set_xticklabels(bar_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1)
    ax.set_title("DeepSol solubility — model comparison (mean ± std, 3 seeds)")
    ax.legend(fontsize=8); fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "capstone_l2_benchmark.png")
    fig.savefig(out, dpi=120)
    print(f"\nBar chart saved to {out}")

    print(
        "\nKey takeaways\n"
        "  Always beat a baseline. Majority-class costs nothing to compute.\n"
        "  Validation absorbs all tuning; test is touched exactly once.\n"
        "  Small datasets are noisy — 3 seeds already reveals real variance.\n"
        "  Random splits overstate accuracy; identity-based splits are honest.\n"
    )

    print(
        """
Things to experiment with:
- Identity-based splits with MMseqs2 or CD-HIT at 30% sequence identity
  to see how much random splitting inflates the numbers.
- k-fold cross-validation when labelled data is very scarce.
- Add ROC-AUC and Matthews Correlation Coefficient (MCC) — both are more
  informative than accuracy on imbalanced protein datasets.
- Paired permutation test across seeds to check statistical significance
  of the gap between Model 3 and Model 4.
- Benchmark against TAPE / PEER / FLIP which provide curated splits and
  public leaderboards for fair comparison with published methods.
- Hyperparameter search (HIDDEN, LR, WINDOW) on the validation set only,
  then report the chosen config alongside the final test numbers.
"""
    )


if __name__ == "__main__":
    main()
