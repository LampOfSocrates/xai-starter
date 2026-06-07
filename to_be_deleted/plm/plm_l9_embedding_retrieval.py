"""
Lesson 9: Embedding Geometry and Retrieval
==========================================
What you'll learn
-----------------
- Visualise WHAT the pLM embedding space encodes using PCA and t-SNE.
- See whether protein families form geometrically distinct clusters.
- Use cosine-similarity nearest-neighbour search as embedding-based retrieval
  (the same principle as BLAST-free homology search at scale).
- Measure retrieval quality: top-1 accuracy and precision@k.
- Compare against a random-retrieval baseline to ground the numbers.

How we build the dataset
------------------------
We synthesise four protein "families" by taking four real seed sequences
(short, diverse) from DeepSol and generating variants at increasing mutation
rates. Within a family, sequences are similar; across families they differ.
This gives clear ground-truth cluster labels and avoids any external download
beyond DeepSol. We draw ~60 sequences per family (~240 total), well within
CPU budget.

Why cosine similarity for retrieval
------------------------------------
ESM-2 embeddings are high-dimensional (~320 dims). Cosine similarity measures
the ANGLE between vectors, ignoring magnitude — a natural fit for
learned representations where scale carries little meaning. This is the same
metric used by modern protein search tools (e.g. FAISS-based indexes in
large-scale proteomics pipelines). Euclidean distance tends to work similarly
in practice for normalised embeddings, but cosine is the community convention.
"""

import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration. Edit these to experiment.
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # 8M params, 320-dim; fast on CPU

DATASET_NAME = "zhanglab/DeepSol"
SEED_POOL_SIZE = 200   # sequences to scan when picking diverse seeds
N_FAMILIES = 4         # number of synthetic protein families
SEQS_PER_FAMILY = 60   # sequences generated per family
MUTATION_RATE = 0.15   # fraction of positions randomly mutated per variant
QUERY_FRACTION = 0.25  # fraction of each family held out as queries
K_NEIGHBORS = 5        # k for precision@k metric
BATCH_SIZE = 8
RESULTS_DIR = "./results"
RANDOM_SEED = 42

# Standard 20 amino acids used when mutating positions.
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def pick_diverse_seeds(sequences, n_seeds, seed_pool_size, rng):
    """Return n_seeds sequences that are roughly diverse by length spread.

    We use a simple heuristic: sort by length and pick evenly spaced indices
    so seeds span the length distribution of the pool. This avoids loading any
    structure data and is sufficient to ensure seeds differ meaningfully.
    """
    pool = sequences[:seed_pool_size]
    pool_sorted = sorted(pool, key=len)
    indices = np.linspace(0, len(pool_sorted) - 1, n_seeds, dtype=int)
    return [pool_sorted[i] for i in indices]


def mutate(sequence, mutation_rate, rng):
    """Return a copy of sequence with ~mutation_rate fraction of positions
    replaced by a uniformly random amino acid (including the original).

    Using randint to pick the new amino acid means there is a 1/20 chance of
    a silent mutation; the effective substitution rate is slightly lower, which
    is fine for our purposes.
    """
    seq = list(sequence)
    for i in range(len(seq)):
        if rng.random() < mutation_rate:
            seq[i] = rng.choice(AMINO_ACIDS)
    return "".join(seq)


def build_dataset(seed_sequences, seqs_per_family, mutation_rate, rng):
    """Build a labelled synthetic dataset of protein variants.

    Returns (sequences, labels) where label is an integer family index.
    Each family consists of seqs_per_family mutants of one seed sequence.
    The seed itself is included as the first member of its family.
    """
    sequences, labels = [], []
    for fam_idx, seed in enumerate(seed_sequences):
        sequences.append(seed)
        labels.append(fam_idx)
        for _ in range(seqs_per_family - 1):
            sequences.append(mutate(seed, mutation_rate, rng))
            labels.append(fam_idx)
    return sequences, np.array(labels)


def train_query_split(sequences, labels, query_fraction, rng):
    """Split into gallery (database) and query sets, stratified by family."""
    gallery_seqs, gallery_labels = [], []
    query_seqs, query_labels = [], []

    unique_labels = np.unique(labels)
    for fam in unique_labels:
        idx = np.where(labels == fam)[0].tolist()
        rng.shuffle(idx)
        n_query = max(1, int(len(idx) * query_fraction))
        for i in idx[:n_query]:
            query_seqs.append(sequences[i])
            query_labels.append(labels[i])
        for i in idx[n_query:]:
            gallery_seqs.append(sequences[i])
            gallery_labels.append(labels[i])

    return (
        gallery_seqs, np.array(gallery_labels),
        query_seqs,  np.array(query_labels),
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embeddings(sequences, model, tokenizer, device, batch_size=BATCH_SIZE):
    """Mean-pool ESM-2 hidden states over non-padding tokens.

    Identical to Lesson 1 — fixed, frozen pLM as a feature extractor.
    """
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state              # (B, L, D)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(pooled.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {min(i + batch_size, len(sequences))}/{len(sequences)}")
    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Geometry: dimensionality reduction plots
# ---------------------------------------------------------------------------

def plot_2d(embeddings_2d, labels, title, path, family_names):
    """Scatter-plot 2D projections coloured by family label."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(family_names)))
    for fam_idx, name in enumerate(family_names):
        mask = labels == fam_idx
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=name,
            s=20,
            alpha=0.7,
            color=colors[fam_idx],
        )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def cosine_normalize(X):
    """L2-normalise rows so that dot product == cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def retrieval_metrics(query_embs, query_labels, gallery_embs, gallery_labels, k):
    """Compute top-1 accuracy and precision@k for cosine-similarity retrieval.

    We use sklearn NearestNeighbors with metric='cosine' on L2-normalised
    embeddings (equivalent to cosine similarity ranking).

    top-1 accuracy: fraction of queries whose nearest gallery neighbour
                    shares the query's family label.
    precision@k:    mean fraction of the k nearest gallery neighbours that
                    share the query's family label.
    """
    q_norm = cosine_normalize(query_embs)
    g_norm = cosine_normalize(gallery_embs)

    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
    nbrs.fit(g_norm)
    distances, indices = nbrs.kneighbors(q_norm)

    top1_correct = 0
    precisions = []
    for q_idx, neighbour_idxs in enumerate(indices):
        neighbour_labels = gallery_labels[neighbour_idxs]
        q_label = query_labels[q_idx]
        top1_correct += int(neighbour_labels[0] == q_label)
        precisions.append(np.mean(neighbour_labels == q_label))

    top1_acc = top1_correct / len(query_labels)
    mean_prec_at_k = float(np.mean(precisions))
    return top1_acc, mean_prec_at_k


def random_retrieval_baseline(query_labels, gallery_labels, k):
    """Expected precision@k if neighbours are drawn uniformly at random.

    For a query of class c, the expected fraction of random neighbours
    sharing class c equals (count(c in gallery)) / len(gallery).
    Top-1 accuracy equals the same fraction for k=1.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    precisions = []
    top1_correct = 0
    for q_label in query_labels:
        sampled = rng.choice(gallery_labels, size=k, replace=False)
        precisions.append(np.mean(sampled == q_label))
        top1_correct += int(sampled[0] == q_label)
    return top1_correct / len(query_labels), float(np.mean(precisions))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng_py = random.Random(RANDOM_SEED)
    rng_np = np.random.default_rng(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load model.
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # 2. Build synthetic multi-family dataset.
    print(f"\nLoading seed pool from: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    raw_seqs = ds["train"].select(range(SEED_POOL_SIZE))["sequence"]

    seeds = pick_diverse_seeds(raw_seqs, N_FAMILIES, SEED_POOL_SIZE, rng_py)
    family_names = [f"Family-{i+1} (seed len {len(s)})" for i, s in enumerate(seeds)]
    print(f"Seed sequence lengths: {[len(s) for s in seeds]}")

    all_seqs, all_labels = build_dataset(seeds, SEQS_PER_FAMILY, MUTATION_RATE, rng_py)
    print(
        f"Synthetic dataset: {len(all_seqs)} sequences, "
        f"{N_FAMILIES} families x {SEQS_PER_FAMILY} each"
    )

    # 3. Split into gallery and query sets.
    gallery_seqs, gallery_labels, query_seqs, query_labels = train_query_split(
        all_seqs, all_labels, QUERY_FRACTION, rng_py
    )
    print(f"Gallery: {len(gallery_seqs)} seqs   Query: {len(query_seqs)} seqs")

    # 4. Embed everything.
    print("\nEmbedding gallery sequences...")
    gallery_embs = get_embeddings(gallery_seqs, model, tokenizer, device)
    print("Embedding query sequences...")
    query_embs = get_embeddings(query_seqs, model, tokenizer, device)

    all_embs   = np.vstack([gallery_embs, query_embs])
    all_lab    = np.concatenate([gallery_labels, query_labels])

    print(f"\nEmbedding dimension: {all_embs.shape[1]}")

    # 5. Geometry: PCA.
    print("\nRunning PCA (2D)...")
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    embs_pca = pca.fit_transform(all_embs)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance explained by 2 PCs: {var_explained:.1%}")
    plot_2d(
        embs_pca, all_lab,
        f"PCA — ESM-2 embeddings ({var_explained:.0%} var explained)",
        os.path.join(RESULTS_DIR, "lesson9_pca.png"),
        family_names,
    )

    # 6. Geometry: t-SNE (slower; shows non-linear structure).
    print("Running t-SNE (2D) — may take ~30s on CPU...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, n_iter=1000)
    embs_tsne = tsne.fit_transform(all_embs)
    plot_2d(
        embs_tsne, all_lab,
        "t-SNE — ESM-2 embeddings",
        os.path.join(RESULTS_DIR, "lesson9_tsne.png"),
        family_names,
    )

    # 7. Retrieval evaluation.
    print(f"\nRetrieval: cosine-similarity nearest-neighbour (k={K_NEIGHBORS})")
    top1, prec_k = retrieval_metrics(
        query_embs, query_labels, gallery_embs, gallery_labels, K_NEIGHBORS
    )
    rand_top1, rand_prec_k = random_retrieval_baseline(
        query_labels, gallery_labels, K_NEIGHBORS
    )

    # 8. Results summary.
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Sequences total         : {len(all_seqs)}")
    print(f"  Families (classes)      : {N_FAMILIES}")
    print(f"  Gallery size            : {len(gallery_seqs)}")
    print(f"  Query size              : {len(query_seqs)}")
    print(f"  Embedding dim           : {all_embs.shape[1]}")
    print(f"  PCA 2-PC variance       : {var_explained:.1%}")
    print()
    print(f"  {'Metric':<22}  {'ESM-2 retrieval':>18}  {'Random baseline':>16}")
    print(f"  {'-'*22}  {'-'*18}  {'-'*16}")
    print(f"  {'Top-1 accuracy':<22}  {top1:>17.3f}  {rand_top1:>15.3f}")
    print(f"  {'Precision@{k}':<22}  {prec_k:>17.3f}  {rand_prec_k:>15.3f}".format(k=K_NEIGHBORS))
    print("=" * 50)
    lift_top1 = (top1 / rand_top1) if rand_top1 > 0 else float("inf")
    lift_prec = (prec_k / rand_prec_k) if rand_prec_k > 0 else float("inf")
    print(f"\n  Top-1 lift over random  : {lift_top1:.1f}x")
    print(f"  Precision@k lift        : {lift_prec:.1f}x")

    if top1 > 0.7:
        print("\n  Interpretation: strong retrieval — the embedding space")
        print("  cleanly separates the synthetic families by cosine angle.")
    elif top1 > 0.4:
        print("\n  Interpretation: moderate retrieval — families partially")
        print("  overlap in embedding space (expected at high mutation rates).")
    else:
        print("\n  Interpretation: weak retrieval — try lowering MUTATION_RATE")
        print("  so within-family sequences are more similar.")

    print(
        """
Things to experiment with:
- Use a REAL protein family dataset (e.g. Pfam-A domains from UniProt) as
  the gallery; query with held-out members to measure true homology retrieval.
- Increase MODEL_NAME to "facebook/esm2_t12_35M_UR50D" — larger models
  produce cleaner embedding clusters at the cost of more compute.
- Compare pooling strategies (CLS token vs mean-pool vs max-pool) by reusing
  the pool() function from Lesson 5 and re-running retrieval for each.
- Replace sklearn NearestNeighbors with FAISS (pip install faiss-cpu) to
  scale to millions of sequences; FAISS uses the same cosine metric.
- Evaluate against a classical baseline: run BLAST or MMseqs2 on the same
  query/gallery split and compare precision@k to the pLM retrieval here.
- Cluster the gallery with sklearn KMeans and measure cluster purity
  (how often the majority label in a cluster matches all members).
"""
    )


if __name__ == "__main__":
    main()
