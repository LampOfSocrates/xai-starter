"""
Lesson 6: Attention Maps and Contact Prediction
================================================
What you'll learn
-----------------
- Extract per-head attention tensors from ESM-2 with output_attentions=True.
- Why certain attention heads encode residue-residue CONTACTS with no supervision.
- How to evaluate heads with Precision@L and AUC against a ground-truth contact map.
- Build a simple unsupervised contact predictor by averaging the best heads.
- Why the supervised "logistic regression on all heads" (the real ESM contact head)
  works even better.

The key result
--------------
Rao et al. (2020) "Transformer protein language models are unsupervised structure
learners" showed that individual attention heads in ESM models learn to encode
which residue pairs are physically close in 3-D space — despite being trained
only on sequence reconstruction (masked language modelling). No structural
supervision whatsoever.

The intuition is that coevolution of contacting residues (if A mutates, B must
compensate, because they touch) is exactly the kind of statistical pattern that
a transformer LM picks up to fill in masked positions. The model learns contacts
because contacts constrain sequences.

Practical note
--------------
In this lesson we generate a synthetic protein structure (two ideal alpha-helices
packed together) with known CA coordinates, so the lesson is fully self-contained
and requires no PDB file download. In real work you would load a PDB file and its
corresponding sequence; every other part of the code stays the same.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # no display needed; saves to file
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # 6 layers, 20 heads — fast on CPU

# Minimum sequence separation for a "long-range" contact (standard in the field).
SEQ_SEP_MIN = 6

# CA-CA distance threshold (Angstroms) used to define a contact.
CONTACT_DIST_THRESHOLD = 8.0

# Top-k heads to average when building the unsupervised predictor.
TOP_K_HEADS = 5

RESULTS_DIR = "./results"


# ---------------------------------------------------------------------------
# Synthetic structure generation
# ---------------------------------------------------------------------------

def _ideal_helix_ca(n_residues, rise_per_residue=1.5, radius=2.3, omega=100.0):
    """Generate CA coordinates for a single ideal alpha-helix.

    Parameters follow standard helix geometry (3.6 residues/turn, 1.5 Å rise).
    omega is in degrees per residue.

    Returns array of shape (n_residues, 3).
    """
    coords = []
    for i in range(n_residues):
        angle = np.radians(omega * i)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = rise_per_residue * i
        coords.append([x, y, z])
    return np.array(coords)


def make_synthetic_structure():
    """Build two ideal helices packed side-by-side and return (sequence, CA coords).

    Helix 1: residues 0-24  (25 residues)
    Linker  : residues 25-27 (3 residues, extended)
    Helix 2: residues 28-52 (25 residues)

    Total: 53 residues. The two helices are offset laterally so they have
    real inter-helix contacts, producing a non-trivial contact map.

    In practice you would replace this with: load PDB -> extract CA coords and
    the corresponding SEQRES / ATOM sequence.
    """
    # Amino-acid sequence: alanine-rich helices connected by a glycine linker.
    # A real lesson would use the actual sequence of a PDB entry.
    helix_seq = "AAKEAAKAAKEAAKAAKEAAKAAKEA"   # 25 residues, helix-favoring
    linker_seq = "GGG"
    sequence = helix_seq + linker_seq + helix_seq  # 53 residues

    # Helix 1 coordinates (standard orientation).
    h1 = _ideal_helix_ca(25)

    # Helix 2: translated +8 Å in X and +4 Å in Z, antiparallel (z reversed).
    h2_raw = _ideal_helix_ca(25)
    h2 = h2_raw.copy()
    h2[:, 0] += 8.0          # lateral offset — places it ~8 Å away from H1
    h2[:, 2] = h2_raw[:, 2].max() - h2_raw[:, 2]  # reverse z -> antiparallel
    h2[:, 2] += h1[-1, 2] + 4.0  # stack above H1 in z

    # Linker: simple extended chain bridging the two helices.
    linker_start = h1[-1]
    linker_end = h2[0]
    linker = np.array([
        linker_start + (linker_end - linker_start) * t
        for t in [0.25, 0.5, 0.75]
    ])

    ca_coords = np.vstack([h1, linker, h2])  # (53, 3)
    return sequence, ca_coords


# ---------------------------------------------------------------------------
# Contact map computation
# ---------------------------------------------------------------------------

def compute_contact_map(ca_coords, dist_threshold=CONTACT_DIST_THRESHOLD,
                        seq_sep_min=SEQ_SEP_MIN):
    """Return binary contact matrix from CA coordinates.

    contact[i, j] = 1 if ||CA_i - CA_j|| < dist_threshold
                       AND |i - j| >= seq_sep_min.

    Short-range pairs (|i-j| < seq_sep_min) are excluded — they are always
    close by chain connectivity, so they contain no structural information.
    """
    L = len(ca_coords)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]   # (L, L, 3)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                # (L, L)

    contacts = (dist < dist_threshold).astype(float)
    # Zero out trivial short-range and diagonal pairs.
    for i in range(L):
        for j in range(L):
            if abs(i - j) < seq_sep_min:
                contacts[i, j] = 0.0
    return contacts


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def get_attention_maps(sequence, model, tokenizer, device):
    """Run ESM-2 on `sequence` and return symmetrised per-head attention maps.

    Returns:
        attn: numpy array of shape (n_layers, n_heads, L, L)
              where L = len(sequence). The <cls> and <eos> tokens have been
              stripped. Each head map is symmetrised: (A + A^T) / 2.
    """
    inputs = tokenizer(sequence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is a tuple of (n_layers,) tensors,
    # each of shape (1, n_heads, L+2, L+2) — the +2 is <cls> and <eos>.
    n_layers = len(outputs.attentions)
    n_heads = outputs.attentions[0].shape[1]
    L = len(sequence)

    attn = np.zeros((n_layers, n_heads, L, L), dtype=np.float32)
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        a = layer_attn[0].cpu().numpy()   # (n_heads, L+2, L+2)
        # Strip <cls> (index 0) and <eos> (index -1).
        a = a[:, 1:-1, 1:-1]             # (n_heads, L, L)
        # Symmetrise: contacts are symmetric, attention is not.
        a = (a + a.transpose(0, 2, 1)) / 2.0
        attn[layer_idx] = a

    return attn   # (n_layers, n_heads, L, L)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def precision_at_L(pred_map, true_contacts, L, seq_sep_min=SEQ_SEP_MIN):
    """Precision of the top-L predicted pairs among long-range contacts.

    Precision@L is the standard metric in the contact prediction literature:
    take the L highest-scoring long-range pairs from the predicted map, count
    what fraction are true contacts.
    """
    # Collect all long-range (i, j) pairs with i < j.
    pairs = [
        (pred_map[i, j], true_contacts[i, j])
        for i in range(L)
        for j in range(i + seq_sep_min, L)
    ]
    if not pairs:
        return 0.0
    pairs.sort(key=lambda x: -x[0])        # sort by predicted score, descending
    top_L = pairs[:L]
    return float(np.mean([p[1] for p in top_L]))


def auc_score(pred_map, true_contacts, L, seq_sep_min=SEQ_SEP_MIN):
    """ROC-AUC for long-range contact prediction."""
    scores, labels = [], []
    for i in range(L):
        for j in range(i + seq_sep_min, L):
            scores.append(pred_map[i, j])
            labels.append(true_contacts[i, j])
    labels = np.array(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5   # degenerate
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Build (or load) the protein structure.
    print("\nGenerating synthetic two-helix structure...")
    sequence, ca_coords = make_synthetic_structure()
    L = len(sequence)
    print(f"  Sequence length: {L}")
    print(f"  Sequence: {sequence}")
    print(f"  CA coords shape: {ca_coords.shape}")

    true_contacts = compute_contact_map(ca_coords)
    n_contacts = int(true_contacts.sum()) // 2   # upper triangle
    print(f"  Long-range contacts (CA < {CONTACT_DIST_THRESHOLD} Å, sep >= {SEQ_SEP_MIN}): {n_contacts}")

    # 2. Load ESM-2 and extract attention maps.
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print("Extracting attention maps (output_attentions=True)...")
    attn = get_attention_maps(sequence, model, tokenizer, device)
    n_layers, n_heads = attn.shape[:2]
    print(f"  Attention tensor shape: {attn.shape}  (layers, heads, L, L)")

    # 3. Score every (layer, head) against the ground-truth contact map.
    print("\nEvaluating every attention head...")
    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            head_map = attn[layer, head]
            prec = precision_at_L(head_map, true_contacts, L)
            auc = auc_score(head_map, true_contacts, L)
            results.append((prec, auc, layer, head))

    results.sort(key=lambda x: -x[0])   # rank by Precision@L

    print(f"\n{'Rank':<6}{'Layer':>6}{'Head':>6}{'Prec@L':>10}{'AUC':>8}")
    print("-" * 40)
    for rank, (prec, auc, layer, head) in enumerate(results[:10], 1):
        print(f"{rank:<6}{layer:>6}{head:>6}{prec:>10.3f}{auc:>8.3f}")

    best_prec, best_auc, best_layer, best_head = results[0]
    print(f"\nBest head: layer={best_layer}, head={best_head}")
    print(f"  Precision@L = {best_prec:.3f}")
    print(f"  AUC         = {best_auc:.3f}")
    print(f"  (Random baseline Prec@L ≈ {n_contacts / (L * (L - 1) / 2):.3f})")

    # 4. Unsupervised contact predictor: average the top-k heads.
    print(f"\nBuilding unsupervised predictor: average of top-{TOP_K_HEADS} heads...")
    top_maps = np.stack([attn[layer, head] for _, _, layer, head in results[:TOP_K_HEADS]])
    avg_map = top_maps.mean(axis=0)

    avg_prec = precision_at_L(avg_map, true_contacts, L)
    avg_auc = auc_score(avg_map, true_contacts, L)
    print(f"  Avg-top-{TOP_K_HEADS} Precision@L = {avg_prec:.3f}")
    print(f"  Avg-top-{TOP_K_HEADS} AUC         = {avg_auc:.3f}")

    # Also try averaging ALL heads (a common baseline).
    all_avg_map = attn.mean(axis=(0, 1))
    all_avg_prec = precision_at_L(all_avg_map, true_contacts, L)
    all_avg_auc = auc_score(all_avg_map, true_contacts, L)
    print(f"  All-heads average Precision@L = {all_avg_prec:.3f}")
    print(f"  All-heads average AUC         = {all_avg_auc:.3f}")

    # 5. Save heatmaps.
    best_map = attn[best_layer, best_head]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(true_contacts, cmap="Blues", vmin=0, vmax=1, origin="upper")
    axes[0].set_title("Ground-truth contacts\n(CA–CA < 8 Å, sep ≥ 6)", fontsize=10)
    axes[0].set_xlabel("Residue j")
    axes[0].set_ylabel("Residue i")

    im1 = axes[1].imshow(best_map, cmap="hot_r", origin="upper")
    axes[1].set_title(
        f"Best attention head\n(layer {best_layer}, head {best_head},"
        f" Prec@L={best_prec:.2f})",
        fontsize=10,
    )
    axes[1].set_xlabel("Residue j")
    axes[1].set_ylabel("Residue i")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(avg_map, cmap="hot_r", origin="upper")
    axes[2].set_title(
        f"Unsupervised predictor\n(avg top-{TOP_K_HEADS} heads,"
        f" Prec@L={avg_prec:.2f})",
        fontsize=10,
    )
    axes[2].set_xlabel("Residue j")
    axes[2].set_ylabel("Residue i")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "l6_attention_contacts.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nHeatmap saved to {out_path}")

    # 6. Brief note on the supervised approach.
    print(
        "\nNote: the ESM contact head (used in esm.predict_contacts) fits a"
        " logistic regression on the concatenated attention maps from all"
        " (layer, head) pairs — giving it L*(L-1)/2 training examples per"
        " protein, supervised on experimentally-determined contacts."
        " That's why it dramatically outperforms any single head."
    )

    print(
        """
Things to experiment with:
- Use a real PDB structure + its sequence: download any PDB file, parse CA
  coordinates with Biopython (Bio.PDB), replace `make_synthetic_structure()`.
- Use a deeper ESM-2 model ("facebook/esm2_t12_35M_UR50D" or larger) — deeper
  models tend to have heads that correlate more strongly with contacts.
- Fit a logistic regression over ALL (layer, head) attention maps at once,
  which is the actual approach in esm.predict_contacts:
      from sklearn.linear_model import LogisticRegression
      X = attn.reshape(n_layers * n_heads, L * L).T  # (L*L, layers*heads)
      y = true_contacts.ravel()
- Apply APC (Average Product Correction): subtract the outer product of row/col
  means, which removes a background signal and improves contact precision.
- Split Precision@L by contact range: short (6<=sep<12), medium (12<=sep<24),
  long (sep>=24) to see which range benefits most from attention.
- Vary TOP_K_HEADS and CONTACT_DIST_THRESHOLD to see how sensitive the results
  are to these hyperparameters.
"""
    )


if __name__ == "__main__":
    main()
