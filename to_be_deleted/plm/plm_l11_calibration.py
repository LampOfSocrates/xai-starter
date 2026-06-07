"""
Lesson 11: Calibration and Uncertainty
=======================================
What you'll learn
-----------------
- Measure calibration: a model that says "90% confident" should be right 90%
  of the time — and often isn't.
- Draw a reliability diagram to visualise miscalibration.
- Compute Expected Calibration Error (ECE) and Brier score from scratch.
- Fix miscalibration with temperature scaling (one scalar parameter, fits in
  seconds on CPU).
- Understand why zero-shot pLM scores from Lesson 2 face the same problem.

What calibration actually means
--------------------------------
Accuracy answers "am I right?"; calibration answers "do I know when I'm right?"
A perfectly calibrated classifier has predicted_confidence == observed_accuracy
in every confidence bin. The reliability diagram shows this: a well-calibrated
model hugs the diagonal.

Temperature scaling
-------------------
Before the sigmoid/softmax, we have raw logits z. Temperature scaling divides
them by a learned scalar T: p = sigmoid(z / T). T > 1 softens (flattens)
probabilities; T < 1 sharpens them. We fit T by minimising NLL on a small
held-out calibration split — the train/test weights never change, so accuracy
on test is unchanged. This is the simplest post-hoc calibration method.

sklearn's CalibratedClassifierCV (method='sigmoid' for Platt scaling or
method='isotonic') can do the same job and often generalises better when you
have enough calibration data.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# ---------------------------------------------------------------------------
# Configuration. Edit these to experiment.
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # 8M params, 320-dim, fast on CPU
DATASET_NAME = "zhanglab/DeepSol"           # binary solubility labels

N_TRAIN = 400    # sequences used to fit the logistic regression head
N_CAL = 150      # held-out calibration split — used ONLY to fit temperature T
N_TEST = 150     # final evaluation; never touched during training or calibration
BATCH_SIZE = 8

N_ECE_BINS = 10  # number of equal-width confidence bins for the reliability diagram

RESULTS_DIR = "./results"
FIG_PATH = os.path.join(RESULTS_DIR, "lesson11_reliability.png")

TEMP_LR = 0.01       # learning rate for temperature optimisation
TEMP_STEPS = 500     # SGD steps for fitting T


# ---------------------------------------------------------------------------
# Embedding helpers (identical recipe to Lesson 1)
# ---------------------------------------------------------------------------

def get_embeddings(sequences, model, tokenizer, device, batch_size=BATCH_SIZE):
    """Mean-pool ESM-2 last hidden states into one vector per sequence."""
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
            hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(pooled.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {min(i + batch_size, len(sequences))}/{len(sequences)}")
    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Calibration metrics (implemented from scratch so you can see the maths)
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, probs, n_bins=N_ECE_BINS):
    """ECE: average |confidence - accuracy| weighted by bin size.

    Intuition: split predictions into confidence bins; in each bin compare
    the mean confidence to the fraction of correct predictions. ECE is the
    weighted average of those gaps.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        mean_conf = probs[mask].mean()
        mean_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(mean_conf - mean_acc)
    return ece


def brier_score(y_true, probs):
    """Mean squared error between probabilities and binary outcomes.

    Lower is better; a perfect forecast scores 0, random 50/50 scores 0.25.
    Brier score rewards both sharpness (confident predictions) and calibration.
    """
    return float(np.mean((probs - y_true) ** 2))


def reliability_diagram(y_true, probs, ax, title, n_bins=N_ECE_BINS):
    """Draw a reliability diagram on the given axes.

    Each bar shows observed accuracy for predictions in that confidence bin.
    The diagonal represents perfect calibration.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(y_true[mask].mean())
        bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_accs = np.array(bin_accs)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(bin_centers, bin_accs, width=0.9 / n_bins, alpha=0.7,
           color="steelblue", edgecolor="white", label="Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (predicted probability)")
    ax.set_ylabel("Accuracy (observed fraction correct)")
    ax.set_title(title)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(logits_cal, y_cal, lr=TEMP_LR, n_steps=TEMP_STEPS):
    """Find scalar T that minimises NLL of sigmoid(logits / T) on the cal set.

    logits_cal: raw pre-sigmoid scores, shape (N,)  — from clf.decision_function
    y_cal:      binary labels, shape (N,)
    Returns the optimal float T.
    """
    # Single learnable parameter, initialised to 1 (no change from baseline).
    T = torch.nn.Parameter(torch.ones(1))
    optimizer = optim.Adam([T], lr=lr)
    logits_t = torch.tensor(logits_cal, dtype=torch.float32)
    labels_t = torch.tensor(y_cal, dtype=torch.float32)

    for step in range(n_steps):
        optimizer.zero_grad()
        # Clamp T > 0.05 to prevent collapse / explosion
        T_clamped = T.clamp(min=0.05)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_t / T_clamped, labels_t
        )
        loss.backward()
        optimizer.step()

    return float(T.clamp(min=0.05).detach())


def logits_to_probs(logits, temperature=1.0):
    """Apply temperature and sigmoid to get calibrated probabilities."""
    return torch.sigmoid(torch.tensor(logits / temperature)).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load model and data.
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    total_needed = N_TRAIN + N_CAL + N_TEST
    pool = ds["train"].select(range(total_needed))
    train_ds = pool.select(range(N_TRAIN))
    cal_ds   = pool.select(range(N_TRAIN, N_TRAIN + N_CAL))
    test_ds  = pool.select(range(N_TRAIN + N_CAL, total_needed))
    print(f"Split: {len(train_ds)} train  |  {len(cal_ds)} cal  |  {len(test_ds)} test")

    # 2. Extract embeddings for all three splits.
    print("\nExtracting train embeddings...")
    X_train = get_embeddings(train_ds["sequence"], model, tokenizer, device)
    print("Extracting calibration embeddings...")
    X_cal = get_embeddings(cal_ds["sequence"], model, tokenizer, device)
    print("Extracting test embeddings...")
    X_test = get_embeddings(test_ds["sequence"], model, tokenizer, device)

    y_train = np.array(train_ds["label"])
    y_cal   = np.array(cal_ds["label"])
    y_test  = np.array(test_ds["label"])

    # 3. Train logistic regression head (same as Lesson 1).
    print("\nTraining logistic regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    # predict_proba returns [P(class0), P(class1)]; we want P(soluble).
    probs_before = clf.predict_proba(X_test)[:, 1]
    acc_before = accuracy_score(y_test, clf.predict(X_test))
    ece_before = expected_calibration_error(y_test, probs_before)
    bs_before  = brier_score(y_test, probs_before)

    print(f"\n--- Before calibration ---")
    print(f"  Accuracy   : {acc_before:.3f}")
    print(f"  ECE        : {ece_before:.4f}  (0 = perfect, 0.25 = random)")
    print(f"  Brier score: {bs_before:.4f}   (lower is better)")

    # 4. Fit temperature T on the calibration split (never touches test data).
    logits_cal  = clf.decision_function(X_cal)   # raw pre-sigmoid scores
    logits_test = clf.decision_function(X_test)

    print("\nFitting temperature scaling on calibration split...")
    T = fit_temperature(logits_cal, y_cal)
    print(f"  Learned temperature T = {T:.4f}  (>1 softens, <1 sharpens)")

    probs_after = logits_to_probs(logits_test, temperature=T)
    acc_after = accuracy_score(y_test, (probs_after >= 0.5).astype(int))
    ece_after = expected_calibration_error(y_test, probs_after)
    bs_after  = brier_score(y_test, probs_after)

    print(f"\n--- After temperature scaling ---")
    print(f"  Accuracy   : {acc_after:.3f}  (should be unchanged)")
    print(f"  ECE        : {ece_after:.4f}")
    print(f"  Brier score: {bs_after:.4f}")
    print(f"\n  ECE reduction: {ece_before - ece_after:+.4f}")

    # 5. sklearn alternative: Platt scaling (sigmoid) or isotonic regression.
    # Both wrap the base classifier; method='isotonic' is more flexible but
    # needs more calibration data to avoid overfitting.
    clf_platt = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, C=1.0), method="sigmoid", cv=3
    )
    clf_platt.fit(X_train, y_train)
    probs_platt = clf_platt.predict_proba(X_test)[:, 1]
    ece_platt   = expected_calibration_error(y_test, probs_platt)
    print(f"\n  sklearn Platt (CalibratedClassifierCV) ECE: {ece_platt:.4f}")

    # 6. Reliability diagrams: before vs after, side by side.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    reliability_diagram(
        y_test, probs_before, axes[0],
        f"Before calibration\nECE={ece_before:.4f}  Brier={bs_before:.4f}"
    )
    reliability_diagram(
        y_test, probs_after, axes[1],
        f"After temperature scaling (T={T:.3f})\nECE={ece_after:.4f}  Brier={bs_after:.4f}"
    )
    fig.suptitle("Lesson 11 — Reliability Diagrams (DeepSol / ESM-2 8M)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    print(f"\nReliability diagram saved to: {FIG_PATH}")

    # 7. Connection to pLM uncertainty.
    print(
        "\n--- pLM uncertainty context ---\n"
        "  Lesson 2's zero-shot variant scorer uses log-likelihood ratios as\n"
        "  confidence proxies. Those scores are also uncalibrated — a ratio of\n"
        "  +3.0 does not mean '95% likely deleterious'. Calibrating against\n"
        "  ProteinGym DMS assay scores would directly address this.\n"
        "\n"
        "  For epistemic uncertainty (does the model know what it doesn't know?):\n"
        "  - MC-Dropout: run the model N times with dropout ON; the spread of\n"
        "    predictions estimates uncertainty.\n"
        "  - Deep ensembles: train K independent heads; disagreement = uncertainty.\n"
        "  Both also apply to the pLM head in Lesson 3 (fine-tuned classification)."
    )

    print(
        """
Things to experiment with:
- Change method='sigmoid' to method='isotonic' in CalibratedClassifierCV
  and compare ECE — isotonic is more flexible but needs more calibration data.
- Calibrate the Lesson 2 zero-shot variant scorer against ProteinGym labels:
  bin the log-likelihood-ratio scores and plot observed enrichment per bin.
- Add MC-Dropout uncertainty: wrap the Lesson 3 fine-tuned model, set
  model.train() at inference, run 30 forward passes, measure std(p).
- Try deep ensembles: train K=5 logistic regression heads with different
  random seeds and use mean/std of their probabilities.
- Deliberately create distribution shift: train on one organism, calibrate on
  another; watch ECE change and ask whether temperature still fixes it.
- Plot per-class reliability diagrams — miscalibration is often asymmetric
  (the model may be overconfident on one label and underconfident on the other).
"""
    )


if __name__ == "__main__":
    main()
