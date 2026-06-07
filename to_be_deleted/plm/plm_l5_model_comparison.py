"""
Lesson 5: Compare pLMs Across Models and Pooling Strategies
============================================================
What you'll learn
-----------------
- Run a small grid: {models} x {pooling methods} on the same task.
- See how model size and pooling choice affect performance.
- Build a reusable benchmarking pattern you can extend.

Why we use the "embedding probe" style for comparison
-----------------------------------------------------
Fine-tuning every model would take hours. We instead use Lesson 1's recipe
(frozen pLM + logistic regression on top) which makes the comparison FAST
and FAIR — no hyperparameter shenanigans, just "which embedding is better?"

If a model wins as a frozen feature extractor, it usually also wins after
fine-tuning. So this is a reasonable cheap proxy.

Output: a CSV table of {model, method, accuracy, F1, time}.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration — the comparison grid.
# ---------------------------------------------------------------------------

MODELS = [
    # (model name, friendly label)
    ("facebook/esm2_t6_8M_UR50D",   "ESM-2 8M"),
    ("facebook/esm2_t12_35M_UR50D", "ESM-2 35M"),
    # Uncomment if you have a GPU (or significant patience):
    # ("facebook/esm2_t30_150M_UR50D", "ESM-2 150M"),
    # ProtBERT uses a DIFFERENT vocab (spaces between AAs) — see comment in embed():
    # ("Rostlab/prot_bert", "ProtBERT"),
]

# Pooling strategies to compare. Each takes (hidden, mask) and returns one
# vector per sequence.
METHODS = ["mean-pool", "max-pool", "cls-token"]

DATASET_NAME = "zhanglab/DeepSol"
N_TRAIN = 300
N_TEST = 100
BATCH_SIZE = 8
RESULTS_DIR = "./results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "lesson5_comparison.csv")


def pool(hidden, mask, method):
    """Reduce (B, L, D) to (B, D) according to the chosen pooling method."""
    if method == "mean-pool":
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1)
    if method == "max-pool":
        # Set padding positions to -inf so they don't win the max.
        m = mask.unsqueeze(-1).bool()
        masked = hidden.masked_fill(~m, float("-inf"))
        return masked.max(dim=1).values
    if method == "cls-token":
        return hidden[:, 0, :]
    raise ValueError(f"Unknown pooling method: {method}")


def embed(sequences, model, tokenizer, device, method, batch_size=BATCH_SIZE):
    """Convert a list of sequences into one vector each.

    Note on ProtBERT-style models: their tokenizer expects spaces between
    amino acids, e.g. "M K T V". If you add a ProtBERT entry above, do
    `[' '.join(s) for s in sequences]` before passing in.
    """
    out = []
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
        pooled = pool(hidden, inputs["attention_mask"], method)
        out.append(pooled.cpu().numpy())
    return np.vstack(out)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset ONCE — reused across all models.
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    train = ds["train"].select(range(N_TRAIN))
    test = ds["test"].select(range(N_TEST))
    train_seqs = train["sequence"]
    test_seqs = test["sequence"]
    y_train = np.array(train["label"])
    y_test = np.array(test["label"])

    rows = []
    for model_name, label in MODELS:
        print(f"\n=== {label}  ({model_name}) ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

        for method in METHODS:
            t0 = time.time()
            X_train = embed(train_seqs, model, tokenizer, device, method)
            X_test = embed(test_seqs, model, tokenizer, device, method)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            elapsed = time.time() - t0

            rows.append(
                {
                    "model": label,
                    "method": method,
                    "embed_dim": X_train.shape[1],
                    "accuracy": round(acc, 3),
                    "f1": round(f1, 3),
                    "time_s": round(elapsed, 1),
                }
            )
            print(
                f"  {method:<12} dim={X_train.shape[1]:>4}  acc={acc:.3f}  f1={f1:.3f}  ({elapsed:.0f}s)"
            )

        # Free memory before loading the next model.
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- Summary ---------------------------------------------------------
    df = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved comparison table to: {RESULTS_CSV}")

    print(
        """
Things to experiment with:
- Add ProtBERT to MODELS — note the space-between-amino-acids quirk in embed().
- Add a regression task (e.g. proteinea/fluorescence) — swap LogisticRegression
  for sklearn.linear_model.Ridge and use Spearman correlation as the metric.
- Bigger N_TRAIN / N_TEST will give a more reliable comparison.
- Plot the table: import matplotlib.pyplot as plt; df.pivot(...).plot.bar()
"""
    )


if __name__ == "__main__":
    main()
