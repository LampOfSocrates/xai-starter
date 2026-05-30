"""
Lesson 1: Protein Embeddings + Linear Probe
============================================
What you'll learn
-----------------
- Load a small Protein Language Model (ESM-2, 8M params).
- Run sequences through it to get FIXED-SIZE numerical vectors ("embeddings").
- Train a tiny scikit-learn classifier on top of those embeddings.

Why this is the right place to start
------------------------------------
This is the cheapest way to USE a pLM. The pLM itself is FROZEN — you never
update its weights. You're just treating the pLM as a "feature extractor"
that converts a protein sequence (a string of amino acids) into a vector
that captures evolutionary / structural / functional information.

Then you do classical ML on those vectors: logistic regression, random
forest, whatever. This is called a "linear probe" when the head is linear.

When this is the right approach
-------------------------------
- You have very little labelled data (linear probes resist overfitting).
- You have no GPU.
- You want a quick baseline before going further.

Runs in a few minutes on CPU.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Configuration. Edit these to experiment.
# ---------------------------------------------------------------------------

# ESM-2 comes in many sizes. The number after `t` is the layer count;
# the number after that is the parameter count.
#   esm2_t6_8M_UR50D     -> 8M params,  320-dim embeddings (default here, fast)
#   esm2_t12_35M_UR50D   -> 35M params, 480-dim embeddings
#   esm2_t30_150M_UR50D  -> 150M params, 640-dim embeddings (GPU recommended)
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# DeepSol predicts whether a protein is "soluble" when expressed in E. coli.
# Binary classification: label 0 = insoluble, label 1 = soluble.
DATASET_NAME = "zhanglab/DeepSol"

# Use small subsets so this is fast. Increase once you trust it works.
N_TRAIN = 500
N_TEST = 200
BATCH_SIZE = 8


def get_embeddings(sequences, model, tokenizer, device, batch_size=BATCH_SIZE):
    """Convert a list of protein sequences into one fixed-size vector each.

    Steps inside this function:
    1. Tokenize each sequence. ESM-2's tokenizer maps each amino acid to one
       token (plus a special <cls> at the start and <eos> at the end).
    2. Run the pLM in `eval` mode and with `torch.no_grad()` (we don't need
       gradients — we're not training).
    3. The model returns one vector per token. We MEAN-POOL across tokens to
       get one vector per sequence. We ignore padding tokens during pooling
       (that's what `attention_mask` is for).
    """
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]

        # Tokenize the whole batch. Padding=True pads to the longest sequence
        # in this batch. truncation=True caps at max_length to fit in memory.
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

        # outputs.last_hidden_state has shape (batch, seq_len, hidden_dim).
        # For ESM-2 8M: hidden_dim == 320.
        hidden = outputs.last_hidden_state

        # Mean-pool over the sequence dimension, ignoring padding tokens.
        # attention_mask is 1 for real tokens, 0 for padding.
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (B, hidden_dim)

        all_embeddings.append(pooled.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  embedded {min(i + batch_size, len(sequences))}/{len(sequences)}")

    return np.vstack(all_embeddings)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the pre-trained pLM (no fine-tuning).
    # `.eval()` puts dropout etc. into inference mode.
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # 2. Load some labelled protein data.
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    train = ds["train"].select(range(N_TRAIN))
    test = ds["test"].select(range(N_TEST))

    print(f"Train sequences: {len(train)}, test sequences: {len(test)}")
    print(f"Example: label={train[0]['label']}, seq[:60]={train[0]['sequence'][:60]}...")

    # 3. Convert sequences -> embeddings.
    print("\nExtracting train embeddings...")
    X_train = get_embeddings(train["sequence"], model, tokenizer, device)
    print("Extracting test embeddings...")
    X_test = get_embeddings(test["sequence"], model, tokenizer, device)

    y_train = np.array(train["label"])
    y_test = np.array(test["label"])

    print(f"\nX_train shape: {X_train.shape}  (n_sequences, embedding_dim)")
    print(f"X_test  shape: {X_test.shape}")

    # 4. Train a classical classifier on top.
    # The pLM is frozen — only the logistic regression learns.
    print("\nTraining logistic regression on the embeddings...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    # 5. Evaluate.
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"\nResults:")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  F1:       {f1:.3f}")
    print(f"  Baseline (always-predict-majority): {max(y_test.mean(), 1 - y_test.mean()):.3f}")

    print(
        """
Things to experiment with:
- MODEL_NAME = "facebook/esm2_t12_35M_UR50D" (better, slower)
- Replace LogisticRegression with sklearn.ensemble.RandomForestClassifier
- Try a different pooling strategy:
    pooled = hidden[:, 0, :]              # use only the [CLS] token
    pooled = hidden.max(dim=1).values     # max-pool instead of mean-pool
- Increase N_TRAIN / N_TEST for a more reliable measurement
- Swap DATASET_NAME to "proteinea/solubility" or any other classification set
"""
    )


if __name__ == "__main__":
    main()
