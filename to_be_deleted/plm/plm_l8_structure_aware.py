"""
Lesson 8: Structure-Aware Protein Language Models
==================================================
What you'll learn
-----------------
- Why sequence-only pLMs (ESM-2) are blind to 3D structure information.
- What the 3Di structural alphabet is and how it converts geometry into text.
- How ProstT5 encodes structural context alongside amino-acid sequence.
- How to run a head-to-head linear-probe comparison: ESM-2 vs ProstT5.

The 3Di structural alphabet (Foldseek)
---------------------------------------
Every residue sits in a small neighbourhood of backbone atoms. Foldseek
discretises that local geometry into one of ~20 "structure letters" — the
3Di alphabet. Crucially, a whole protein then becomes a SECOND STRING that
sits alongside the amino-acid sequence:

    AA sequence :  M  K  L  V  F  A  G ...
    3Di string  :  d  p  c  s  q  v  m ...

Structure letters are NOT amino acids; they are geometric tokens. Because
the representation is textual, a language model can process 3Di strings
exactly like amino-acid sequences. Foldseek generates them from PDB/mmCIF
coordinates. ProstT5 can PREDICT them from sequence alone (see below).

ProstT5 and the sequence <-> structure bridge
----------------------------------------------
ProstT5 (Rostlab/ProstT5) is a T5-based encoder-decoder. It was trained on
paired (AA sequence, Foldseek 3Di string) data in TWO directions:
  * Sequence -> 3Di translation (predict structure tokens from AA sequence)
  * 3Di -> Sequence translation (back-translate structure to sequence)

For EMBEDDING purposes we use its ENCODER, which sees both AA and 3Di
tokens and therefore produces structure-aware representations. When real
experimental structures are unavailable, ProstT5 translation lets you
generate 3Di tokens purely from sequence — no structure predictor needed.
This lesson uses ProstT5 in encoder-only mode (no translation step) to keep
things self-contained and CPU-runnable.

Newer models like SaProt (fuses AA + 3Di) and ESM-3 (multi-track: sequence,
structure, function) push this further — see "Things to experiment with".
"""

import time
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Configuration. Edit these to experiment.
# ---------------------------------------------------------------------------

# Sequence-only baseline (ESM-2, 8M params, 320-dim embeddings, very fast).
ESM_MODEL = "facebook/esm2_t6_8M_UR50D"

# Structure-aware model.
# ProstT5 is a T5 encoder-decoder; we extract encoder embeddings only (1024-dim).
# Prefix "<AA2fold>" is required by the ProstT5 tokenizer for AA input; the
# encoder then produces structure-aware hidden states.
PROST_MODEL = "Rostlab/ProstT5"

DATASET_NAME = "zhanglab/DeepSol"   # binary: 0 = insoluble, 1 = soluble
N_TRAIN = 300
N_TEST = 100
BATCH_SIZE = 4   # ProstT5 is larger; keep batch small for CPU

# ProstT5 requires this prefix so it knows the input is an amino-acid sequence
# (as opposed to a 3Di string, which uses "<fold2AA>").
PROST_PREFIX = "<AA2fold>"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_esm(sequences, model, tokenizer, device):
    """Mean-pool ESM-2 last-hidden-state, ignoring padding tokens.

    Reuses the same pattern as Lesson 1 — included here so this file is
    fully self-contained.
    """
    all_vecs = []
    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state   # (B, L, 320)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_vecs.append(pooled.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            done = min(i + BATCH_SIZE, len(sequences))
            print(f"    [{done}/{len(sequences)}]")
    return np.vstack(all_vecs)


def embed_prostt5(sequences, model, tokenizer, device):
    """Mean-pool ProstT5 encoder last-hidden-state.

    ProstT5's tokenizer expects:
    1. The prefix "<AA2fold>" to signal amino-acid input.
    2. Spaces between each amino-acid character (same as ProtBERT style).

    We prepend the prefix and space-separate before passing to the tokenizer.
    The encoder's hidden states already encode structural context because
    ProstT5 was trained on paired (sequence, 3Di) data — the encoder learned
    to represent sequences in a space that bridges the two alphabets.
    """
    all_vecs = []
    for i in range(0, len(sequences), BATCH_SIZE):
        batch_raw = sequences[i : i + BATCH_SIZE]
        # Space-separate residues; prepend required prefix.
        batch = [PROST_PREFIX + " " + " ".join(list(s)) for s in batch_raw]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state   # (B, L, 1024)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_vecs.append(pooled.cpu().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            done = min(i + BATCH_SIZE, len(sequences))
            print(f"    [{done}/{len(sequences)}]")
    return np.vstack(all_vecs)


def linear_probe(X_train, y_train, X_test, y_test):
    """Train a logistic regression on embeddings and return (accuracy, f1)."""
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load data ONCE — reused for both models.
    print(f"\nLoading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)
    train = ds["train"].select(range(N_TRAIN))
    test = ds["test"].select(range(N_TEST))
    train_seqs = train["sequence"]
    test_seqs = test["sequence"]
    y_train = np.array(train["label"])
    y_test = np.array(test["label"])
    majority = max(y_test.mean(), 1 - y_test.mean())
    print(f"  Train: {len(train_seqs)}, Test: {len(test_seqs)}")
    print(f"  Majority-class baseline: {majority:.3f}")

    results = []

    # -------------------------------------------------------------------------
    # 2. ESM-2 (sequence-only baseline)
    # -------------------------------------------------------------------------
    print(f"\n=== ESM-2 (sequence-only)  [{ESM_MODEL}] ===")
    esm_tok = AutoTokenizer.from_pretrained(ESM_MODEL)
    esm_model = AutoModel.from_pretrained(ESM_MODEL).to(device).eval()

    t0 = time.time()
    print("  Embedding train...")
    X_train_esm = embed_esm(train_seqs, esm_model, esm_tok, device)
    print("  Embedding test...")
    X_test_esm = embed_esm(test_seqs, esm_model, esm_tok, device)
    acc_esm, f1_esm = linear_probe(X_train_esm, y_train, X_test_esm, y_test)
    t_esm = time.time() - t0

    print(f"  acc={acc_esm:.3f}  f1={f1_esm:.3f}  ({t_esm:.0f}s)")
    results.append(("ESM-2 8M (seq-only)", X_train_esm.shape[1], acc_esm, f1_esm, t_esm))

    del esm_model, esm_tok
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. ProstT5 (structure-aware)
    # -------------------------------------------------------------------------
    # NOTE: This lesson uses ProstT5 in ENCODER-ONLY mode. Its encoder was
    # trained on (AA sequence, 3Di structure) pairs, so its hidden states carry
    # structural context even without real 3Di input — the model learned to
    # predict structure as part of pre-training.
    #
    # For truly structure-conditioned embeddings you would:
    #   a) Run Foldseek on PDB coordinates to get real 3Di tokens, then feed
    #      AA+3Di pairs to SaProt (westlake-repl/SaProt_650M_AF2).
    #   b) Or: use ProstT5 in decoder mode to *translate* the AA sequence
    #      into a predicted 3Di string, then feed that to SaProt — no
    #      experimental structure required.
    # Both routes are described in "Things to experiment with" below.
    print(f"\n=== ProstT5 (structure-aware)  [{PROST_MODEL}] ===")
    print("  Loading tokenizer/model (may take a moment on first run)...")
    prost_tok = T5Tokenizer.from_pretrained(PROST_MODEL, do_lower_case=False)
    prost_model = T5EncoderModel.from_pretrained(PROST_MODEL).to(device).eval()

    t0 = time.time()
    print("  Embedding train...")
    X_train_prost = embed_prostt5(train_seqs, prost_model, prost_tok, device)
    print("  Embedding test...")
    X_test_prost = embed_prostt5(test_seqs, prost_model, prost_tok, device)
    acc_prost, f1_prost = linear_probe(X_train_prost, y_train, X_test_prost, y_test)
    t_prost = time.time() - t0

    print(f"  acc={acc_prost:.3f}  f1={f1_prost:.3f}  ({t_prost:.0f}s)")
    results.append(("ProstT5 (struct-aware)", X_train_prost.shape[1], acc_prost, f1_prost, t_prost))

    del prost_model, prost_tok
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Comparison table
    # -------------------------------------------------------------------------
    # Be honest: on a small solubility slice ProstT5 may NOT beat ESM-2.
    # Solubility correlates more with amino-acid composition than 3D geometry.
    # The lesson's value is the CONCEPT and the API pattern — structure-aware
    # embeddings pay off most on geometry-dependent tasks (fold classification,
    # enzyme commission numbers, binding-site prediction).
    print("\n" + "=" * 64)
    print(f"{'Model':<26} {'Dim':>5} {'Acc':>6} {'F1':>6} {'Time(s)':>8}")
    print("-" * 64)
    for name, dim, acc, f1, t in results:
        print(f"{name:<26} {dim:>5} {acc:>6.3f} {f1:>6.3f} {t:>8.0f}")
    print("-" * 64)
    print(f"{'Majority baseline':<26} {'':>5} {majority:>6.3f}")
    print("=" * 64)

    print(
        """
Things to experiment with:
- Use SaProt (westlake-repl/SaProt_650M_AF2) with REAL Foldseek 3Di tokens:
    run `foldseek structureto3di` on a PDB file to get the 3Di string, then
    interleave AA+3Di tokens as SaProt expects (e.g. "MdKpLc...").
- Generate 3Di tokens with ProstT5 translation (no structure file needed):
    set the input prefix to "<AA2fold>", run the full T5 model (not just the
    encoder), and decode the output tokens — that gives a predicted 3Di string
    you can feed into SaProt.
- Try a structure-DEPENDENT task where geometry matters more than composition:
    fold classification (SCOPe / CATH labels) or enzyme commission (EC) numbers
    — that's where structure-aware models show the largest gains over ESM-2.
- Compare ESM-3 (evolutionaryscale/esm3-sm-open-v1): it has separate sequence,
    structure, and function tracks; you can embed with any combination.
- Feed per-residue ProstT5 states (shape B x L x 1024) directly into a GNN
    as node features — ties this lesson directly to the GNN series (Lessons 6-7).
"""
    )


if __name__ == "__main__":
    main()
