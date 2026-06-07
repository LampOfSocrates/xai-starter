"""
Lesson 4: Per-Residue Prediction (Token Classification)
========================================================
What you'll learn
-----------------
- The difference between SEQUENCE-level prediction (one label per protein) and
  RESIDUE-level prediction (one label per amino acid).
- Use AutoModelForTokenClassification — same as NER in NLP.
- The trickiest part: ALIGNING labels to tokens (handling <cls>, <eos>, padding).

The example task: 3-state secondary structure (Q3)
--------------------------------------------------
Each residue is labelled:
   H = alpha helix
   E = beta sheet
   C = coil / random coil

   Input:  "MKTVRQERLKSIVRILERSKEPVSGA..."
   Output: "CCCHHHHHHHHHHHHEEEEECCCCCC..."

Structurally identical to NER: each token gets a label, padding tokens get
the special label `-100` which the loss function ignores.

About the dataset
-----------------
SS3 datasets exist on Hugging Face under names like:
   "proteinea/secondary_structure"
   "El-Husseini/Protein_secondary_structure"
   FLIP / TAPE benchmarks
Column names vary. This script prints `dataset.features` so you can adjust
SEQUENCE_KEY / LABEL_KEY below to match. If the dataset doesn't exist or
columns differ, see the README for a list of working alternatives.
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuration — adjust the keys below to match your chosen dataset.
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DATASET_NAME = "proteinea/secondary_structure"
SEQUENCE_KEY = "input"          # column containing the amino acid sequence
LABEL_KEY = "labels_ss3"        # column containing the per-residue SS labels
LABEL_NAMES = ["H", "E", "C"]   # 3-state SS
OUTPUT_DIR = "./results/lesson4"
N_TRAIN = 200
N_VAL = 50

label_to_id = {l: i for i, l in enumerate(LABEL_NAMES)}


def preprocess_example(example, tokenizer):
    """Tokenize one sequence and align per-residue labels to tokens.

    Why alignment is non-trivial:
    - The tokenizer adds <cls> at position 0 and <eos> at the end.
    - These have no matching amino acid in the input, so we set their
      labels to -100. The cross-entropy loss IGNORES positions with label -100.
    - If the sequence is truncated to max_length, we truncate labels to match.
    """
    seq = example[SEQUENCE_KEY]
    ss = example[LABEL_KEY]  # e.g. "CCCHHHEEE..."

    tokenized = tokenizer(seq, truncation=True, max_length=512)
    n_tokens = len(tokenized["input_ids"])

    # The first token is <cls> — set label to -100 (ignore).
    aligned = [-100]

    # The middle tokens (n_tokens - 2 of them, if not truncated) correspond
    # 1:1 to amino acids in the input. Map them to label IDs.
    n_residue_tokens = n_tokens - 2  # subtract <cls> and <eos>
    for i in range(n_residue_tokens):
        if i < len(ss):
            aligned.append(label_to_id.get(ss[i], -100))
        else:
            aligned.append(-100)

    # Final <eos> token — also -100.
    aligned.append(-100)

    # Defensive: pad/truncate to exactly match input_ids length.
    aligned = aligned[:n_tokens] + [-100] * (n_tokens - len(aligned))
    tokenized["labels"] = aligned
    return tokenized


def compute_metrics(eval_pred):
    """Per-residue accuracy, ignoring positions with label -100."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100  # ignore <cls>/<eos>/padding
    correct = (preds[mask] == labels[mask]).sum()
    total = mask.sum()
    return {"accuracy": float(correct) / float(total)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)
    )

    print(f"Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    print(f"\nDataset features: {raw[list(raw.keys())[0]].features}")
    print(f"Splits available: {list(raw.keys())}")
    print(
        "\nIf SEQUENCE_KEY / LABEL_KEY at the top of this file don't match the "
        "feature names above, edit them and re-run.\n"
    )

    # Pick splits — datasets vary in their split names.
    train_split = "train" if "train" in raw else list(raw.keys())[0]
    val_split = "validation" if "validation" in raw else (
        "test" if "test" in raw else train_split
    )
    raw_train = raw[train_split].select(range(min(N_TRAIN, len(raw[train_split]))))
    raw_val = raw[val_split].select(range(min(N_VAL, len(raw[val_split]))))

    print(f"Train: {len(raw_train)} sequences, val: {len(raw_val)} sequences")

    # Preprocess: tokenize + align labels.
    train_ds = raw_train.map(
        lambda ex: preprocess_example(ex, tokenizer),
        remove_columns=raw_train.column_names,
    )
    val_ds = raw_val.map(
        lambda ex: preprocess_example(ex, tokenizer),
        remove_columns=raw_val.column_names,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
    )

    # DataCollatorForTokenClassification pads BOTH input_ids AND labels.
    # The label_pad_token_id=-100 ensures padded positions are ignored by loss.
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Inference on a single sequence — show the predicted SS string.
    example = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=512).to(
        model.device
    )
    with torch.no_grad():
        logits = model(**inputs).logits[0]    # (L+2, num_labels)
    pred_ids = logits.argmax(dim=-1)[1:-1]    # drop <cls> and <eos>
    pred_ss = "".join(LABEL_NAMES[i] for i in pred_ids.tolist())
    print(f"\nSequence:           {example}")
    print(f"Predicted SS3:      {pred_ss[: len(example)]}")

    print(
        """
Things to experiment with:
- 8-state SS (Q8) instead of Q3: change LABEL_NAMES to the 8-state set and use the matching label column.
- Predict disorder regions (binary token classification — IDR vs structured).
- Predict binding sites — small datasets exist for this on Hugging Face.
- Switch model: "facebook/esm2_t12_35M_UR50D" generally improves SS accuracy noticeably.
"""
    )


if __name__ == "__main__":
    main()
