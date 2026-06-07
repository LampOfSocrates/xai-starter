"""
Lesson 7: Parameter-Efficient Fine-Tuning (LoRA)
=================================================
What you'll learn
-----------------
- Why full fine-tuning is expensive and what LoRA does instead.
- Wrap a pLM with a LoRA adapter using the `peft` library (one config object).
- Print and understand the trainable-parameter fraction (<1% of total).
- Fine-tune only the adapters and compare accuracy/F1 to Lesson 3's full fine-tune.

What LoRA does mathematically
------------------------------
A standard weight matrix W (shape d_out x d_in) is updated by Delta-W during
fine-tuning. LoRA approximates Delta-W as a product of two SMALL matrices:
    Delta-W = B @ A    where A is (r x d_in) and B is (d_out x r), r << d_in.
The forward pass becomes  W_eff x = (W + B @ A) x.
Only A and B are trained; W is frozen. For r=8 in ESM-2 8M this cuts trainable
parameters from ~8 million down to a few thousand — roughly 0.1-0.5%.

Why this matters
----------------
- Memory: frozen weights need no gradient storage.
- Speed: fewer parameters to update each step.
- Portability: you can share a tiny adapter file (~100 KB) instead of a full
  model checkpoint, and swap adapters at inference time without reloading W.
- Quality: often matches full fine-tuning at the same data size.

Runs in a few minutes on CPU with ESM-2 8M.
# pip install peft
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"   # 8M-param ESM-2; fast on CPU
DATASET_NAME = "zhanglab/DeepSol"           # binary solubility: 0=insoluble, 1=soluble
NUM_LABELS = 2
N_TRAIN = 400                               # keep small for CPU friendliness
N_TEST = 150
MAX_LEN = 256                               # truncate sequences longer than this
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-4                        # adapters learn faster than full fine-tune
LORA_R = 8                                  # rank; lower = fewer params, higher = more capacity
LORA_ALPHA = 16                             # scaling factor: effective lr scales as alpha/r
LORA_DROPOUT = 0.05
# ESM-2 attention projection names. "query" and "value" are the standard
# LoRA targets; adding "key" is one of the things to experiment with.
LORA_TARGET_MODULES = ["query", "value"]


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    """Tokenize protein sequences and cache tensors in memory."""

    def __init__(self, sequences, labels, tokenizer):
        self.encodings = tokenizer(
            sequences,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    """One pass over the training set; returns mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (accuracy, f1) over a DataLoader."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, f1


def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load tokenizer and base classification model.
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # 2. Wrap with LoRA adapters.
    # get_peft_model FREEZES all base weights and inserts A/B matrices into
    # every linear layer whose name matches LORA_TARGET_MODULES.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",            # don't train bias terms (keeps adapter tiny)
    )
    model = get_peft_model(base_model, lora_config)

    # 3. Print trainable-parameter stats — this is the key LoRA insight.
    trainable, total = count_parameters(model)
    pct = 100.0 * trainable / total
    print(f"\nTrainable parameters: {trainable:,} / {total:,}  ({pct:.2f}% of total)")
    print("  (Full fine-tuning — Lesson 3 — would train all of the above.)")
    model.print_trainable_parameters()   # peft's built-in summary

    model = model.to(device)

    # 4. Load and subset the dataset.
    print(f"\nLoading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    train_raw = raw["train"].select(range(N_TRAIN))
    test_raw = raw["test"].select(range(N_TEST))
    print(f"  Train: {len(train_raw)} sequences  |  Test: {len(test_raw)} sequences")

    # 5. Tokenize into PyTorch Datasets.
    print("Tokenizing...")
    train_ds = ProteinDataset(train_raw["sequence"], train_raw["label"], tokenizer)
    test_ds = ProteinDataset(test_raw["sequence"], test_raw["label"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Only the adapter A/B weights have requires_grad=True, so this optimizer
    # only receives those tensors.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # 6. Training loop.
    print(f"\nFine-tuning LoRA adapters for {EPOCHS} epoch(s)...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc, f1 = evaluate(model, test_loader, device)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  test_acc={acc:.3f}  test_f1={f1:.3f}")

    # 7. Final results.
    acc, f1 = evaluate(model, test_loader, device)
    print(f"\nFinal test results:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  F1       : {f1:.3f}")
    print(f"  Params trained: {trainable:,} ({pct:.2f}% of {total:,} total)")
    print(
        "  Compare: Lesson 3 (full fine-tune) trains all params for similar accuracy."
    )

    print(
        """
Things to experiment with:
- Vary rank: LORA_R = 2, 4, 16, 32 — lower rank = fewer params but less capacity.
- Increase lora_alpha relative to r (e.g. alpha=32, r=8) for a stronger scaling.
- Add more target modules: LORA_TARGET_MODULES = ["query", "key", "value", "dense"]
- Use a larger model: MODEL_NAME = "facebook/esm2_t12_35M_UR50D" (watch RAM usage).
- Merge adapters into the base weights for zero-overhead inference:
      merged = model.merge_and_unload()
- Try IA3 (even fewer params than LoRA): peft.IA3Config instead of LoraConfig.
"""
    )


if __name__ == "__main__":
    main()
