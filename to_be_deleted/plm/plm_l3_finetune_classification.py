"""
Lesson 3: Fine-Tune a pLM for Sequence Classification
======================================================
What you'll learn
-----------------
- Load a pLM with a CLASSIFICATION HEAD bolted on (a small MLP on top of the
  pooled sequence embedding).
- Fine-tune the entire model (pLM weights + classifier head) end-to-end.
- Use Hugging Face's Trainer API, which handles the training loop, evaluation,
  checkpointing, and best-model selection.

When to use this vs. Lesson 1 (frozen embeddings + linear probe)
----------------------------------------------------------------
- More data       => fine-tuning will beat a linear probe.
- Less data       => linear probe is more robust (the pLM doesn't overfit).
- More compute    => fine-tune.
- CPU only        => stick with Lesson 1, or use a tiny model + few epochs.

What "fine-tuning" actually does
--------------------------------
The pLM is initialised from pre-trained weights. A NEW classification head
is added on top, randomly initialised. During training, gradients flow
through both the head AND the pLM body — so the pLM weights get updated too.

You can also FREEZE the pLM body and only train the head — see the comment
at the bottom for the one-line change.

This lesson supersedes the older `plm_demo.py` in this directory; you can
delete that file once you're comfortable with this one.
"""

import os
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DATASET_NAME = "zhanglab/DeepSol"
NUM_LABELS = 2                       # binary: insoluble (0) vs soluble (1)
OUTPUT_DIR = "./results/lesson3"
N_TRAIN = 1000                        # subset size; bump up once it works
N_TEST = 200


def compute_metrics(eval_pred):
    """The Trainer calls this after each evaluation pass.

    eval_pred is a tuple (logits, labels) where:
      - logits: numpy array, shape (N, num_labels)
      - labels: numpy array, shape (N,)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load tokenizer and model.
    # AutoModelForSequenceClassification adds a small classifier head on top
    # of the pLM. The head is randomly initialised — it has not seen any data
    # yet. The pLM body is loaded from pre-trained weights.
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # 2. Load and subset the dataset.
    print(f"Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME)
    ds = DatasetDict(
        {
            "train": raw["train"].select(range(N_TRAIN)),
            "test": raw["test"].select(range(N_TEST)),
        }
    )

    # 3. Tokenize.
    # We do NOT pad here. Instead we use DataCollatorWithPadding (below),
    # which pads each batch dynamically to the longest sequence in *that*
    # batch. Much more efficient than padding everything to a global max.
    def tokenize(batch):
        return tokenizer(batch["sequence"], truncation=True, max_length=512)

    ds = ds.map(tokenize, batched=True, remove_columns=["sequence"])

    # 4. Training configuration.
    # The TrainingArguments object holds every hyperparameter — see HF docs
    # for the full list. The settings below are reasonable defaults for a
    # small classification problem.
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",          # evaluate at the end of every epoch
        save_strategy="epoch",          # save a checkpoint at the end of every epoch
        learning_rate=2e-5,             # standard for transformer fine-tuning
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,              # mild L2 regularisation
        load_best_model_at_end=True,    # restore best checkpoint after training
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,             # keep at most 2 checkpoints (saves disk)
        logging_steps=50,
        report_to="none",               # set to "wandb" or "tensorboard" for logging
    )

    # 5. Build the Trainer. This is the magic object that orchestrates
    # the entire training loop, evaluation, and checkpointing.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 6. Train!
    print("\nFine-tuning...")
    trainer.train()

    # 7. Final evaluation on the held-out test set.
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 8. Save the fine-tuned model. Loadable later with
    # AutoModelForSequenceClassification.from_pretrained(...).
    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    print(f"\nSaved fine-tuned model to: {final_dir}")

    # 9. Inference on a single sequence.
    example = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    inputs = tokenizer(
        example, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    print(f"\nExample inference on a short sequence:")
    print(f"  P(insoluble) = {probs[0]:.3f}")
    print(f"  P(soluble)   = {probs[1]:.3f}")

    print(
        """
Things to experiment with:
- Larger pLM:   MODEL_NAME = "facebook/esm2_t12_35M_UR50D"  or  "..._t30_150M_UR50D"
- Multi-class:  swap to "proteinea/localization", set NUM_LABELS = number-of-classes,
                and change average="binary" to average="weighted" in compute_metrics.
- Freeze the body, train only the head (linear-probe-ish but with end-to-end loss):
      for p in model.esm.parameters(): p.requires_grad = False
      # ...then build the Trainer.
- Parameter-efficient fine-tuning with LoRA (handle bigger models on the same GPU):
      pip install peft
      from peft import LoraConfig, get_peft_model
      model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16))
"""
    )


if __name__ == "__main__":
    main()
