#!/usr/bin/env python3
"""
Protein Language Model Demo
This script demonstrates fine-tuning and testing a protein language model (ESM-2)
on a protein solubility prediction task using the DeepSol dataset.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_data():
    """Load the DeepSol dataset for protein solubility prediction."""
    print("Loading DeepSol dataset...")
    dataset = load_dataset("zhanglab/DeepSol")

    # The dataset has 'train' and 'test' splits
    # For demo, we'll use a small subset
    small_train = dataset['train'].select(range(1000))  # First 1000 samples
    small_test = dataset['test'].select(range(200))     # First 200 test samples

    return DatasetDict({
        'train': small_train,
        'test': small_test
    })

def tokenize_function(examples, tokenizer):
    """Tokenize protein sequences."""
    return tokenizer(
        examples["sequence"],
        truncation=True,
        padding=False,  # We'll use data collator for dynamic padding
        max_length=512
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fine_tune_model(model_name="facebook/esm2_t6_8M_UR50D", output_dir="./results"):
    """Fine-tune the pLM on the solubility prediction task."""

    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification: soluble/insoluble
        problem_type="single_label_classification"
    )

    # Load and prepare data
    dataset = load_data()

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["sequence"]  # Remove original sequence column
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting fine-tuning...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f"{output_dir}/fine_tuned_model")
    print(f"Model saved to {output_dir}/fine_tuned_model")

    return trainer

def evaluate_model(trainer, dataset):
    """Evaluate the trained model on test data."""
    print("Evaluating model...")
    results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    return results

def predict_single_sequence(model_path, sequence, tokenizer_name="facebook/esm2_t6_8M_UR50D"):
    """Make prediction on a single protein sequence."""
    print(f"Predicting solubility for sequence: {sequence[:50]}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize sequence
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()

    # Interpret results
    class_names = ["Insoluble", "Soluble"]
    result = {
        "sequence": sequence,
        "prediction": class_names[predicted_class],
        "confidence": confidence,
        "probabilities": predictions[0].tolist()
    }

    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
    return result

def main():
    """Main function to run the demo."""
    print("Protein Language Model Demo")
    print("=" * 40)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Fine-tune model
    trainer = fine_tune_model()

    # Evaluate model
    dataset = load_data()
    evaluate_model(trainer, dataset["test"])

    # Example inference
    print("\nExample Inference:")
    example_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"  # Example protein sequence
    predict_single_sequence("./results/fine_tuned_model", example_sequence)

if __name__ == "__main__":
    main()