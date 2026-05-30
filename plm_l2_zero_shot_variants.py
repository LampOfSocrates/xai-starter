"""
Lesson 2: Zero-Shot Variant Effect Prediction
==============================================
What you'll learn
-----------------
- Use a pLM's MASKED LANGUAGE MODELLING head to score mutations.
- The "wild-type marginal" trick — comparing P(mutant aa) vs P(wild-type aa).
- Why pLMs work as zero-shot variant predictors *without any training*.

The intuition
-------------
ESM-2 was trained by hiding (masking) random amino acids in real protein
sequences and asking the model to fill them back in. To do this well, it
had to learn what amino acids are "plausible" at every position, given the
surrounding context. That's evolutionary / structural intuition, baked in.

So: if a mutation is deleterious, the model will think the wild-type residue
is much more likely than the mutant residue at that position. The score is:

    score(A -> B at position i) = log P(B | sequence with i masked)
                                  - log P(A | sequence with i masked)

  - score < 0  =>  model favours wild-type, mutation is probably deleterious
  - score ~ 0  =>  mutation is neutral
  - score > 0  =>  model thinks mutant is plausible (often tolerated)

This is exactly how pLMs are used in benchmarks like ProteinGym — no training,
just clever use of the masked-LM head.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ESM-1v was specifically trained for variant effect (5 ensemble members).
# But it's 650M params. ESM-2 8M works for demo purposes.
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
# For real benchmarks try: "facebook/esm1v_t33_650M_UR90S_1"

# A short example sequence (a fragment of a real protein).
# In practice you'd load this from FASTA / UniProt.
WILD_TYPE = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# The 20 standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def score_mutations(wild_type, mutations, model, tokenizer, device):
    """Score a list of single-point mutations using masked-LM scoring.

    Args:
        wild_type: the wild-type protein sequence (string).
        mutations: list of (position, mutant_aa) tuples. Position is 0-indexed
                   within `wild_type`. mutant_aa is a single-letter amino acid.

    Returns:
        list of float log-likelihood-ratio scores, one per mutation.
    """
    # Tokenize the wild-type sequence ONCE, then mutate the token IDs in place.
    inputs = tokenizer(wild_type, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # shape (1, L+2) — has <cls> and <eos>

    scores = []
    for pos, mut_aa in mutations:
        # The tokenizer prepends a <cls> token at index 0, so amino-acid
        # position `pos` in the original sequence is at token index `pos + 1`.
        token_idx = pos + 1
        wt_aa = wild_type[pos]

        # Replace the residue at `token_idx` with the <mask> token.
        masked = input_ids.clone()
        masked[0, token_idx] = tokenizer.mask_token_id

        # Forward pass through the masked-LM head. The model returns logits
        # over the FULL vocabulary at every position.
        with torch.no_grad():
            logits = model(masked).logits  # shape (1, L+2, vocab_size)

        # Convert logits at the masked position into log-probabilities.
        log_probs = torch.log_softmax(logits[0, token_idx], dim=-1)

        # Look up the token IDs for the wild-type and mutant amino acids.
        wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)

        # Score = log P(mut) - log P(wt) at the masked position.
        score = (log_probs[mut_id] - log_probs[wt_id]).item()
        scores.append(score)

    return scores


def saturation_mutagenesis(wild_type, model, tokenizer, device, max_positions=20):
    """Compute scores for EVERY amino acid at EVERY position (up to a limit).

    Returns a (positions, 20) numpy matrix of scores.
    """
    positions = list(range(min(max_positions, len(wild_type))))
    mutations = [(p, aa) for p in positions for aa in AMINO_ACIDS]
    flat = score_mutations(wild_type, mutations, model, tokenizer, device)
    return np.array(flat).reshape(len(positions), len(AMINO_ACIDS))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pLM with the MASKED-LM head exposed (not just the encoder body).
    # AutoModelForMaskedLM gives us access to .logits over the vocab.
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()

    print(f"\nWild-type sequence (length {len(WILD_TYPE)}):")
    print(f"  {WILD_TYPE}")

    # ---- Demo 1: a few hand-picked mutations -----------------------------
    # Convention: <wt><pos+1><mut>. e.g. "M1A" = position 0 (M) -> A.
    test_mutations = [
        (0, "A"),    # M1A   - changes start codon, usually catastrophic
        (1, "L"),    # K2L   - charged -> hydrophobic, often disruptive
        (5, "A"),    # Q6A   - alanine scan: removes side-chain function
        (10, "K"),   # E11K  - charge swap (acidic -> basic)
        (10, "D"),   # E11D  - conservative (acidic -> acidic)
    ]

    print("\nScoring single mutations...")
    scores = score_mutations(WILD_TYPE, test_mutations, model, tokenizer, device)

    print(f"\n{'Mutation':<10}{'Score':>10}    Interpretation")
    print("-" * 50)
    for (pos, mut_aa), s in zip(test_mutations, scores):
        wt_aa = WILD_TYPE[pos]
        mut_str = f"{wt_aa}{pos + 1}{mut_aa}"
        if s > 0:
            verdict = "model prefers mutant (likely tolerated)"
        elif s > -2:
            verdict = "neutral / mildly disfavoured"
        else:
            verdict = "model strongly prefers wild-type (likely deleterious)"
        print(f"{mut_str:<10}{s:>+10.3f}    {verdict}")

    # ---- Demo 2: full saturation mutagenesis on a window -----------------
    # For each position, score every possible substitution. This gives you
    # a (positions, 20) heatmap — exactly what DMS experiments measure.
    print("\nRunning saturation mutagenesis on first 20 positions...")
    matrix = saturation_mutagenesis(WILD_TYPE, model, tokenizer, device, max_positions=20)

    print(f"  Score matrix shape: {matrix.shape}  (positions, amino_acids)")
    print(f"  Mean score:    {matrix.mean():+.3f}  (typically negative — mutations usually disfavoured)")
    print(f"  Min  / Max:    {matrix.min():+.3f} / {matrix.max():+.3f}")

    # Show a small ASCII preview: which AA does the model prefer at each position?
    print("\nTop preferred AA per position (vs wild-type letter):")
    for p in range(min(20, len(WILD_TYPE))):
        wt_aa = WILD_TYPE[p]
        best_idx = matrix[p].argmax()
        best_aa = AMINO_ACIDS[best_idx]
        same = " (matches WT)" if best_aa == wt_aa else ""
        print(f"  pos {p + 1:3d}  WT={wt_aa}  model_prefers={best_aa}{same}")

    print(
        """
Things to experiment with:
- Use ESM-1v (purpose-built for variant prediction):
    MODEL_NAME = "facebook/esm1v_t33_650M_UR90S_1"   # 650M, GPU recommended
- Pull a real DMS dataset from ProteinGym (https://proteingym.org/)
  and compute Spearman correlation between zero-shot scores and experimental fitness.
- Compare scoring schemes:
    "masked marginal"   - mask one position at a time (this lesson's approach)
    "wild-type marginal"- never mask, just score at each WT position
    "pseudo-likelihood" - sum of masked-marginal scores across all positions
"""
    )


if __name__ == "__main__":
    main()
