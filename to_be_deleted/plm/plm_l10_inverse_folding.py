"""
Lesson 10: Generative Protein Design with Masked Language Models
================================================================
What you'll learn
-----------------
- pLMs are GENERATIVE: the masked-LM head can propose new residues, not
  just score existing ones.
- Implement Gibbs-style iterative redesign: repeatedly mask and resample a
  subset of positions to evolve a sequence while keeping it plausible.
- Measure NATIVE SEQUENCE RECOVERY — the fraction of positions where
  one-shot argmax prediction matches the wild-type residue.
- Understand how TEMPERATURE controls diversity vs. fidelity in sampling.

The generative counterpart to discriminative scoring
----------------------------------------------------
Lessons 1-5 used pLMs DISCRIMINATIVELY — embed or score an existing sequence.
This lesson flips the direction: we use the masked-LM head to GENERATE new
sequences. Each forward pass returns a probability distribution over the 20
amino acids at every masked position; sampling from that distribution gives
us new residue choices.

True inverse folding (e.g. ProteinMPNN, ESM-IF) conditions sequence design
on a 3D backbone structure. This lesson is the SEQUENCE-ONLY analogue — no
structural input — but the same sampling machinery applies. See the "Things
to experiment with" section at the end for structure-conditioned pointers.

Gibbs-style design
------------------
Gibbs sampling over sequences works as follows:
  1. Start with a wild-type (or random) sequence.
  2. Randomly mask a MASK_FRACTION of positions.
  3. Run one forward pass; get logit distributions at each masked position.
  4. Optionally divide logits by TEMPERATURE before softmax (lower T =
     greedier, higher T = more diverse).
  5. Sample a new residue from the resulting distribution at each masked
     position and accept it unconditionally.
  6. Repeat for N_ITERATIONS. Track pseudo-log-likelihood (PLL) to confirm
     the sequence stays plausible.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # 8M params, fast on CPU

# Same wild-type as Lesson 2 for easy cross-lesson comparison.
WILD_TYPE = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"   # 20 standard AAs; random baseline = 1/20 = 5%

# Gibbs redesign settings
N_ITERATIONS = 20            # number of Gibbs sweeps
MASK_FRACTION = 0.15         # fraction of positions to mask per iteration
TEMPERATURE_GREEDY = 0.01    # near-zero -> argmax (greedy / deterministic)
TEMPERATURE_SAMPLE = 1.0     # T=1 -> true model distribution

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def compute_pll(sequence, model, tokenizer, device):
    """Compute pseudo-log-likelihood (PLL) for a sequence.

    Masks one position at a time and sums log P(true residue | context).
    Higher PLL means the model considers the sequence more plausible.
    This is the same per-position masked-marginal used in Lesson 2.
    """
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # (1, L+2) with <cls> and <eos>

    total_log_prob = 0.0
    for pos in range(len(sequence)):
        token_idx = pos + 1  # offset by 1 for <cls>
        true_aa = sequence[pos]
        true_id = tokenizer.convert_tokens_to_ids(true_aa)

        masked = input_ids.clone()
        masked[0, token_idx] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked).logits   # (1, L+2, vocab_size)

        log_probs = F.log_softmax(logits[0, token_idx], dim=-1)
        total_log_prob += log_probs[true_id].item()

    return total_log_prob


def sample_residue(logits_1d, temperature):
    """Sample one residue token from a logit vector with temperature scaling.

    Args:
        logits_1d: 1-D tensor of vocab logits at one position.
        temperature: float. Low values sharpen the distribution (greedy),
                     high values flatten it (more random).

    Returns:
        sampled token index (int).
    """
    scaled = logits_1d / max(temperature, 1e-6)   # avoid division by zero
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ---------------------------------------------------------------------------
# Design loop
# ---------------------------------------------------------------------------

def gibbs_redesign(wild_type, model, tokenizer, device,
                   n_iterations, mask_fraction, temperature):
    """Iterative Gibbs-style sequence redesign.

    Starts from `wild_type`, then at each iteration masks a random subset
    of positions and replaces them by sampling from the model's distribution.

    Returns:
        designed_sequence: str  — the final designed sequence.
        pll_history: list of float — PLL after each iteration.
    """
    # Work with a mutable list of amino acid characters.
    current_seq = list(wild_type)
    seq_len = len(wild_type)
    n_mask = max(1, int(seq_len * mask_fraction))

    # Evaluate PLL on the starting sequence.
    pll_history = [compute_pll(wild_type, model, tokenizer, device)]

    inputs = tokenizer(wild_type, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"].clone()  # (1, L+2)

    for iteration in range(n_iterations):
        # Choose positions to mask this sweep.
        mask_positions = random.sample(range(seq_len), n_mask)

        # Build masked input from the CURRENT sequence state.
        current_str = "".join(current_seq)
        enc = tokenizer(current_str, return_tensors="pt").to(device)
        ids = enc["input_ids"].clone()

        for pos in mask_positions:
            ids[0, pos + 1] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(ids).logits   # (1, L+2, vocab_size)

        # Sample a new residue at each masked position.
        for pos in mask_positions:
            new_token_id = sample_residue(logits[0, pos + 1], temperature)
            new_aa = tokenizer.convert_ids_to_tokens(new_token_id)
            # Only accept standard amino acids (skip special tokens).
            if new_aa in AMINO_ACIDS:
                current_seq[pos] = new_aa
            # If the model proposes a non-AA token (rare), keep current residue.

        current_str = "".join(current_seq)
        pll = compute_pll(current_str, model, tokenizer, device)
        pll_history.append(pll)

        print(f"  iter {iteration + 1:3d}/{n_iterations}  PLL={pll:.1f}  "
              f"seq: {current_str[:30]}...")

    return "".join(current_seq), pll_history


# ---------------------------------------------------------------------------
# Native sequence recovery
# ---------------------------------------------------------------------------

def native_sequence_recovery(wild_type, model, tokenizer, device):
    """Compute native sequence recovery: fraction of positions where the
    model's argmax (greedy) prediction matches the original residue.

    Each position is masked one at a time and the top-1 prediction is taken.
    This is a standard metric in inverse-folding papers (ProteinMPNN etc.),
    though here it is SEQUENCE-ONLY — no 3D backbone is provided.

    Random baseline: 1/20 = 5% (uniform over 20 standard amino acids).
    """
    inputs = tokenizer(wild_type, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    recovered = 0
    predictions = []

    for pos in range(len(wild_type)):
        token_idx = pos + 1
        true_aa = wild_type[pos]

        masked = input_ids.clone()
        masked[0, token_idx] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked).logits

        pred_token_id = logits[0, token_idx].argmax().item()
        pred_aa = tokenizer.convert_ids_to_tokens(pred_token_id)
        predictions.append(pred_aa)

        if pred_aa == true_aa:
            recovered += 1

    recovery_pct = 100.0 * recovered / len(wild_type)
    return recovery_pct, predictions


# ---------------------------------------------------------------------------
# Per-residue identity between two sequences
# ---------------------------------------------------------------------------

def sequence_identity(seq_a, seq_b):
    """Fraction of positions where seq_a and seq_b share the same residue."""
    assert len(seq_a) == len(seq_b), "sequences must have equal length"
    matches = sum(a == b for a, b in zip(seq_a, seq_b))
    return 100.0 * matches / len(seq_a)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()

    print(f"\nWild-type sequence (length {len(WILD_TYPE)}):")
    print(f"  {WILD_TYPE}")

    # ---- 1. Native sequence recovery ------------------------------------
    # Mask one position at a time and take argmax — how often does the model
    # guess back the original residue?
    print("\n--- Native Sequence Recovery ---")
    print("(masking each position individually, taking argmax prediction)")
    recovery_pct, predictions = native_sequence_recovery(
        WILD_TYPE, model, tokenizer, device
    )
    print(f"\n  Wild-type:   {WILD_TYPE}")
    print(f"  Predictions: {''.join(predictions)}")
    print(f"\n  Recovery: {recovery_pct:.1f}%   (random baseline: 5.0%)")
    print(f"  {'Above' if recovery_pct > 5.0 else 'At or below'} random — model exploits sequence context.")

    # ---- 2. Greedy redesign (low temperature) ---------------------------
    print(f"\n--- Gibbs Redesign: GREEDY  (temperature={TEMPERATURE_GREEDY}) ---")
    print("Low temperature concentrates probability mass on the top residue.")
    greedy_seq, greedy_pll = gibbs_redesign(
        WILD_TYPE, model, tokenizer, device,
        n_iterations=N_ITERATIONS,
        mask_fraction=MASK_FRACTION,
        temperature=TEMPERATURE_GREEDY,
    )
    greedy_id = sequence_identity(WILD_TYPE, greedy_seq)
    print(f"\n  Final greedy design:")
    print(f"    {greedy_seq}")
    print(f"  Identity to wild-type: {greedy_id:.1f}%")
    print(f"  PLL start: {greedy_pll[0]:.1f}  ->  end: {greedy_pll[-1]:.1f}")

    # ---- 3. Sampled redesign (temperature = 1) --------------------------
    print(f"\n--- Gibbs Redesign: SAMPLED (temperature={TEMPERATURE_SAMPLE}) ---")
    print("Temperature=1 samples the full model distribution — more diverse sequences.")
    sampled_seq, sampled_pll = gibbs_redesign(
        WILD_TYPE, model, tokenizer, device,
        n_iterations=N_ITERATIONS,
        mask_fraction=MASK_FRACTION,
        temperature=TEMPERATURE_SAMPLE,
    )
    sampled_id = sequence_identity(WILD_TYPE, sampled_seq)
    print(f"\n  Final sampled design:")
    print(f"    {sampled_seq}")
    print(f"  Identity to wild-type: {sampled_id:.1f}%")
    print(f"  PLL start: {sampled_pll[0]:.1f}  ->  end: {sampled_pll[-1]:.1f}")

    # ---- 4. Summary comparison ------------------------------------------
    print("\n--- Summary ---")
    print(f"  {'Sequence':<22}  {'Identity to WT':>16}  {'Final PLL':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Wild-type':<22}  {'100.0%':>16}  {greedy_pll[0]:>12.1f}")
    print(f"  {'Greedy design':<22}  {greedy_id:>15.1f}%  {greedy_pll[-1]:>12.1f}")
    print(f"  {'Sampled design':<22}  {sampled_id:>15.1f}%  {sampled_pll[-1]:>12.1f}")
    print(f"\n  Note: TRUE inverse folding (ProteinMPNN, ESM-IF) conditions")
    print(f"  sequence design on a 3D BACKBONE — you get 'design the sequence")
    print(f"  for *this* structure'. Here we have no structural input, so the")
    print(f"  model generalises from sequence context alone.")

    print(
        """
Things to experiment with:
- Real inverse folding: ProteinMPNN (github.com/dauparas/ProteinMPNN) or
  ESM-IF (facebook/esm_if1_gvp4_t16_142M_UR50) condition on a 3D backbone
  — much higher recovery because the structure constrains the problem.
- Constrain a motif: fix active-site residues (skip them in the mask) and
  redesign only the surrounding scaffold.
- Vary TEMPERATURE (try 0.5, 1.5, 2.0) and MASK_FRACTION (0.3, 0.5) to
  explore the diversity vs. plausibility trade-off in the PLL trajectory.
- Score your designs with the Lesson 2 variant scorer to see whether the
  model considers them better or worse than wild-type at each position.
- Filter designs with the Lesson 1 solubility probe: embed each designed
  sequence, run the trained classifier, and keep only predicted-soluble ones.
- Measure pairwise diversity across an ensemble of sampled designs (average
  Hamming distance) to characterise the design landscape explored.
"""
    )


if __name__ == "__main__":
    main()
