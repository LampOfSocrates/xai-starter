# Project: attrib-EDITS — counterfactual edits validated against ΔΔG

## Standing constraint — data leakage via the borrowed predictor

This project trains nothing; predictors are off-the-shelf. The leakage risk is
therefore **not** between my train/test but between the **predictor's training
set and my evaluation set**. Any ΔΔG-agreement result must be reported on a split
disjoint from what the predictor was trained on, or the counterfactual-recovery
claim is confounded with memorisation.

Always establish and report the predictor's baseline accuracy on that same
disjoint split **before** interpreting counterfactual–ΔΔG agreement, so the
headline reads: *"given a predictor of accuracy X on unseen proteins,
counterfactual edits recover real ΔΔG-signed moves at rate Y."*

## Verified facts (confirmed via sources, June 2026)

- **ThermoMPNN** is trained directly on the **Megascale (Tsuboyama 2023)** training
  split (~272k single-substitution mutants, 298 proteins, 8:1:1). Its default
  weights are the Megascale-trained model; ProteinMPNN is a frozen feature
  extractor. → **ThermoMPNN × Megascale-as-ground-truth = severe label leakage.**
- **ESM-1v** is trained only on sequences (UniRef90, 2020-03, ~98M seqs), with no
  supervision from experimental data. It never saw ΔΔG labels. → **ESM-1v × any
  ΔΔG set = mild, sequence-level overlap only.**
- **Leakage ranking:** ThermoMPNN×Megascale (severe) > ThermoMPNN×FireProtDB
  (moderate, homology) > ESM-1v×Megascale ≈ ESM-1v×FireProtDB (mild).

## Operating rules

1. **Reuse ThermoMPNN's published splits, don't hand-roll.** The
   `Kuhlman-Lab/ThermoMPNN` repo ships the exact train/val/test `.pkl` files in
   `dataset_splits/`, plus a homologue-free (HF) FireProt split (N≈2,578).
   Evaluate on the held-out test split + FireProt-HF so disjointness is inherited
   and trivially defensible.
2. **Run both predictors (ESM-1v and ThermoMPNN).** They sit at opposite ends of
   the leakage spectrum; agreement holding for *both* on disjoint data is the
   robust headline.
3. **Stratify by sequence-identity-to-training** (likely-seen vs likely-novel) and
   report both bins. The gap quantifies memorisation directly — treat it as a
   result, not just a safeguard. Use a 25–30% identity threshold
   (MMseqs2/Foldseek clustering; SPURS 2026 used >25%).
4. **Keep the two framings separate in the writeup.** The counterfactual search is
   correctness-agnostic (it flips the model's own boundary); the ΔΔG validation is
   conditional on a decent predictor.
5. **Model-vs-reality disagreements** (edit flips the model but contradicts
   measured ΔΔG) localise where the predictor is unfaithful — report as a feature,
   not only noise.

## Open item before building

Inspect what `dataset_splits/` actually contains (column format, multi-mutant
handling) to know exactly what is being inherited.

---

## Shared leakage rule (applies to all three tracks)

These projects reuse off-the-shelf models, so leakage lives between each borrowed
model's training set and my eval set, not my own train/test. Before benchmarking:
verify what every borrowed model/dataset was trained on, evaluate on a disjoint
split (prefer authors' published held-out / homology-filtered splits), and report
each predictor's baseline on that same split. When unsure, stratify by
identity-to-training (25–30%, MMseqs2/Foldseek) and report seen-vs-novel bins —
the gap is a finding.

- **attrib-EDITS** — ThermoMPNN trained on Megascale ΔΔG (severe); ESM-1v
  sequence-only/UniRef90 (mild). Reuse ThermoMPNN `dataset_splits/` + FireProt-HF.
