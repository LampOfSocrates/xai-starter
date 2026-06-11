# Project: contact-CLASP — attention vs coevolution, with IG fusion

## Shared leakage rule (applies to all three tracks)

These projects reuse off-the-shelf models, so leakage lives between each borrowed
model's training set and my eval set, not my own train/test. Before benchmarking:
verify what every borrowed model/dataset was trained on, evaluate on a disjoint
split (prefer authors' published held-out / homology-filtered splits), and report
each predictor's baseline on that same split. When unsure, stratify by
identity-to-training (25–30%, MMseqs2/Foldseek) and report seen-vs-novel bins —
the gap is a finding.

## This track specifically

**contact-CLASP** — ESM-2 saw UniRef50 *sequences*, not contact labels, so the
leakage is sequence-level (the eval chains may resemble pretraining sequences),
not label-level. Check eval PDB chains / MSAs against what ESM-2 was pretrained on,
and bin results by both **Neff/L** and **identity-to-pretraining**. The Neff/L
degradation curve is the headline; the identity binning guards it.
*(reasoned — verify the pretraining overlap before reporting.)*
