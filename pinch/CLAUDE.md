# Project: attrib-PINCH — cross-method xAI for PPI hotspots

## Shared leakage rule (applies to all three tracks)

These projects reuse off-the-shelf models, so leakage lives between each borrowed
model's training set and my eval set, not my own train/test. Before benchmarking:
verify what every borrowed model/dataset was trained on, evaluate on a disjoint
split (prefer authors' published held-out / homology-filtered splits), and report
each predictor's baseline on that same split. When unsure, stratify by
identity-to-training (25–30%, MMseqs2/Foldseek) and report seen-vs-novel bins —
the gap is a finding.

## This track specifically

**attrib-PINCH** — inherit Sendin's PINDER cluster splits; check that the SKEMPI
validation complexes are **not** in the training clusters, or the hotspot-recovery
AUROC is confounded with the classifier having memorised those complexes.
*(reasoned — verify against the actual PINDER cluster assignments before reporting.)*

Note the model here (Struct2Graph) **is** trained by this repo, unlike EDITS/CLASP
which borrow frozen predictors — so standard train/test discipline also applies to
the GCN itself, on top of the borrowed-data check above.
