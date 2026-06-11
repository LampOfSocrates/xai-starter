# `edits/` — attrib-EDITS: minimal realistic counterfactual edits vs ΔΔG

> **Status: planned.** Folder scaffolds the track; notebooks are to be built.
> Reuses the IG / gradient machinery from the [`common/`](../common/) primers.

The question (Definitive Guide project **#5 `attrib-EDITS`**):

> What is the smallest, still-realistic edit that flips a stability/fitness
> predictor — and do those edits match real ΔΔG? On-manifold (plausibility-
> constrained) counterfactuals, validated against measured ΔΔG on a split
> **disjoint from the predictor's training data**.

> **Main document (source of truth):**
> `G:\My Drive\ObsidianDB\vault\obsidian\xAI for DL in Proteins\00_xAI_for_DL_in_Proteins_Definitive_Guide.md`
> — project **#5** (Part 6).
> **Read [`CLAUDE.md`](CLAUDE.md) before building** — this track's headline result
> is only valid on a leakage-free split, and the constraints there are load-bearing.

## Planned notebooks

| Notebook | What it will do |
|---|---|
| `edits_l1_predictor` | Wrap the predictor to attack — ESM-1v fitness (masked-marginals) or a ThermoMPNN-style ΔΔG head; sanity-check predictions, define the "flip" target, **report the predictor's baseline on the disjoint eval split** (see CLAUDE.md). |
| `edits_l2_greedy_saturation` | Greedy in-silico saturation: try all 19 substitutions per position, take the largest-moving edit, iterate — the discrete search baseline. |
| `edits_l3_gradient_counterfactual` | Gradient/IG counterfactual in embedding space, projected back to the nearest residue — the continuous search. |
| `edits_l4_plausibility_constraint` | Add the ESM pseudo-likelihood plausibility penalty **inside** the search loop; compare on-manifold vs off-manifold edits (the novelty). |
| `edits_l5_validation` | Validate on Megascale / FireProtDB on a split disjoint from the predictor's training; report validity / proximity (#edits) / plausibility / sign-rank agreement + predicted-edit-vs-measured-ΔΔG scatter, stratified by identity-to-training. |

## Engine & datasets (planned)

- `edits_common.py` — track engine; imports `common/` (PDB I/O, ESM scoring,
  IG/gradient helpers); adds the counterfactual search + on-manifold penalty.
- Predictors: **ESM-1v** (sequence-only, mild leakage) and **ThermoMPNN**
  (Megascale-trained, severe leakage) — run both; agreement on disjoint data is
  the robust headline.
- Ground truth: **Megascale** (Tsuboyama 2023, ~776k ΔΔG) and **FireProtDB**,
  using ThermoMPNN's published `dataset_splits/` + FireProt-HF so disjointness is
  inherited. Details and the full operating rules are in [`CLAUDE.md`](CLAUDE.md).
