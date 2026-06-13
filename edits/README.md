# `edits/` — attrib-EDITS: minimal realistic counterfactual edits vs ΔΔG

> **Status: scaffolded — runnable miniature.** Five notebooks run end to end on a
> GB1 domain via the tested engine [`edits_common.py`](edits_common.py);
> [`_smoke_edits.py`](_smoke_edits.py) checks the pipeline (~4 s). The miniature
> predictor is **ESM-2 used zero-shot**; the real ΔΔG validation (edits_l5) is
> gated behind the leakage protocol in [`CLAUDE.md`](CLAUDE.md).

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

## Notebooks (run top to bottom; miniature = GB1 domain, ESM-2 zero-shot)

| Notebook | What it does |
|---|---|
| [edits_l1_predictor](edits_l1_predictor.ipynb) | The borrowed predictor — ESM-2 zero-shot fitness (one forward → log P of every substitution); fitness(WT) + a position×AA log-prob heatmap. **Read CLAUDE.md first.** |
| [edits_l2_greedy_saturation](edits_l2_greedy_saturation.ipynb) | Greedy in-silico saturation: score every single substitution from one forward, apply the most fitness-reducing, iterate — the discrete (off-manifold) search + trajectory. |
| [edits_l3_gradient_counterfactual](edits_l3_gradient_counterfactual.ipynb) | Gradient of fitness w.r.t. embeddings → per-position "where to edit" saliency (the IG hook). |
| [edits_l4_plausibility_constraint](edits_l4_plausibility_constraint.ipynb) | Add the ESM plausibility penalty **inside** the search loop; on-manifold vs off-manifold proximity↔plausibility trade-off (the novelty). |
| [edits_l5_validation](edits_l5_validation.ipynb) | Search-quality summary; the real ΔΔG validation is **gated** — the notebook prints the required leakage-free protocol. |

## Engine, smoke test & datasets

- [`edits_common.py`](edits_common.py) — the tested engine; imports `common/`;
  adds the ESM-2 zero-shot predictor, the greedy counterfactual with an
  on-manifold plausibility penalty, gradient saliency, and a `validate_against_ddg`
  that **raises** until wired on a leakage-free split.
- [`_build_edits_notebooks.py`](_build_edits_notebooks.py) regenerates the five
  notebooks; [`_smoke_edits.py`](_smoke_edits.py) is the fast end-to-end check.
- **Scale-up predictors:** **ESM-1v** (sequence-only, mild leakage) and
  **ThermoMPNN** (Megascale-trained, severe). **Ground truth:** Megascale
  (~776k ΔΔG) + FireProtDB via ThermoMPNN's `dataset_splits/` + FireProt-HF. Full
  protocol in [`CLAUDE.md`](CLAUDE.md).
