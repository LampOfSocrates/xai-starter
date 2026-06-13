# `clasp/` — contact-CLASP: attention vs coevolution, with IG fusion

> **Status: scaffolded — runnable miniature.** Five notebooks run end to end on
> BPTI (`2ptc:I`) via the tested engine [`clasp_common.py`](clasp_common.py);
> [`_smoke_clasp.py`](_smoke_clasp.py) checks the whole pipeline (~8 s). Building
> blocks reused: [`plm/plm_l6_attention_contacts`](../plm/plm_l6_attention_contacts.ipynb)
> (ESM attention → contacts, Rao 2020), [`gnn/gnn_l8_contact_prediction`](../gnn/gnn_l8_contact_prediction.ipynb)
> (precision@L), and the IG primers in [`common/`](../common/).

The question (Definitive Guide project **#7 `contact-CLASP`**):

> Does ESM attention recover physical contacts the way classical coevolution
> (DCA) does, learned from sequence alone — and does that recovery **decay as
> alignment depth (Neff/L) falls**? Does fusing an IG-derived contact map with
> attention help in the low-Neff/L regime?

> **Main document (source of truth):**
> `G:\My Drive\ObsidianDB\vault\obsidian\xAI for DL in Proteins\00_xAI_for_DL_in_Proteins_Definitive_Guide.md`
> — project **#7** (Part 6). See `CLAUDE.md` here for the data-leakage constraint.

## Notebooks (run top to bottom; miniature = BPTI `2ptc:I`)

| Notebook | What it does | Builds on |
|---|---|---|
| [clasp_l1_attention_contacts](clasp_l1_attention_contacts.ipynb) | ESM-2 attention → symmetrise → APC-corrected head average (Rao 2021); precision@L vs true Cα contacts. | `plm_l6`, `gnn_l8` |
| [clasp_l2_dca_baseline](clasp_l2_dca_baseline.ipynb) | mean-field DCA on an MSA (simulated, contact-seeded — real MSA is the scale-up); precision@L — the classical comparator. | — |
| [clasp_l3_neff_degradation](clasp_l3_neff_degradation.ipynb) | subsample the MSA, plot precision@L vs Neff/L → **the headline attention-vs-DCA degradation curve** (the novel axis). | l1 + l2 |
| [clasp_l4_ig_fusion](clasp_l4_ig_fusion.ipynb) | gradient/IG-derived L×L contact map; rank-fusion with attention; Δprecision. | `common/ig_l1`, `ig_l2` |
| [clasp_l5_benchmark](clasp_l5_benchmark.ipynb) | one call: attention / DCA / fusion precision@L across Neff/L bins; honest read. | `capstone_l2` |

## Engine, smoke test & datasets

- [`clasp_common.py`](clasp_common.py) — the tested engine; imports shared
  primitives from [`common/`](../common/) (PDB I/O, contact maps, `rank01`); adds
  attention extraction, mean-field DCA, Neff/L, the gradient contact map, and the
  precision@L-vs-Neff/L experiment.
- [`_build_clasp_notebooks.py`](_build_clasp_notebooks.py) regenerates the five
  notebooks; [`_smoke_clasp.py`](_smoke_clasp.py) is the fast end-to-end check.
- Ground truth: **real PDB Cα contacts** (< 8 Å, |i−j| ≥ 6). **Scale-up:** ESM-2
  650M, real jackhmmer/hhblits MSAs, many chains with bootstrap CIs, identity
  stratification (see `CLAUDE.md`).
