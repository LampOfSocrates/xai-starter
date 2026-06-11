# `clasp/` — contact-CLASP: attention vs coevolution, with IG fusion

> **Status: planned.** Folder scaffolds the track; notebooks are to be built.
> Building blocks already exist in the repo: [`plm/plm_l6_attention_contacts`](../plm/plm_l6_attention_contacts.ipynb)
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

## Planned notebooks

| Notebook | What it will do | Builds on |
|---|---|---|
| `clasp_l1_attention_contacts` | ESM-2 attention → symmetrise → average informative heads / logistic-regression over heads (Rao 2021); precision@L on a few PDB chains. | `plm_l6`, `gnn_l8` |
| `clasp_l2_dca_baseline` | plmDCA / pseudolikelihood DCA on the same MSAs; precision@L — the classical comparator. | — |
| `clasp_l3_neff_degradation` | Neff/L per MSA, bin precision@L by depth → **the headline attention-vs-DCA degradation curve** (the novel axis). | l1 + l2 |
| `clasp_l4_ig_fusion` | IG-derived L×L contact map; rank-fusion / z-score average with attention; Δprecision in the low-Neff/L regime. | `common/ig_l1`, `ig_l2` |
| `clasp_l5_benchmark` | aggregate precision@L-vs-Neff/L across a PDB set with bootstrap CIs; IG-vs-attention agreement map. | `capstone_l2` (honest eval) |

## Engine & datasets (planned)

- `clasp_common.py` — track engine; imports shared primitives from
  [`common/`](../common/) (PDB I/O, contact maps, `rank01`); adds MSA/Neff/L +
  DCA helpers (`msa.py`-style) and attention extraction.
- Ground truth: **PDB Cβ–Cβ contacts** (< 8 Å, |i−j| ≥ 6). MSAs for Neff/L and
  DCA. Lightest compute and cleanest ground truth of the three tracks.
