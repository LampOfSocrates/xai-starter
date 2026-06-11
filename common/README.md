# `common/` — shared code + Integrated Gradients foundations

The code every research track reuses, plus the two IG primer notebooks that are
prerequisites for all of them. The three tracks — [`pinch/`](../pinch/)
(attrib-PINCH), [`clasp/`](../clasp/) (contact-CLASP), [`edits/`](../edits/)
(attrib-EDITS) — import this package rather than copying primitives.

> **Main document (source of truth for the science):**
> `G:\My Drive\ObsidianDB\vault\obsidian\xAI for DL in Proteins\00_xAI_for_DL_in_Proteins_Definitive_Guide.md`
> — Part 2 covers Integrated Gradients (the two notebooks here); Part 6 lists the
> 17 project ideas the three tracks are drawn from.

## Code modules

| Module | Provides | Used by |
|---|---|---|
| [`config.py`](config.py) | `AMINO_ACIDS`, `AA_TO_IDX`, `THREE_TO_ONE`, `get_device()` | all tracks |
| [`structures.py`](structures.py) | `fetch_pdb()`, `parse_chain()` — PDB download + Cα parse | PINCH (graphs), CLASP (contacts), EDITS (structure predictors) |
| [`graphs.py`](graphs.py) | `one_hot_aa()`, `contact_graph()`, `interface_mask()` | PINCH, CLASP |
| [`metrics.py`](metrics.py) | `rank01()` — rank-normalise to [0,1] before fusing signals | PINCH (consensus), CLASP (attention–DCA fusion) |
| [`__init__.py`](__init__.py) | re-exports the above as a flat API | — |

Track-specific machinery (the Struct2Graph model, SKEMPI loaders, DCA, the
counterfactual search) lives in each track's own `<track>_common.py`, not here.

## Foundation notebooks

| Notebook | What it teaches |
|---|---|
| [ig_l1_simple](ig_l1_simple.ipynb) | IG from scratch on a tiny known function: the path integral of gradients, the completeness axiom, why IG beats a plain gradient. |
| [ig_l2_tiny_network](ig_l2_tiny_network.ipynb) | IG on a small trained network. |

## Importing `common` from a track notebook

Each track notebook puts the repo root on `sys.path` first, then imports:

```python
import os, sys
ROOT = os.path.abspath("")
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "common")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from common import fetch_pdb, parse_chain, contact_graph, get_device, rank01
```
