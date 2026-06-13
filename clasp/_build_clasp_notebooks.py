"""Build the contact-CLASP notebooks (clasp_l1 .. clasp_l5) from cell specs.

Mirrors pinch/_build_pinch_notebooks.py: thin notebooks that import the tested
engine in ``clasp_common.py`` so the science lives in one place. Run this to
(re)generate the notebooks, then execute them to populate outputs.
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

BOOTSTRAP = '''\
import os, sys
ROOT = os.path.abspath("")
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "common")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)                          # for `import common`
sys.path.insert(0, os.path.join(ROOT, "clasp"))   # for `import clasp_common`
DATA = os.path.join(ROOT, "data")
print("repo root:", ROOT)'''


def md(t): return ("md", t)
def code(t): return ("code", t)


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t)
                   for k, t in cells]
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                     "name": "python3"},
                      "language_info": {"name": "python"}}
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path}  ({len(nb['cells'])} cells)")


PLOT = '''\
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
ax[0].imshow(true, cmap="Greys", origin="lower"); ax[0].set_title("true Cα contacts")
im = ax[1].imshow(attn, cmap="viridis", origin="lower"); ax[1].set_title("ESM-2 attention (APC)")
for a in ax: a.set_xlabel("residue"); a.set_ylabel("residue")
plt.tight_layout(); plt.show()'''

build("clasp_l1_attention_contacts.ipynb", [
    md("# contact-CLASP · clasp_l1 — ESM-2 attention recovers contacts\n\n"
       "**Question (project #7):** does a sequence-only pLM's *attention* already "
       "concentrate on residues that touch in 3D, the way coevolution does?\n\n"
       "We take a short real chain (BPTI, `2ptc:I`), compute its true Cα contact "
       "map, and compare it to the symmetrised, APC-corrected mean of ESM-2's "
       "attention heads (the unsupervised Rao 2021 signal). Metric: **precision@L** "
       "— of the top-L predicted pairs, how many are real contacts?\n\n"
       "> The 8M model here is for speed; contact signal is far stronger in ESM-2 "
       "650M (the scale-up). See [`common/`](../common) for the shared primitives."),
    code(BOOTSTRAP),
    code("from clasp_common import load_demo_chain, esm_attention_contacts, precision_at_l\n"
         "seq, coords, true = load_demo_chain(cache_dir=DATA)\n"
         "print(f'{len(seq)} residues; {int(true[true].sum())//1} contact cells')\n"
         "attn = esm_attention_contacts(seq)\n"
         "print('precision@L (attention):', round(precision_at_l(attn, true), 3))"),
    code(PLOT),
    md("### Things to experiment with\n"
       "- Swap `facebook/esm2_t6_8M_UR50D` → `...t33_650M...` and watch precision@L jump.\n"
       "- Replace the head *average* with a logistic regression over heads (Rao 2021, supervised).\n"
       "- Try other chains from `common`'s demo complexes."),
])

build("clasp_l2_dca_baseline.ipynb", [
    md("# contact-CLASP · clasp_l2 — the coevolution baseline (mean-field DCA)\n\n"
       "The classical comparator: **direct coupling analysis**. Given a multiple-"
       "sequence alignment, co-evolving residue pairs usually touch in 3D. Here the "
       "MSA is *simulated* with covariation seeded from the true contacts (a stand-in "
       "for a real jackhmmer/hhblits alignment — that swap is the scale-up); "
       "mean-field DCA should then recover those pairs.\n\n"
       "Metric again is **precision@L** against the same true contacts as clasp_l1."),
    code(BOOTSTRAP),
    code("from clasp_common import (load_demo_chain, simulate_msa, mfdca_contacts,\n"
         "                          neff, precision_at_l)\n"
         "seq, coords, true = load_demo_chain(cache_dir=DATA)\n"
         "msa = simulate_msa(seq, true, n_seqs=500)\n"
         "dca = mfdca_contacts(msa)\n"
         "print('MSA:', msa.shape, ' Neff/L:', round(neff(msa)/len(seq), 2))\n"
         "print('precision@L (DCA):', round(precision_at_l(dca, true), 3))"),
    code("import matplotlib.pyplot as plt\n"
         "fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))\n"
         "ax[0].imshow(true, cmap='Greys', origin='lower'); ax[0].set_title('true contacts')\n"
         "ax[1].imshow(dca, cmap='magma', origin='lower'); ax[1].set_title('mean-field DCA (APC)')\n"
         "plt.tight_layout(); plt.show()"),
    md("### Things to experiment with\n"
       "- Lower `n_seqs` and `couple_p` in `simulate_msa` and watch DCA degrade.\n"
       "- Swap in a **real** MSA (jackhmmer/hhblits) — the scale-up that makes this a real benchmark."),
])

build("clasp_l3_neff_degradation.ipynb", [
    md("# contact-CLASP · clasp_l3 — the headline: precision@L vs Neff/L\n\n"
       "**The novel axis.** DCA needs many diverse sequences; attention is computed "
       "from one. So as alignment depth (**Neff/L**) drops, DCA should fall while "
       "attention stays flat. We subsample the MSA to sweep Neff/L and plot both."),
    code(BOOTSTRAP),
    code("import numpy as np\n"
         "from clasp_common import (load_demo_chain, esm_attention_contacts, simulate_msa,\n"
         "                          mfdca_contacts, neff, precision_at_l)\n"
         "seq, coords, true = load_demo_chain(cache_dir=DATA)\n"
         "L = len(seq)\n"
         "attn_p = precision_at_l(esm_attention_contacts(seq), true)\n"
         "full = simulate_msa(seq, true, n_seqs=600)\n"
         "rows = []\n"
         "for frac in (1.0, 0.5, 0.25, 0.1, 0.05):\n"
         "    sub = full[: max(5, int(600 * frac))]\n"
         "    rows.append((neff(sub) / L, precision_at_l(mfdca_contacts(sub), true)))\n"
         "rows.sort()\n"
         "nl, dca_p = zip(*rows)\n"
         "print('Neff/L :', [round(x, 2) for x in nl])\n"
         "print('DCA P@L:', [round(x, 2) for x in dca_p])\n"
         "print('attention P@L (flat):', round(attn_p, 3))"),
    code("import matplotlib.pyplot as plt\n"
         "plt.figure(figsize=(6, 4))\n"
         "plt.plot(nl, dca_p, 'o-', label='DCA')\n"
         "plt.axhline(attn_p, ls='--', c='crimson', label='ESM attention')\n"
         "plt.xscale('log'); plt.xlabel('Neff / L'); plt.ylabel('precision@L')\n"
         "plt.title('Contact recovery vs alignment depth'); plt.legend(); plt.tight_layout(); plt.show()"),
    md("### Things to experiment with\n"
       "- This is the figure the project turns on. With real MSAs, the crossover point "
       "(where attention overtakes DCA) is the headline result.\n"
       "- Bin by identity-to-pretraining too (see `CLAUDE.md`)."),
])

build("clasp_l4_ig_fusion.ipynb", [
    md("# contact-CLASP · clasp_l4 — fusing attention with a gradient/IG map\n\n"
       "If attention weakens in the low-Neff/L regime, can a second sequence-only "
       "signal help? We build a per-position masked-LM **gradient contact map** (the "
       "fast cousin of a full Integrated-Gradients map — the IG machinery is in "
       "[`common/ig_l1`](../common/ig_l1_simple.ipynb)) and **fuse** it with attention "
       "by rank-averaging."),
    code(BOOTSTRAP),
    code("from clasp_common import (load_demo_chain, esm_attention_contacts,\n"
         "                          gradient_contact_map, fuse, precision_at_l)\n"
         "seq, coords, true = load_demo_chain(cache_dir=DATA)\n"
         "attn = esm_attention_contacts(seq)\n"
         "grad = gradient_contact_map(seq)\n"
         "fused = fuse(attn, grad, method='rank')\n"
         "for name, m in [('attention', attn), ('gradient', grad), ('fusion', fused)]:\n"
         "    print(f'{name:10s} precision@L = {precision_at_l(m, true):.3f}')"),
    code("import matplotlib.pyplot as plt\n"
         "fig, ax = plt.subplots(1, 3, figsize=(12, 4))\n"
         "for a, (t, m) in zip(ax, [('attention', attn), ('gradient', grad), ('fusion', fused)]):\n"
         "    a.imshow(m, cmap='viridis', origin='lower'); a.set_title(t)\n"
         "plt.tight_layout(); plt.show()"),
    md("### Things to experiment with\n"
       "- Swap the gradient map for a true Captum **Integrated Gradients** map (`common/ig_l1`).\n"
       "- Try `fuse(..., method='zscore')`; measure fusion gain specifically in the low-Neff/L bin."),
])

build("clasp_l5_benchmark.ipynb", [
    md("# contact-CLASP · clasp_l5 — the benchmark, end to end\n\n"
       "Pulls clasp_l1–l4 together: one call computes attention / DCA / fusion "
       "precision@L across Neff/L bins. This is the miniature of the real "
       "PDB-wide benchmark; scale-up = many chains, real MSAs, ESM-2 650M, "
       "bootstrap CIs, identity stratification (`CLAUDE.md`)."),
    code(BOOTSTRAP),
    code("from clasp_common import run_clasp_experiment\n"
         "m = run_clasp_experiment(neff_fractions=(1.0, 0.3, 0.1))\n"
         "for k, v in m.items():\n"
         "    print(f'{k:24s}: {v}')"),
    md("### Honest read\n"
       "- On the tiny 8M model attention precision@L is low — expected; the *axis* "
       "(DCA falling as Neff/L drops) is the point, and it reproduces here.\n"
       "- A real verdict needs ESM-2 650M, real MSAs, and many chains with CIs.\n"
       "### Things to experiment with\n"
       "- Wrap this in an MLflow runner (mirror `pinch/run_pinch.py`) to sweep models and chains."),
])

print("done.")
