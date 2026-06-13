"""Build the attrib-EDITS notebooks (edits_l1 .. edits_l5) from cell specs.

Mirrors pinch/_build_pinch_notebooks.py: thin notebooks over the tested engine in
``edits_common.py``. Run this to (re)generate the notebooks, then execute them.
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
sys.path.insert(0, os.path.join(ROOT, "edits"))   # for `import edits_common`
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


build("edits_l1_predictor.ipynb", [
    md("# attrib-EDITS · edits_l1 — the borrowed predictor\n\n"
       "**Question (project #5):** what is the smallest *realistic* edit that flips "
       "a stability/fitness predictor — and does it match real ΔΔG?\n\n"
       "This track **trains nothing**: the predictor is off-the-shelf. Here the "
       "miniature predictor is **ESM-2 used zero-shot** — one forward gives the "
       "log-probability of every residue at every position, so the model's fitness "
       "score for any sequence (and any single mutation) is a table lookup.\n\n"
       "> ⚠️ **Read [`CLAUDE.md`](CLAUDE.md) first.** Because the predictor is "
       "borrowed, leakage lives between *its* training data and *our* eval set. The "
       "real ΔΔG validation (edits_l5) must run on a disjoint split."),
    code(BOOTSTRAP),
    code("from edits_common import DEMO_SEQ, fitness, position_logp\n"
         "print('WT (GB1 domain):', DEMO_SEQ)\n"
         "print('len:', len(DEMO_SEQ), ' fitness(WT) =', round(fitness(DEMO_SEQ), 3))\n"
         "lp = position_logp(DEMO_SEQ)\n"
         "print('position_logp:', lp.shape, '(L x 20)')"),
    code("import matplotlib.pyplot as plt\n"
         "plt.figure(figsize=(10, 3.2))\n"
         "plt.imshow(lp.T, aspect='auto', cmap='viridis', origin='lower')\n"
         "plt.yticks(range(20), list('ACDEFGHIKLMNPQRSTVWY'))\n"
         "plt.xlabel('position'); plt.ylabel('amino acid'); plt.title('ESM-2 log P(aa | context)')\n"
         "plt.colorbar(label='log prob'); plt.tight_layout(); plt.show()"),
    md("### Things to experiment with\n"
       "- Swap ESM-2 for **ESM-1v** (variant-effect specialist) — still sequence-only, mild leakage.\n"
       "- The scale-up predictor is **ThermoMPNN** (Megascale-trained → severe leakage; see CLAUDE.md)."),
])

build("edits_l2_greedy_saturation.ipynb", [
    md("# attrib-EDITS · edits_l2 — greedy in-silico saturation\n\n"
       "The discrete counterfactual search. At each step we score *every* single "
       "substitution from one forward and apply the one that most reduces the "
       "predictor's fitness, until the score drops by a target margin. This is the "
       "**unconstrained (off-manifold)** search — it is allowed to pick implausible "
       "residues."),
    code(BOOTSTRAP),
    code("from edits_common import DEMO_SEQ, greedy_counterfactual, edit_metrics\n"
         "res = greedy_counterfactual(DEMO_SEQ, target_drop=8.0, plausibility_weight=0.0)\n"
         "print('metrics:', edit_metrics(res))\n"
         "for (pos, wt, mut) in res['edits']:\n"
         "    print(f'  {wt}{pos}{mut}')"),
    code("import matplotlib.pyplot as plt\n"
         "plt.figure(figsize=(6, 4))\n"
         "plt.plot(range(len(res['trajectory'])), res['trajectory'], 'o-')\n"
         "plt.xlabel('edits applied'); plt.ylabel('predictor fitness (total logL)')\n"
         "plt.title('Greedy counterfactual trajectory'); plt.tight_layout(); plt.show()"),
    md("### Things to experiment with\n"
       "- Raise `target_drop` and watch the number of edits grow.\n"
       "- Note which positions get hit — they should cluster on positions the model is most confident about."),
])

build("edits_l3_gradient_counterfactual.ipynb", [
    md("# attrib-EDITS · edits_l3 — the gradient counterfactual direction\n\n"
       "Where *should* we edit? The gradient of the predictor's score w.r.t. the "
       "input embeddings gives a per-position sensitivity — the continuous cousin of "
       "the discrete search, and the natural place to plug in **Integrated "
       "Gradients** (see [`common/ig_l1`](../common/ig_l1_simple.ipynb))."),
    code(BOOTSTRAP),
    code("import numpy as np\n"
         "from edits_common import DEMO_SEQ, gradient_saliency\n"
         "sal = gradient_saliency(DEMO_SEQ)\n"
         "top = np.argsort(sal)[::-1][:8]\n"
         "print('most sensitive positions:', [(int(i), DEMO_SEQ[i]) for i in top])"),
    code("import matplotlib.pyplot as plt\n"
         "plt.figure(figsize=(10, 3))\n"
         "plt.bar(range(len(sal)), sal)\n"
         "plt.xlabel('position'); plt.ylabel('|∂fitness/∂embedding|')\n"
         "plt.title('Gradient saliency along the sequence'); plt.tight_layout(); plt.show()"),
    md("### Things to experiment with\n"
       "- Do the high-saliency positions overlap the edits picked in edits_l2?\n"
       "- Replace the raw gradient with Captum Integrated Gradients for a baseline-anchored attribution."),
])

build("edits_l4_plausibility_constraint.ipynb", [
    md("# attrib-EDITS · edits_l4 — on-manifold vs off-manifold (the novelty)\n\n"
       "An edit that flips the model but produces an unnatural sequence is not a "
       "useful design hypothesis. We add an **ESM plausibility penalty inside the "
       "search loop** and compare: the constrained search should trade a few more "
       "edits for far more realistic substitutions."),
    code(BOOTSTRAP),
    code("from edits_common import DEMO_SEQ, greedy_counterfactual, edit_metrics\n"
         "off = greedy_counterfactual(DEMO_SEQ, target_drop=8.0, plausibility_weight=0.0)\n"
         "on  = greedy_counterfactual(DEMO_SEQ, target_drop=8.0, plausibility_weight=1.0)\n"
         "print('off-manifold:', edit_metrics(off))\n"
         "print(' edits:', [f'{w}{p}{m}' for p, w, m in off['edits']])\n"
         "print('on-manifold :', edit_metrics(on))\n"
         "print(' edits:', [f'{w}{p}{m}' for p, w, m in on['edits']])"),
    code("import matplotlib.pyplot as plt\n"
         "labels = ['off-manifold', 'on-manifold']\n"
         "nedits = [off['n_edits'], on['n_edits']]\n"
         "plaus = [off['mean_plausibility'], on['mean_plausibility']]\n"
         "fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))\n"
         "ax[0].bar(labels, nedits, color=['slategray', 'seagreen']); ax[0].set_title('#edits (proximity)')\n"
         "ax[1].bar(labels, plaus, color=['slategray', 'seagreen']); ax[1].set_title('mean plausibility (logP)')\n"
         "plt.tight_layout(); plt.show()"),
    md("### Things to experiment with\n"
       "- Sweep `plausibility_weight` and trace the proximity↔plausibility Pareto front.\n"
       "- The on-manifold constraint is what makes a counterfactual a *falsifiable* design hypothesis."),
])

build("edits_l5_validation.ipynb", [
    md("# attrib-EDITS · edits_l5 — validation against measured ΔΔG (gated)\n\n"
       "The counterfactual search above is **correctness-agnostic** — it flips the "
       "model's own boundary. The scientific claim is separate: do the proposed "
       "edits match *real* ΔΔG? That validation is intentionally **gated** because "
       "it is only meaningful on a split disjoint from the predictor's training data."),
    code(BOOTSTRAP),
    code("from edits_common import run_edits_experiment, validate_against_ddg\n"
         "print('search-quality summary (off vs on manifold):')\n"
         "for k, v in run_edits_experiment().items():\n"
         "    print(f'  {k:32s}: {v}')"),
    code("# The real ΔΔG validation is deliberately not wired — it must run on a\n"
         "# leakage-free split. Calling it explains the required protocol.\n"
         "try:\n"
         "    validate_against_ddg()\n"
         "except NotImplementedError as e:\n"
         "    print('GATED:', e)"),
    md("### The scale-up (see [`CLAUDE.md`](CLAUDE.md))\n"
       "1. Predictors: **ESM-1v** (mild leakage) **and** ThermoMPNN (severe — Megascale-trained).\n"
       "2. Eval on ThermoMPNN's published `dataset_splits/` test split **+ FireProt-HF**.\n"
       "3. Report each predictor's baseline accuracy on that disjoint split *first*.\n"
       "4. Stratify by identity-to-training (25–30%, MMseqs2/Foldseek); the seen-vs-novel gap is a finding.\n"
       "5. Then report counterfactual→ΔΔG sign/rank agreement, validity, proximity, plausibility."),
])

print("done.")
