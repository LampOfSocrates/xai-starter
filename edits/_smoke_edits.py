r"""End-to-end smoke test of edits_common.py - validates the full attrib-EDITS
search pipeline before the edits_l1..edits_l5 notebooks are written around it.

Run with the canonical venv:
    .\.venv\Scripts\python.exe edits\_smoke_edits.py
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for `import common`
sys.path.insert(0, HERE)                     # this dir, for `import edits_common`

from edits_common import (
    DEMO_SEQ, fitness, position_logp, greedy_counterfactual, gradient_saliency,
    edit_metrics, run_edits_experiment, validate_against_ddg,
)

t0 = time.time()

# 1) the borrowed predictor (ESM-2 zero-shot fitness proxy)
print(f"[1] WT {len(DEMO_SEQ)} res, fitness(WT) = {fitness(DEMO_SEQ):.4f}")
print(f"    position_logp shape = {position_logp(DEMO_SEQ).shape}")

# 2) gradient saliency: where is the model most sensitive?
sal = gradient_saliency(DEMO_SEQ)
top = sal.argsort()[::-1][:5]
print(f"[2] gradient saliency top-5 positions: {top.tolist()}")

# 3) unconstrained (off-manifold) counterfactual
off = greedy_counterfactual(DEMO_SEQ, target_drop=8.0, plausibility_weight=0.0)
print(f"[3] off-manifold: {edit_metrics(off)}")
print(f"    edits: {off['edits']}")

# 4) on-manifold counterfactual (plausibility-penalised)
on = greedy_counterfactual(DEMO_SEQ, target_drop=8.0, plausibility_weight=1.0)
print(f"[4] on-manifold:  {edit_metrics(on)}")
print(f"    edits: {on['edits']}")

# 5) the comparison experiment + the gated ΔΔG validation
print(f"[5] experiment: {run_edits_experiment()}")
try:
    validate_against_ddg()
except NotImplementedError:
    print("    validate_against_ddg() correctly gated (see edits/CLAUDE.md)")

print(f"\nOK - full pipeline ran in {time.time()-t0:.1f}s")
