r"""End-to-end smoke test of clasp_common.py - validates the full contact-CLASP
pipeline before the clasp_l1..clasp_l5 notebooks are written around it.

Run with the canonical venv:
    .\.venv\Scripts\python.exe clasp\_smoke_clasp.py
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for `import common`
sys.path.insert(0, HERE)                     # this dir, for `import clasp_common`
import numpy as np

from clasp_common import (
    load_demo_chain, esm_attention_contacts, mfdca_contacts, gradient_contact_map,
    simulate_msa, neff, fuse, precision_at_l, run_clasp_experiment,
)

t0 = time.time()

# 1) real ground truth on the miniature chain (BPTI)
seq, coords, true = load_demo_chain()
L = len(seq)
print(f"[1] {len(seq)} res, {int(np.triu(true, 6).sum())} long-range contacts")

# 2) ESM-2 attention contacts
attn = esm_attention_contacts(seq)
print(f"[2] attention map {attn.shape}, precision@L = {precision_at_l(attn, true):.3f}")

# 3) simulated MSA + mean-field DCA, full vs subsampled (the Neff/L axis)
msa = simulate_msa(seq, true, n_seqs=400)
dca_full = mfdca_contacts(msa)
dca_thin = mfdca_contacts(msa[:60])
print(f"[3] DCA  full: Neff/L={neff(msa)/L:.2f} P@L={precision_at_l(dca_full, true):.3f}"
      f"  | thin: Neff/L={neff(msa[:60])/L:.2f} P@L={precision_at_l(dca_thin, true):.3f}")

# 4) gradient contact map + fusion with attention
grad = gradient_contact_map(seq)
fused = fuse(attn, grad)
print(f"[4] gradient P@L={precision_at_l(grad, true):.3f}  fusion P@L={precision_at_l(fused, true):.3f}")

# 5) the headline experiment
print("[5] run_clasp_experiment:")
for k, v in run_clasp_experiment(neff_fractions=(1.0, 0.3, 0.1)).items():
    print(f"      {k:24s}: {v}")

print(f"\nOK - full pipeline ran in {time.time()-t0:.1f}s")
