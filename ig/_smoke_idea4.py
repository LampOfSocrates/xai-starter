"""End-to-end smoke test of idea4_common.py - validates the full Idea 4 pipeline
before the ig_l3..ig_l7 notebooks are written around it.

Run with the canonical venv:
    .\.venv\Scripts\python.exe ig\_smoke_idea4.py
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from idea4_common import (
    get_device, fetch_pdb, parse_chain, contact_graph, interface_mask,
    build_demo_dataset, train_struct2graph, load_or_train_demo,
    compute_all_saliencies, fetch_skempi, skempi_ddg, map_ddg_to_nodes, rank01,
)

t0 = time.time()
print(f"device: {get_device()}")

# 1) data primitives on one complex (barnase/barstar)
path = fetch_pdb("1brs")
sa, xa, rna = parse_chain(path, "A")
sb, xb, rnb = parse_chain(path, "D")
print(f"[1] 1brs A: {len(sa)} res, D: {len(sb)} res")
ga, gb = contact_graph(sa, xa), contact_graph(sb, xb)
print(f"    graph A: {ga.num_nodes} nodes / {ga.num_edges} edges")
mka, mkb = interface_mask(xa, xb)
print(f"    interface residues: A={int(mka.sum())} D={int(mkb.sum())}")

# 2) tiny dataset + train
ds = build_demo_dataset(augment=3)
print(f"[2] demo dataset: {len(ds)} samples, {len({s['label'] for s in ds})} classes")
model = train_struct2graph(ds, epochs=40, verbose=True)

# 3) all three xAI saliencies on the barnase/barstar complex
s0 = next(s for s in ds if not s["augmented"])  # pristine 1brs
sal = compute_all_saliencies(model, s0["ga"], s0["gb"])
for chain in ("a", "d"):
    keys = ("attention", "ig", "gnnexplainer", "consensus")
    shapes = {k: np.asarray(sal[chain][k]).shape for k in keys}
    print(f"[3] chain {chain}: {shapes}")
print(f"    target class={sal['target']}  IG completeness delta={sal['ig_delta']:.2e}")

# 4) SKEMPI ground truth + map onto nodes
df = fetch_skempi()
print(f"[4] SKEMPI rows: {len(df)}")
ddg = skempi_ddg(df, "1brs")
print(f"    1brs single-point alanine ddG rows: {len(ddg)}")
ddg_node, hotspot, n_meas = map_ddg_to_nodes(ddg, "A", rna)
print(f"    chain A: {n_meas} measured, {int(hotspot.sum())} hotspots (ddG>=2)")

# 5) the actual hypothesis check: does consensus beat single methods on hotspots?
if hotspot.sum() > 0 and n_meas > 0:
    from sklearn.metrics import roc_auc_score
    measured = np.isfinite(ddg_node)
    y = hotspot[measured].astype(int)
    if y.sum() > 0 and y.sum() < y.size:
        print("[5] hotspot-recovery AUROC (chain A, measured residues only):")
        for k in ("attention", "ig", "gnnexplainer", "consensus"):
            auc = roc_auc_score(y, np.asarray(sal["a"][k])[measured])
            print(f"      {k:13s}: {auc:.3f}")
    else:
        print("[5] not enough class balance on measured residues for AUROC")
else:
    print("[5] no hotspots mapped for chain A - try another complex")

print(f"\nOK - full pipeline ran in {time.time()-t0:.1f}s")
