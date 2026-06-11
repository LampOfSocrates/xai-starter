"""Quick sweep: how do dropout + weight decay affect the IG completeness delta
on the same 5 demo complexes? Throwaway helper to pick honest hyperparameters
for the pinch_l3 appendix."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # repo root, for `import common`
sys.path.insert(0, HERE)                     # this dir, for `import pinch_common`
import torch
from captum.attr import IntegratedGradients
from pinch_common import (build_demo_dataset, train_struct2graph,
                          compute_all_saliencies, get_device)

device = get_device()
samples = build_demo_dataset(cache_dir="./data")
s0 = next(s for s in samples if not s["augmented"])  # barnase/barstar
ga, gb = s0["ga"].to(device), s0["gb"].to(device)


def ig_delta(model):
    model = model.to(device).eval()
    h_a = model.encode(ga.x, ga.edge_index).detach()
    h_b = model.encode(gb.x, gb.edge_index).detach()
    tgt = int(model(ga, gb).argmax(1).item())
    ig = IntegratedGradients(lambda a, b: model.head_from_embeddings(a, b))
    (_, _), d = ig.attribute((h_a, h_b),
                             baselines=(torch.zeros_like(h_a), torch.zeros_like(h_b)),
                             target=tgt, n_steps=200, return_convergence_delta=True)
    return float(d.abs().max())


def train_acc(model):
    model = model.to(device).eval()
    c = 0
    with torch.no_grad():
        for s in samples:
            c += int(model(s["ga"].to(device), s["gb"].to(device)).argmax(1)) == s["label"]
    return c / len(samples)


print(f"{'dropout':>8} {'wd':>8} {'delta':>8} {'train_acc':>10}")
for drop, wd in [(0.0, 0.0), (0.3, 1e-3), (0.5, 1e-2), (0.5, 3e-2), (0.3, 1e-2)]:
    m = train_struct2graph(samples, epochs=80, dropout=drop, weight_decay=wd,
                           device=device, verbose=False)
    print(f"{drop:>8} {wd:>8.0e} {ig_delta(m):>8.3f} {train_acc(m):>10.2f}")
