"""Build the attrib-PINCH research notebooks (pinch_l1 .. pinch_l5) from cell specs.

These five notebooks walk through the Sendin-2025 PPI explainability pipeline
step by step, reusing the tested helpers in ``pinch_common.py`` so each notebook
stays short and readable. Unlike the auto-generated lesson notebooks (which come
from self-contained ``*.py`` scripts via ``_py_to_notebook.py``), these import a
shared module, so we author their cells explicitly here.

    .\\.venv\\Scripts\\python.exe ig\\_build_pinch_notebooks.py
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

# Path-bootstrap cell prepended to every notebook: works whether the kernel's
# cwd is the repo root or the pinch/ folder. It walks up to the repo root (the
# dir holding common/), puts both that root (for `import common`) and pinch/
# (for `import pinch_common`) on sys.path, and points one shared ./data cache.
BOOTSTRAP = '''\
import os, sys
ROOT = os.path.abspath("")
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "common")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)                          # for `import common`
sys.path.insert(0, os.path.join(ROOT, "pinch"))   # for `import pinch_common`
DATA = os.path.join(ROOT, "data")
WEIGHTS = os.path.join(DATA, "pinch_struct2graph.pt")
print("repo root:", ROOT)'''


def md(text):
    return ("md", text)


def code(text):
    return ("code", text)


def build(name, cells):
    nb = nbf.v4.new_notebook()
    out = []
    for kind, text in cells:
        if kind == "md":
            out.append(nbf.v4.new_markdown_cell(text))
        else:
            out.append(nbf.v4.new_code_cell(text))
    nb["cells"] = out
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path}  ({len(out)} cells)")


RUN_ORDER = (
    "> **Run order matters.** These cells build on each other — run them top to "
    "bottom (*Run All*). Heavy steps (training, Integrated Gradients) are small "
    "by design but still compute; on a shared machine, run them when the box is "
    "idle."
)

# ---------------------------------------------------------------------------
# pinch_l1 — from two protein chains to two graphs
# ---------------------------------------------------------------------------
L3 = [
    md("# attrib-PINCH · pinch_l1 — From a PPI complex to two residue graphs\n\n"
       "The research question (see the repo README and the Obsidian research "
       "vault) is whether the **consensus** of three explainability methods "
       "predicts experimentally-measured binding hotspots better than any "
       "single method. This first notebook builds the data primitive everything "
       "else stands on: turning a two-chain protein complex into a **pair of "
       "graphs**, exactly as Sendin (2025) did on PINDER.\n\n"
       "We use **barnase–barstar (PDB `1BRS`)** as the running example — the "
       "textbook alanine-scanning hotspot system, and one with rich `SKEMPI "
       "v2.0` ΔΔG data we'll validate against in lesson 7."),
    md(RUN_ORDER),
    md("## Setup\n\nLocate the shared helpers and the data cache."),
    code(BOOTSTRAP),
    code("from pinch_common import (fetch_pdb, parse_chain, contact_graph,\n"
         "                          interface_mask, get_device)\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "print('device:', get_device())"),
    md("## Step 1 — Download the complex\n\n"
       "`fetch_pdb` pulls the structure from RCSB once and caches it. It pins "
       "`certifi`'s CA bundle so the download works inside the venv even when a "
       "conda base environment has exported a stale `SSL_CERT_FILE`."),
    code("pdb_path = fetch_pdb('1brs', cache_dir=DATA)\n"
         "print('cached at:', pdb_path)"),
    md("## Step 2 — Parse the two chains\n\n"
       "In `1BRS`, chain **A** is barnase (the enzyme / receptor) and chain "
       "**D** is barstar (its inhibitor / ligand). `parse_chain` keeps only "
       "standard residues with a Cα atom and returns the sequence, the Cα "
       "coordinates, and the author residue numbers (we need those numbers in "
       "lesson 7 to line residues up with SKEMPI mutations)."),
    code("seq_a, ca_a, resnums_a = parse_chain(pdb_path, 'A')\n"
         "seq_d, ca_d, resnums_d = parse_chain(pdb_path, 'D')\n"
         "print(f'chain A (barnase): {len(seq_a)} residues')\n"
         "print(f'chain D (barstar): {len(seq_d)} residues')\n"
         "print('barnase seq:', seq_a)"),
    md("## Step 3 — Build a residue contact graph per chain\n\n"
       "Each residue becomes a node (positioned at its Cα); an edge joins any "
       "two residues whose Cα atoms are within **8 Å** — the standard contact "
       "cutoff. Node features are one-hot amino acids (swap in ESM-2 embeddings "
       "later for the multi-modal extension Sendin flagged)."),
    code("g_a = contact_graph(seq_a, ca_a, threshold=8.0)\n"
         "g_d = contact_graph(seq_d, ca_d, threshold=8.0)\n"
         "print('barnase graph:', g_a.num_nodes, 'nodes /', g_a.num_edges, 'edges')\n"
         "print('barstar graph:', g_d.num_nodes, 'nodes /', g_d.num_edges, 'edges')\n"
         "print('node feature shape:', tuple(g_a.x.shape))"),
    md("## Step 4 — Sanity-check the graph as a contact map\n\n"
       "A contact map is just the graph's adjacency matrix drawn as an image. "
       "The diagonal band is the protein backbone (residue *i* near *i±1*); "
       "off-diagonal blocks are tertiary contacts where the chain folds back on "
       "itself."),
    code("A = np.zeros((g_a.num_nodes, g_a.num_nodes))\n"
         "ei = g_a.edge_index.numpy()\n"
         "A[ei[0], ei[1]] = 1\n"
         "fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))\n"
         "ax[0].imshow(A, cmap='Greys', origin='lower')\n"
         "ax[0].set_title('Barnase contact map (8 Å)')\n"
         "ax[0].set_xlabel('residue'); ax[0].set_ylabel('residue')\n"
         "deg = A.sum(1)\n"
         "ax[1].bar(range(len(deg)), deg, color='steelblue')\n"
         "ax[1].set_title('Per-residue contact degree')\n"
         "ax[1].set_xlabel('residue'); ax[1].set_ylabel('# contacts')\n"
         "plt.tight_layout(); plt.show()"),
    md("## Step 5 — Where is the interface?\n\n"
       "`interface_mask` flags residues on each chain whose Cα lies within a "
       "cutoff of *any* Cα on the partner chain — a Cα approximation of the "
       "binding interface. These are the residues most likely to contain "
       "hotspots, and the structural reference we'll compare the saliency "
       "rankings against (with SKEMPI as the experimental ground truth)."),
    code("mask_a, mask_d = interface_mask(ca_a, ca_d, cutoff=8.0)\n"
         "print(f'barnase interface residues: {int(mask_a.sum())} / {len(mask_a)}')\n"
         "print(f'barstar interface residues: {int(mask_d.sum())} / {len(mask_d)}')\n"
         "iface_resnums = [resnums_a[i] for i in np.where(mask_a)[0]]\n"
         "print('barnase interface residue numbers:', iface_resnums)"),
    code("plt.figure(figsize=(11, 2.4))\n"
         "plt.bar(range(len(mask_a)), mask_a.astype(int), color='crimson', width=1.0)\n"
         "plt.title('Barnase: interface residues (1 = within 8 Å of barstar)')\n"
         "plt.xlabel('residue index'); plt.yticks([0, 1])\n"
         "plt.tight_layout(); plt.show()"),
    md("## Recap\n\n"
       "We turned one PPI complex into two PyG graphs and located its interface "
       "— all from a single cached `.pdb`. **Lesson 4** trains a Struct2Graph "
       "classifier (shared-weight GCN + mutual attention) on a small set of such "
       "complexes; **lessons 5–7** explain it three ways and test the consensus "
       "against SKEMPI hotspots."),
]

# ---------------------------------------------------------------------------
# pinch_l2 — Struct2Graph: the model and its training
# ---------------------------------------------------------------------------
L4 = [
    md("# attrib-PINCH · pinch_l2 — Struct2Graph: model + training\n\n"
       "We re-implement, in miniature, the architecture Sendin adapted from "
       "**Struct2Graph** (Baranwal & Mayank 2022):\n\n"
       "1. two chains → two graphs (lesson 3),\n"
       "2. a **shared-weight GCN** encodes each chain's residues,\n"
       "3. a **mutual-attention** block couples the two chains' residues,\n"
       "4. attended context vectors are pooled, concatenated, and classified.\n\n"
       "Sendin trained a 31-class PINDER *cluster* classifier (100 PPIs each). "
       "PINDER is multi-GB, so as a runnable stand-in we treat a handful of "
       "real complexes — all present in SKEMPI v2.0 — as their own classes, with "
       "light edge-dropout augmentation for several examples each. Same shape of "
       "task; trains in seconds. The full PINDER loader is the natural next step "
       "(see the closing notes)."),
    md(RUN_ORDER),
    md("## Setup"),
    code(BOOTSTRAP),
    code("from pinch_common import (DEMO_COMPLEXES, build_demo_dataset,\n"
         "                          Struct2Graph, train_struct2graph,\n"
         "                          load_or_train_demo, get_device)\n"
         "import torch\n"
         "import matplotlib.pyplot as plt\n"
         "device = get_device(); print('device:', device)"),
    md("## Step 1 — The miniature multi-class dataset\n\n"
       "Each complex below is one class. `build_demo_dataset` downloads each "
       "structure, builds the two chain graphs, and adds edge-dropout variants "
       "so every class has a few non-identical examples."),
    code("for i, (pdb, ca, cb, label) in enumerate(DEMO_COMPLEXES):\n"
         "    print(f'class {i}: {label:24s} ({pdb} {ca}/{cb})')"),
    code("samples = build_demo_dataset(threshold=8.0, augment=6, cache_dir=DATA)\n"
         "n_classes = len({s['label'] for s in samples})\n"
         "print(f'{len(samples)} samples across {n_classes} classes')"),
    md("## Step 2 — The model\n\n"
       "A compact `Struct2Graph`: 2 GCN layers, hidden width 32, a mutual-"
       "attention head, and a small classifier. The forward pass stashes the "
       "node embeddings and attention matrix on `model._cache` for the "
       "explainability notebooks."),
    code("model = Struct2Graph(in_dim=20, hidden=32, num_classes=n_classes)\n"
         "n_params = sum(p.numel() for p in model.parameters())\n"
         "print(f'{n_params:,} trainable parameters')"),
    md("Forward pass on one complex, untrained — just to confirm the shapes. "
       "The attention matrix is (L_receptor × L_ligand): one weight per pair of "
       "residues across the interface."),
    code("s0 = next(s for s in samples if not s['augmented'])  # pristine 1brs\n"
         "logits = model(s0['ga'], s0['gb'])\n"
         "print('logits shape:', tuple(logits.shape))\n"
         "print('attention shape (La x Ld):', tuple(model._cache['attn_ab'].shape))"),
    md("## Step 3 — Train\n\n"
       "Categorical cross-entropy, Adam, a few dozen epochs. At this scale the "
       "model essentially memorises the classes — which is *on message*: "
       "Sendin's headline result was that a 0.97-F1 PINDER model leaned on "
       "global shape shortcuts. We are not chasing a number here; we want a "
       "trained model whose attention, IG, and GNNExplainer signals the next "
       "notebooks can interrogate. We save the weights so lessons 5–7 reuse "
       "this exact model."),
    code("model = train_struct2graph(samples, hidden=32, epochs=80, lr=5e-3,\n"
         "                           device=device, verbose=True)\n"
         "import os\n"
         "os.makedirs(DATA, exist_ok=True)\n"
         "torch.save(model.state_dict(), WEIGHTS)\n"
         "print('saved weights ->', WEIGHTS)"),
    md("## Step 4 — Training accuracy\n\n"
       "On the demo set we expect near-perfect accuracy (it is tiny and the "
       "model overfits). The value of the project is **not** this number — it is "
       "whether the *explanations* point at real biology, which lesson 7 tests."),
    code("model.eval()\n"
         "correct = 0\n"
         "with torch.no_grad():\n"
         "    for s in samples:\n"
         "        ga, gb = s['ga'].to(device), s['gb'].to(device)\n"
         "        pred = int(model(ga, gb).argmax(1))\n"
         "        correct += (pred == s['label'])\n"
         "print(f'train accuracy: {correct}/{len(samples)} = {correct/len(samples):.2f}')"),
    md("## Closing notes — scaling to real PINDER\n\n"
       "To turn this miniature into the actual experiment:\n\n"
       "- replace `build_demo_dataset` with a PINDER loader (`pip install "
       "pinder`; iterate the 31-cluster holo subset),\n"
       "- keep the same `Struct2Graph` and training loop,\n"
       "- optionally swap one-hot node features for ESM-2 embeddings "
       "(`gnn_l4` shows how).\n\n"
       "Everything downstream (lessons 5–7) is written against the model "
       "interface, not the dataset, so it carries over unchanged."),
]

# ---------------------------------------------------------------------------
# pinch_l3 — Integrated Gradients on the node embeddings
# ---------------------------------------------------------------------------
L5 = [
    md("# attrib-PINCH · pinch_l3 — Integrated Gradients on the GCN node embeddings\n\n"
       "**xAI method #2 of 3.** Integrated Gradients (Sundararajan, Taly & Yan "
       "2017) attributes a prediction to its inputs by integrating gradients "
       "along a straight path from a neutral *baseline* to the actual input.\n\n"
       "For a message-passing GCN, attributing all the way back to the raw "
       "one-hot atoms is awkward (gradients through discrete graph structure). "
       "The recommended target — and what Sendin's future-work note points at — "
       "is the **post-GCN node-embedding matrix**: a clean continuous tensor on "
       "which IG behaves. We use Captum and a zero-embedding baseline."),
    md(RUN_ORDER),
    md("## Setup"),
    code(BOOTSTRAP),
    code("from pinch_common import load_or_train_demo, interface_mask, parse_chain, fetch_pdb, get_device\n"
         "from captum.attr import IntegratedGradients\n"
         "import numpy as np, torch\n"
         "import matplotlib.pyplot as plt\n"
         "device = get_device(); print('device:', device)"),
    md("## Step 1 — Load the trained model\n\n"
       "`load_or_train_demo` reuses the weights saved in lesson 4 (or trains a "
       "fresh model if you run this notebook standalone)."),
    code("model, samples = load_or_train_demo(WEIGHTS, hidden=32, device=device,\n"
         "                                    cache_dir=DATA)\n"
         "model = model.to(device).eval()\n"
         "s0 = next(s for s in samples if not s['augmented'])  # barnase/barstar\n"
         "ga, gb = s0['ga'].to(device), s0['gb'].to(device)\n"
         "print('explaining complex:', s0['name'])"),
    md("## Step 2 — Encode, then integrate\n\n"
       "We freeze the two chains' node embeddings `h_a`, `h_b` (post-GCN), then "
       "ask IG how the predicted-class logit depends on each. The baseline is "
       "the all-zero embedding ('absence of signal'). We use **200 steps** and "
       "print the *completeness delta* — by the completeness axiom the "
       "attributions should sum to `F(input) − F(baseline)`, so this delta "
       "should be ~0. Watch what it actually does below; the discussion at the "
       "end explains the result."),
    code("target = int(model(ga, gb).argmax(1).item())\n"
         "h_a = model.encode(ga.x, ga.edge_index).detach()\n"
         "h_b = model.encode(gb.x, gb.edge_index).detach()\n"
         "ig = IntegratedGradients(lambda a, b: model.head_from_embeddings(a, b))\n"
         "(att_a, att_b), delta = ig.attribute(\n"
         "    (h_a, h_b),\n"
         "    baselines=(torch.zeros_like(h_a), torch.zeros_like(h_b)),\n"
         "    target=target, n_steps=200, return_convergence_delta=True)\n"
         "print('target class:', target)\n"
         "print('completeness delta (want ~0):', float(delta.abs().max()))"),
    md("## Step 3 — One score per residue\n\n"
       "IG returns one attribution per embedding dimension; we take the "
       "absolute sum across dimensions to get a single saliency per residue."),
    code("ig_a = att_a.abs().sum(1).cpu().numpy()\n"
         "ig_d = att_b.abs().sum(1).cpu().numpy()\n"
         "print('barnase IG saliency shape:', ig_a.shape)\n"
         "top = np.argsort(ig_a)[::-1][:8]\n"
         "print('top-8 barnase residues by IG:', top.tolist())"),
    md("## Step 4 — Visualise saliency vs the interface\n\n"
       "Overlay the IG saliency on the barnase chain and mark the interface "
       "residues. A faithful explanation should place mass on or near the "
       "interface — though, as Sendin warns, attention (and saliency) often also "
       "light up distal residues."),
    code("path = fetch_pdb('1brs', cache_dir=DATA)\n"
         "_, ca_a, _ = parse_chain(path, 'A')\n"
         "_, ca_d, _ = parse_chain(path, 'D')\n"
         "mask_a, _ = interface_mask(ca_a, ca_d, cutoff=8.0)\n"
         "fig, ax = plt.subplots(figsize=(11, 3))\n"
         "ax.bar(range(len(ig_a)), ig_a, color='slateblue', width=1.0, label='IG saliency')\n"
         "ymax = ig_a.max() if ig_a.max() > 0 else 1.0\n"
         "ax.bar(range(len(mask_a)), mask_a * ymax, color='crimson', width=1.0,\n"
         "       alpha=0.18, label='interface')\n"
         "ax.set_title(f'Barnase — Integrated Gradients (class {target})')\n"
         "ax.set_xlabel('residue index'); ax.set_ylabel('|IG| summed')\n"
         "ax.legend(); plt.tight_layout(); plt.show()"),
    md("## About that completeness delta — an honest read\n\n"
       "The delta printed above is **large (~8–9), and it does *not* shrink when "
       "you raise `n_steps` from 48 to 200.** That rules out Riemann-"
       "approximation error (which would shrink with more steps). The real cause "
       "is **gradient saturation**: our demo model is trained to near-zero loss "
       "on five complexes, so its logit surface is flat almost everywhere and "
       "the gradients IG samples along the path are ~0 except in a thin region "
       "near the baseline. IG then under-counts the true `F(input) − F(baseline)` "
       "gap.\n\n"
       "This is not a bug in the code — it is IG correctly reporting that an "
       "**overfit** model is a poor explanation target, and it rhymes with the "
       "project's central theme (Sendin's shortcut-learning collapse). The fixes "
       "are *model-side*, not *step-side*:\n\n"
       "- train a **regularised, non-memorising** model (the unused dropout / "
       "weight-decay Sendin mentioned; or simply more data — real PINDER),\n"
       "- or use a **non-zero baseline** (e.g. the mean embedding) closer to the "
       "data manifold.\n\n"
       "Other standing IG caveats to carry into lesson 7:\n\n"
       "- **Baseline dependence.** Attributions change with the baseline; we "
       "report the zero-embedding baseline.\n"
       "- **No interactions.** IG scores residues individually; it won't say "
       "that residue *i* matters *because* it pairs with residue *j* — a real "
       "limitation for interface hotspots, which are often cooperative."),
    md("## Appendix — can regularization rescue it? An honest sweep\n\n"
       "If saturation from overfitting is the cause, turning on **dropout + L2 "
       "weight decay** — same real data, same architecture, only the training "
       "regime changes — should reduce the completeness delta. Let's sweep a few "
       "regimes and look at the IG delta **and** the train accuracy together. "
       "The accuracy is the catch: a small delta is only meaningful if the model "
       "still discriminates."),
    code("from pinch_common import train_struct2graph\n"
         "\n"
         "def _ig_delta(m):\n"
         "    m = m.to(device).eval()\n"
         "    ha = m.encode(ga.x, ga.edge_index).detach()\n"
         "    hb = m.encode(gb.x, gb.edge_index).detach()\n"
         "    tgt = int(m(ga, gb).argmax(1).item())\n"
         "    ig_ = IntegratedGradients(lambda a, b: m.head_from_embeddings(a, b))\n"
         "    (_, _), d = ig_.attribute((ha, hb),\n"
         "        baselines=(torch.zeros_like(ha), torch.zeros_like(hb)),\n"
         "        target=tgt, n_steps=200, return_convergence_delta=True)\n"
         "    return float(d.abs().max())\n"
         "\n"
         "def _train_acc(m):\n"
         "    m = m.to(device).eval()\n"
         "    c = sum(int(m(s['ga'].to(device), s['gb'].to(device)).argmax(1)) == s['label']\n"
         "            for s in samples)\n"
         "    return c / len(samples)\n"
         "\n"
         "regimes = [(0.0, 0.0), (0.3, 1e-3), (0.3, 1e-2), (0.5, 1e-2)]\n"
         "sweep = []\n"
         "for drop, wd in regimes:\n"
         "    m = train_struct2graph(samples, epochs=80, dropout=drop,\n"
         "                           weight_decay=wd, device=device, verbose=False)\n"
         "    sweep.append({'dropout': drop, 'wd': wd,\n"
         "                  'delta': _ig_delta(m), 'acc': _train_acc(m)})\n"
         "print(f\"{'dropout':>8}{'wd':>9}{'IG delta':>10}{'train acc':>11}\")\n"
         "for r in sweep:\n"
         "    print(f\"{r['dropout']:>8}{r['wd']:>9.0e}{r['delta']:>10.3f}{r['acc']:>11.2f}\")"),
    code("import numpy as np\n"
         "labels = [f\"d={r['dropout']}\\nwd={r['wd']:.0e}\" for r in sweep]\n"
         "x = np.arange(len(sweep))\n"
         "fig, ax1 = plt.subplots(figsize=(8, 4))\n"
         "ax1.bar(x - 0.2, [r['delta'] for r in sweep], width=0.4,\n"
         "        color='firebrick', label='IG delta')\n"
         "ax1.set_ylabel('IG completeness delta', color='firebrick')\n"
         "ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)\n"
         "ax2 = ax1.twinx()\n"
         "ax2.bar(x + 0.2, [r['acc'] for r in sweep], width=0.4,\n"
         "        color='steelblue', label='train accuracy')\n"
         "ax2.set_ylabel('train accuracy', color='steelblue'); ax2.set_ylim(0, 1.05)\n"
         "ax2.axhline(1/5, ls='--', c='grey', lw=0.8)  # 5-class chance\n"
         "ax1.set_title('No sweet spot at 5 complexes: memorize (big delta) vs collapse (acc=0.2)')\n"
         "plt.tight_layout(); plt.show()"),
    md("## What the sweep actually shows — and why it backs the 'real data' point\n\n"
       "Read the two bars together:\n\n"
       "- **Light regularization** (`wd=1e-3`) still reaches ~100% train accuracy "
       "— the model is *still memorising* — so the delta barely drops.\n"
       "- **Heavier regularization** (`wd≥1e-2`) sends train accuracy to ~0.20, "
       "i.e. **chance** for 5 classes: the model has **collapsed** to a near-"
       "constant output. Its IG delta is ~0, but that is a *degenerate* win — "
       "`F(input) ≈ F(baseline)` because the model no longer discriminates, not "
       "because IG found a faithful explanation.\n\n"
       "So at this scale there is **no regularization setting that is both "
       "discriminating and non-saturated.** With only five complexes the model "
       "can memorise or give up — it cannot learn *generalisable* interaction "
       "features, which is exactly what a converged, trustworthy IG needs.\n\n"
       "That is the real lesson, and it sharpens the answer to *'are we not "
       "working with real data?'*: the inputs and labels are fully real, but five "
       "complexes is too small a *slice* for any hyperparameter to fix. The "
       "honest lever is **scale** — training on the real PINDER 31-cluster subset "
       "(~3,100 complexes), where the model has enough signal to learn real "
       "features and moderate regularization *can* then keep it out of the "
       "saturated regime. That is **option (b)**, to be run only once agreed."),
]

# ---------------------------------------------------------------------------
# pinch_l4 — attention + GNNExplainer (methods 1 and 3)
# ---------------------------------------------------------------------------
L6 = [
    md("# attrib-PINCH · pinch_l4 — Mutual attention & GNNExplainer\n\n"
       "**xAI methods #1 and #3.** Two more views of the same trained model:\n\n"
       "- **Mutual attention** — the model's own attention weights, summed per "
       "residue (what Sendin used, and what he cautioned is *not* a guaranteed-"
       "faithful explanation).\n"
       "- **GNNExplainer** (Ying 2019) — learns a soft mask over nodes/edges "
       "that preserves the prediction; PyG ships it built-in.\n\n"
       "We then ask whether the two agree. If three methods disagree (the Fazel "
       "2025 finding on pLMs), a **consensus** may be more trustworthy than any "
       "one — which is exactly the hypothesis lesson 7 tests."),
    md(RUN_ORDER),
    md("## Setup"),
    code(BOOTSTRAP),
    code("from pinch_common import (load_or_train_demo, _gnnexplainer_saliency,\n"
         "                          rank01, get_device)\n"
         "import numpy as np, torch\n"
         "import matplotlib.pyplot as plt\n"
         "device = get_device(); print('device:', device)"),
    code("model, samples = load_or_train_demo(WEIGHTS, hidden=32, device=device,\n"
         "                                    cache_dir=DATA)\n"
         "model = model.to(device).eval()\n"
         "s0 = next(s for s in samples if not s['augmented'])\n"
         "ga, gb = s0['ga'].to(device), s0['gb'].to(device)\n"
         "print('explaining complex:', s0['name'])"),
    md("## Step 1 — Attention saliency\n\n"
       "The mutual-attention matrix is (barnase × barstar). Summing each "
       "barnase residue's attention *to* barstar gives a per-residue saliency: "
       "how much this residue attends across the interface."),
    code("_ = model(ga, gb)  # populates model._cache['attn_ab']\n"
         "attn = model._cache['attn_ab'].detach().cpu().numpy()  # (La, Ld)\n"
         "attn_a = attn.sum(1)  # barnase residues\n"
         "attn_d = attn.sum(0)  # barstar residues\n"
         "print('attention saliency (barnase) shape:', attn_a.shape)"),
    md("## Step 2 — GNNExplainer saliency\n\n"
       "GNNExplainer expects a single graph→prediction signature, but our model "
       "takes a *pair* of graphs. The helper wraps it: it freezes the partner "
       "chain's embeddings and learns a node mask over *this* chain — so the "
       "explanation is conditional on the partner. (Documented design choice.)"),
    code("h_b = model.encode(gb.x, gb.edge_index).detach()\n"
         "h_a = model.encode(ga.x, ga.edge_index).detach()\n"
         "gnnx_a = _gnnexplainer_saliency(model, ga.x, ga.edge_index, h_b, epochs=80)\n"
         "gnnx_d = _gnnexplainer_saliency(model, gb.x, gb.edge_index, h_a, epochs=80)\n"
         "print('GNNExplainer saliency (barnase) shape:', np.asarray(gnnx_a).shape)"),
    md("## Step 3 — Do the two methods agree?\n\n"
       "Rank-normalise both to [0, 1] (they live on different scales) and "
       "correlate. Low agreement is the *motivation* for a consensus, not a bug "
       "— it echoes Fazel 2025: no single attribution method is reliable alone."),
    code("ra, rg = rank01(attn_a), rank01(gnnx_a)\n"
         "rho = float(np.corrcoef(ra, rg)[0, 1])\n"
         "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
         "ax[0].scatter(ra, rg, s=14, color='teal')\n"
         "ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8)\n"
         "ax[0].set_xlabel('attention rank'); ax[0].set_ylabel('GNNExplainer rank')\n"
         "ax[0].set_title(f'Barnase: agreement (Pearson r = {rho:.2f})')\n"
         "ax[1].bar(range(len(ra)), ra, width=1.0, alpha=0.6, label='attention', color='goldenrod')\n"
         "ax[1].bar(range(len(rg)), rg, width=1.0, alpha=0.6, label='GNNExplainer', color='purple')\n"
         "ax[1].set_xlabel('residue index'); ax[1].set_ylabel('rank-normalised saliency')\n"
         "ax[1].legend(); ax[1].set_title('Two methods, same model')\n"
         "plt.tight_layout(); plt.show()\n"
         "print(f'rank correlation barnase: {rho:.3f}')"),
    md("## Recap\n\n"
       "We now have all three single-method saliencies (IG from lesson 5, "
       "attention and GNNExplainer here). They don't fully agree — so which, if "
       "any, recovers the *experimentally measured* hotspots? **Lesson 7** pulls "
       "SKEMPI v2.0, builds the consensus, and runs the benchmark."),
]

# ---------------------------------------------------------------------------
# pinch_l5 — the hypothesis test against SKEMPI
# ---------------------------------------------------------------------------
L7 = [
    md("# attrib-PINCH · pinch_l5 — The hypothesis test: consensus vs SKEMPI hotspots\n\n"
       "Everything converges here. The falsifiable hypothesis:\n\n"
       "> The **consensus** of mutual attention, Integrated Gradients, and "
       "GNNExplainer isolates PPI-causal residues **better than any single "
       "method**, measured by agreement with alanine-scanning ΔΔG hotspots "
       "(ΔΔG ≥ 2 kcal/mol) in **SKEMPI v2.0**.\n\n"
       "We compute all three saliencies + their consensus, pull the SKEMPI "
       "ground truth for barnase–barstar, and score each method's hotspot "
       "recovery with AUROC."),
    md(RUN_ORDER),
    md("## Setup"),
    code(BOOTSTRAP),
    code("from pinch_common import (load_or_train_demo, compute_all_saliencies,\n"
         "                          fetch_skempi, skempi_ddg, map_ddg_to_nodes,\n"
         "                          fetch_pdb, parse_chain, get_device)\n"
         "import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "from sklearn.metrics import roc_auc_score\n"
         "device = get_device(); print('device:', device)"),
    md("## Step 1 — All three methods + consensus, in one call\n\n"
       "`compute_all_saliencies` runs attention, IG, and GNNExplainer and "
       "returns the per-residue signals plus the **consensus** = mean of the "
       "three rank-normalised scores, for both chains."),
    code("model, samples = load_or_train_demo(WEIGHTS, hidden=32, device=device,\n"
         "                                    cache_dir=DATA)\n"
         "s0 = next(s for s in samples if not s['augmented'])  # barnase/barstar\n"
         "sal = compute_all_saliencies(model, s0['ga'], s0['gb'], ig_steps=200,\n"
         "                             gnnx_epochs=80, device=device)\n"
         "print('explained class:', sal['target'])\n"
         "print('IG completeness delta:', f\"{sal['ig_delta']:.2e}\")\n"
         "print('methods:', [k for k in sal['a'] if k != 'target'])"),
    md("## Step 2 — SKEMPI v2.0 ground truth\n\n"
       "SKEMPI v2.0 holds ~7,000 mutation-induced binding-affinity changes. We "
       "compute ΔΔG = RT·ln(Kd_mut / Kd_wt) for single-point **alanine** "
       "mutations on `1BRS`, then map them onto our graph's residues. A residue "
       "is a **hotspot** if its ΔΔG ≥ 2 kcal/mol (the standard threshold)."),
    code("df = fetch_skempi(cache_dir=DATA)\n"
         "ddg = skempi_ddg(df, '1brs', alanine_only=True)\n"
         "print(f'SKEMPI: {len(df)} rows; 1brs single-point alanine: {len(ddg)} rows')\n"
         "path = fetch_pdb('1brs', cache_dir=DATA)\n"
         "_, _, resnums_a = parse_chain(path, 'A')\n"
         "_, _, resnums_d = parse_chain(path, 'D')\n"
         "ddg_a, hot_a, n_a = map_ddg_to_nodes(ddg, 'A', resnums_a)\n"
         "ddg_d, hot_d, n_d = map_ddg_to_nodes(ddg, 'D', resnums_d)\n"
         "print(f'chain A (barnase): {n_a} residues measured, {int(hot_a.sum())} hotspots')\n"
         "print(f'chain D (barstar): {n_d} residues measured, {int(hot_d.sum())} hotspots')"),
    md("## Step 3 — The benchmark\n\n"
       "For every method, on the residues SKEMPI actually measured, compute "
       "AUROC for separating hotspots from non-hotspots. We pool both chains to "
       "get as many measured residues as possible. **Higher = better hotspot "
       "recovery.**"),
    code("methods = ['attention', 'ig', 'gnnexplainer', 'consensus']\n"
         "\n"
         "def measured_pairs(sal_chain, ddg_node, hot):\n"
         "    m = np.isfinite(ddg_node)\n"
         "    return m, hot[m].astype(int)\n"
         "\n"
         "m_a, y_a = measured_pairs(sal['a'], ddg_a, hot_a)\n"
         "m_d, y_d = measured_pairs(sal['d'], ddg_d, hot_d)\n"
         "y = np.concatenate([y_a, y_d])\n"
         "print(f'pooled measured residues: {y.size}, hotspots: {int(y.sum())}')\n"
         "\n"
         "results = {}\n"
         "if 0 < y.sum() < y.size:\n"
         "    for k in methods:\n"
         "        score = np.concatenate([np.asarray(sal['a'][k])[m_a],\n"
         "                                np.asarray(sal['d'][k])[m_d]])\n"
         "        results[k] = roc_auc_score(y, score)\n"
         "    for k in methods:\n"
         "        print(f'  {k:13s} AUROC = {results[k]:.3f}')\n"
         "else:\n"
         "    print('not enough class balance to score AUROC on this complex')"),
    code("if results:\n"
         "    fig, ax = plt.subplots(figsize=(7, 4))\n"
         "    colors = ['goldenrod', 'slateblue', 'purple', 'crimson']\n"
         "    ax.bar(results.keys(), results.values(), color=colors)\n"
         "    ax.axhline(0.5, ls='--', c='k', lw=0.8, label='random')\n"
         "    ax.set_ylabel('hotspot-recovery AUROC'); ax.set_ylim(0, 1)\n"
         "    ax.set_title('Does consensus beat single methods? (barnase–barstar)')\n"
         "    ax.legend(); plt.tight_layout(); plt.show()"),
    md("## Step 4 — Read the result honestly\n\n"
       "On this **single toy complex** with a handful of measured residues and a "
       "model trained on five structures, do not expect the hypothesis to be "
       "settled — the AUROCs are noisy and the consensus may or may not lead. "
       "That is the correct scientific posture, and it mirrors the project's "
       "ethos (Sendin's value was honest error analysis, not a headline F1).\n\n"
       "**What a real verdict needs:**\n\n"
       "1. **PINDER at scale** — train on the 31-cluster holo subset, not five "
       "complexes, so the model learns genuine interaction features rather than "
       "memorising classes.\n"
       "2. **Many SKEMPI complexes** — aggregate AUROC across every PINDER↔SKEMPI "
       "overlap (~hundreds of complexes), with confidence intervals, instead of "
       "one.\n"
       "3. **A converged IG** — keep the completeness delta near zero.\n"
       "4. **Per-residue, not per-class** — optionally reframe as interface-"
       "residue prediction so every residue is a labelled example.\n\n"
       "The plumbing for all of that is in `pinch_common.py`; only the dataset "
       "swaps. This notebook set is the de-risked week-1→3 scaffold from the "
       "research plan — it proves the pipeline runs end to end before committing "
       "compute to full PINDER."),
]

if __name__ == "__main__":
    build("pinch_l1_pinder_graphs.ipynb", L3)
    build("pinch_l2_struct2graph_train.ipynb", L4)
    build("pinch_l3_ig_node_embeddings.ipynb", L5)
    build("pinch_l4_attention_gnnexplainer.ipynb", L6)
    build("pinch_l5_skempi_consensus_benchmark.ipynb", L7)
