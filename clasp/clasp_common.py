"""
clasp_common.py - building blocks for the contact-CLASP research notebooks
==========================================================================
These five notebooks (``clasp_l1`` .. ``clasp_l5``) test one question
(Definitive Guide project #7):

    Does ESM-2 attention recover physical residue contacts the way classical
    coevolution (DCA) does, learned from sequence alone - and does that recovery
    DECAY as alignment depth (Neff/L) falls? Does fusing an IG/gradient-derived
    contact map with attention help in the low-Neff/L regime?

This is the runnable MINIATURE. The ground truth is real (PDB Cα contacts of a
short well-characterised chain - BPTI from ``2ptc``). ESM-2 attention and the
gradient contact map are real, run on that chain. The one stand-in is the MSA:
a real project builds it with jackhmmer/hhblits; here we *simulate* an MSA whose
covariation is seeded from the chain's TRUE contacts, so mean-field DCA has
something real to recover and the Neff/L axis can be swept by subsampling it.
Swapping in a real MSA is the "scale up" step (see clasp_l2 / clasp_l5).

Shared primitives (PDB I/O, the AA alphabet, rank01) come from the repo-level
``common/`` package; everything specific to contact prediction lives here.
"""

import os

import numpy as np
import torch

from common import AMINO_ACIDS, AA_TO_IDX, get_device, fetch_pdb, parse_chain, rank01

# A short, well-characterised single chain as the miniature: BPTI (~58 res).
DEMO_PDB, DEMO_CHAIN = "2ptc", "I"
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"


# ---------------------------------------------------------------------------
# Ground truth - real PDB Cα contacts
# ---------------------------------------------------------------------------

def true_contacts(coords, threshold=8.0, min_sep=6):
    """Boolean L×L contact map: residues within ``threshold`` Å and ≥ ``min_sep``
    apart in sequence (the standard long-range contact definition)."""
    coords = np.asarray(coords, float)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    L = len(coords)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= min_sep
    return (d < threshold) & sep


def apc(matrix):
    """Average Product Correction - removes per-position and background bias that
    inflates contact scores (Dunn 2008). Standard post-processing for both
    attention-based and DCA contact maps."""
    m = np.asarray(matrix, float)
    total = m.sum()
    if total == 0:
        return m
    corr = m.sum(0, keepdims=True) * m.sum(1, keepdims=True) / total
    return m - corr


def precision_at_l(score, true, min_sep=6, frac=1.0):
    """Top-(frac·L) long-range contact precision over the upper triangle.

    The headline metric: of the L highest-scoring residue pairs (|i−j| ≥ min_sep),
    what fraction are real contacts?"""
    L = score.shape[0]
    iu = np.triu_indices(L, k=min_sep)
    s, t = np.asarray(score)[iu], np.asarray(true)[iu]
    n = int(L * frac)
    if n <= 0 or len(s) == 0:
        return float("nan")
    order = np.argsort(-s)[:n]
    return float(t[order].mean())


# ---------------------------------------------------------------------------
# Method 1 - ESM-2 attention contacts (Rao 2021, unsupervised average)
# ---------------------------------------------------------------------------

def esm_attention_contacts(seq, model_name=DEFAULT_MODEL, device=None):
    """Symmetrised, APC-corrected mean over all ESM-2 attention heads/layers.

    The unsupervised version of Rao et al. 2021: attention maps from a sequence-
    only pLM already concentrate on spatial contacts. Returns an (L, L) score."""
    from transformers import AutoTokenizer, EsmModel

    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    # ``eager`` attention so output_attentions actually returns the maps.
    model = EsmModel.from_pretrained(model_name, attn_implementation="eager").to(device).eval()
    enc = tok(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attn = torch.stack(out.attentions)[:, 0]        # (layers, heads, T, T)
    attn = attn[:, :, 1:-1, 1:-1]                    # strip <cls>/<eos>
    mean = attn.mean((0, 1)).float().cpu().numpy()   # (L, L)
    return apc(mean + mean.T)


# ---------------------------------------------------------------------------
# Method 2 - mean-field DCA on an MSA (the classical coevolution comparator)
# ---------------------------------------------------------------------------

def seq_to_ints(seq):
    """Map a 1-letter sequence to 0..19 indices (unknown -> 0)."""
    return np.array([AA_TO_IDX.get(a, 0) for a in seq], dtype=int)


def simulate_msa(seq, contact_map, n_seqs=500, mut_rate=0.15, couple_p=0.85, seed=0):
    """Simulate an MSA whose covariation is seeded from the TRUE contact map.

    Stand-in for a real jackhmmer/hhblits alignment: starting from the reference
    sequence, mutate positions at ``mut_rate``; when a position that participates
    in a true contact mutates, co-mutate its partner with prob ``couple_p`` to a
    coupled state. mean-field DCA should then recover those contacting pairs.
    Returns an (n_seqs, L) int array."""
    rng = np.random.default_rng(seed)
    base = seq_to_ints(seq)
    L = len(base)
    partner = {i: int(np.where(contact_map[i])[0][0])
               for i in range(L) if contact_map[i].any()}
    msa = np.tile(base, (n_seqs, 1))
    for n in range(n_seqs):
        for i in range(L):
            if rng.random() < mut_rate:
                msa[n, i] = rng.integers(20)
                j = partner.get(i)
                if j is not None and rng.random() < couple_p:
                    msa[n, j] = (msa[n, i] + 7) % 20   # deterministic coupling
    return msa


def neff(msa_ints, theta=0.2):
    """Effective number of sequences: down-weight near-duplicates (identity ≥
    1−theta). Neff/L is the alignment-depth axis the whole study turns on."""
    same = (msa_ints[:, None, :] == msa_ints[None, :, :]).mean(2) >= (1 - theta)
    return float((1.0 / same.sum(1)).sum())


def mfdca_contacts(msa_ints, q=20, pseudocount=0.5):
    """Mean-field DCA contact scores (Morcos 2011), APC-corrected.

    One-hot the MSA (reduced to q−1 states for invertibility), add a pseudocount,
    invert the covariance matrix to get couplings J, take the Frobenius norm of
    each i,j coupling block. Returns an (L, L) score."""
    N, L = msa_ints.shape
    qr = q - 1                                    # drop last state (gauge)
    fi = np.zeros((L, q))
    for a in range(q):
        fi[:, a] = (msa_ints == a).mean(0)
    fi = (1 - pseudocount) * fi + pseudocount / q
    # pairwise frequencies
    fij = np.zeros((L, q, L, q))
    onehot = np.zeros((N, L, q))
    onehot[np.arange(N)[:, None], np.arange(L)[None, :], msa_ints] = 1.0
    fij = np.einsum("nia,njb->iajb", onehot, onehot) / N
    fij = (1 - pseudocount) * fij + pseudocount / (q * q)
    # covariance over reduced states
    C = (fij[:, :qr, :, :qr]
         - fi[:, :qr, None, None] * fi[None, None, :, :qr]).reshape(L * qr, L * qr)
    J = -np.linalg.inv(C + 1e-4 * np.eye(L * qr)).reshape(L, qr, L, qr)
    score = np.sqrt((J ** 2).sum((1, 3)))         # Frobenius norm per pair
    np.fill_diagonal(score, 0.0)
    return apc(score)


# ---------------------------------------------------------------------------
# Method 3 - gradient/IG contact map from the masked-LM (for fusion)
# ---------------------------------------------------------------------------

def gradient_contact_map(seq, model_name=DEFAULT_MODEL, device=None, max_len=80):
    """Per-position masked-LM saliency → an (L, L) influence map.

    For each position i, mask it, take the gradient of the predicted-residue
    log-prob w.r.t. the input embeddings of every position j; ‖·‖ over the
    embedding dim gives influence[i, j]. This is the fast, runnable cousin of a
    full Integrated-Gradients contact map (the IG machinery lives in
    ``common/ig_l1``/``ig_l2``; swapping it in is the scale-up). Returns (L, L)."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    device = device or get_device()
    if len(seq) > max_len:
        raise ValueError(f"gradient map capped at {max_len} residues for the miniature")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
    mask_id = tok.mask_token_id
    enc = tok(seq, return_tensors="pt").to(device)
    ids = enc["input_ids"]
    L = ids.shape[1] - 2                            # minus <cls>/<eos>
    emb_layer = model.get_input_embeddings()
    influence = np.zeros((L, L))
    for i in range(L):
        masked = ids.clone()
        masked[0, i + 1] = mask_id
        inp = emb_layer(masked).detach().clone().requires_grad_(True)
        out = model(inputs_embeds=inp, attention_mask=enc["attention_mask"])
        logp = torch.log_softmax(out.logits[0, i + 1], -1)[ids[0, i + 1]]
        model.zero_grad(set_to_none=True)
        logp.backward()
        g = inp.grad[0, 1:-1].norm(dim=-1).detach().cpu().numpy()   # (L,)
        influence[i] = g
    return apc(influence + influence.T)


def fuse(map_a, map_b, method="rank"):
    """Combine two contact maps onto a common footing then average.

    ``rank`` rank-normalises each (robust to scale, the default); ``zscore``
    standardises each. Used to fuse attention with the gradient map in the
    low-Neff/L regime where attention alone weakens."""
    a, b = np.asarray(map_a, float), np.asarray(map_b, float)
    iu = np.triu_indices(a.shape[0], k=1)

    def norm(m):
        out = np.zeros_like(m)
        if method == "zscore":
            v = m[iu]
            out[iu] = (m[iu] - v.mean()) / (v.std() + 1e-9)
        else:
            out[iu] = rank01(m[iu])
        return out + out.T

    return 0.5 * (norm(a) + norm(b))


# ---------------------------------------------------------------------------
# The experiment: precision@L vs Neff/L, attention vs DCA vs fusion
# ---------------------------------------------------------------------------

def load_demo_chain(pdb=DEMO_PDB, chain=DEMO_CHAIN, cache_dir="./data"):
    """Fetch the miniature chain and return (seq, coords, true_contact_map)."""
    path = fetch_pdb(pdb, cache_dir)
    seq, coords, _ = parse_chain(path, chain)
    return seq, coords, true_contacts(coords)


def run_clasp_experiment(pdb=DEMO_PDB, chain=DEMO_CHAIN, model_name=DEFAULT_MODEL,
                         neff_fractions=(1.0, 0.3, 0.1), n_seqs=500, seed=0,
                         do_gradient=True, device=None, cache_dir="./data"):
    """End-to-end miniature: compute attention / DCA / fusion contact maps and
    report precision@L, with DCA + fusion swept across MSA subsample fractions
    (the Neff/L degradation axis). Returns a flat metrics dict."""
    device = device or get_device()
    seq, coords, true = load_demo_chain(pdb, chain, cache_dir)
    L = len(seq)

    attn = esm_attention_contacts(seq, model_name, device)
    metrics = {"L": L, "n_true_contacts": int(np.triu(true, 6).sum()),
               "precision_attention": precision_at_l(attn, true)}

    grad = None
    if do_gradient and L <= 80:
        grad = gradient_contact_map(seq, model_name, device)
        metrics["precision_gradient"] = precision_at_l(grad, true)

    full_msa = simulate_msa(seq, true, n_seqs=n_seqs, seed=seed)
    for frac in neff_fractions:
        sub = full_msa[: max(5, int(n_seqs * frac))]
        dca = mfdca_contacts(sub)
        nl = neff(sub) / L
        tag = f"{frac:g}"
        metrics[f"neff_per_L@{tag}"] = round(nl, 3)
        metrics[f"precision_dca@{tag}"] = precision_at_l(dca, true)
        if grad is not None:
            fused = fuse(attn, grad)
            metrics[f"precision_fusion@{tag}"] = precision_at_l(fused, true)
    return metrics
