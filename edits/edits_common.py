"""
edits_common.py - building blocks for the attrib-EDITS research notebooks
=========================================================================
These five notebooks (``edits_l1`` .. ``edits_l5``) test one question
(Definitive Guide project #5):

    What is the smallest, still-realistic edit that flips a stability/fitness
    predictor - and do those edits match real ΔΔG? On-manifold (plausibility-
    constrained) counterfactuals, validated on a split DISJOINT from the
    predictor's training data.

This is the runnable MINIATURE. The predictor here is **ESM-2 itself** used as a
zero-shot fitness proxy (one forward gives the log-prob of every single
substitution - the standard masked/wt-marginal trick). The counterfactual search
and the on-manifold plausibility penalty are real and run on a small protein
(GB1 domain). What is intentionally NOT wired is the ΔΔG validation against
Megascale / FireProtDB through ThermoMPNN + ESM-1v - that is the scale-up, and it
is gated by the **data-leakage rules in this folder's ``CLAUDE.md``** (use
ThermoMPNN's published splits + FireProt-HF, run both predictors, report baselines
on a disjoint split). ``validate_against_ddg`` raises with that pointer so a demo
run can't silently fake the headline.

Shared primitives come from the repo-level ``common/`` package; everything
specific to the counterfactual search lives here.
"""

import numpy as np
import torch

from common import AMINO_ACIDS, get_device

# GB1 B1 domain (56 aa) - small, sequence-only, has a public DMS landscape (the
# optim track / ProteinGym), so it is a natural miniature for edit experiments.
DEMO_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"

_CACHE = {}


def _load(model_name=DEFAULT_MODEL, device=None):
    """Cache (tokenizer, MLM model, list of 20 AA token ids) per model name."""
    device = device or get_device()
    if model_name not in _CACHE:
        from transformers import AutoTokenizer, EsmForMaskedLM
        tok = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
        aa_ids = [tok.convert_tokens_to_ids(a) for a in AMINO_ACIDS]
        _CACHE[model_name] = (tok, model, aa_ids)
    return _CACHE[model_name]


# ---------------------------------------------------------------------------
# The borrowed predictor - ESM-2 as a zero-shot fitness/naturalness proxy
# ---------------------------------------------------------------------------

def position_logp(seq, model_name=DEFAULT_MODEL, device=None):
    """One forward → (L, 20) log-probabilities over the 20 AAs at each position.

    From a single forward of ``seq`` (the wt-marginal trick) we read off the
    model's preferred residue at every position; the log-prob of any single
    substitution is then a table lookup, so a greedy edit step costs one forward."""
    tok, model, aa_ids = _load(model_name, device)
    device = device or get_device()
    enc = tok(seq, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]                       # (T, V)
    logp = torch.log_softmax(logits, -1)[1:-1][:, aa_ids]     # (L, 20)
    return logp.float().cpu().numpy()


def fitness(seq, model_name=DEFAULT_MODEL, device=None):
    """Predictor score = total log-likelihood the model assigns to ``seq``'s own
    residues (a pseudo-naturalness / stability proxy). Higher = more 'fit'. Summed
    (not averaged) so each edit's contribution accumulates in the search."""
    from common import AA_TO_IDX
    lp = position_logp(seq, model_name, device)
    idx = [AA_TO_IDX.get(a, 0) for a in seq]
    return float(lp[np.arange(len(seq)), idx].sum())


# ---------------------------------------------------------------------------
# Counterfactual search - smallest edit that flips the predictor
# ---------------------------------------------------------------------------

def greedy_counterfactual(seq, target_drop=8.0, max_edits=8, plausibility_weight=0.0,
                          model_name=DEFAULT_MODEL, device=None):
    """Greedily edit ``seq`` to DROP the predictor's fitness by ``target_drop``.

    Each step: one forward gives (L, 20) log-probs; score every untouched
    single substitution by

        J(i, a) = Δlogp(i→a)  −  plausibility_weight · logp(i, a)

    and apply the minimiser (most fitness-reducing, optionally kept plausible).
    ``plausibility_weight = 0`` is the unconstrained (off-manifold) search;
    > 0 is the on-manifold version - the comparison is the project's novelty.

    Returns a dict: ``edits`` [(pos, wt, mut)], ``final_seq``, ``trajectory``
    (fitness after each edit), ``flipped`` (bool), ``n_edits``, ``mean_plausibility``
    (mean logp of the substituted residues)."""
    from common import AA_TO_IDX
    seq0 = seq
    cur = list(seq)
    f0 = fitness("".join(cur), model_name, device)
    edits, traj, plaus = [], [f0], []
    used = set()
    for _ in range(max_edits):
        lp = position_logp("".join(cur), model_name, device)         # (L, 20)
        wt_idx = np.array([AA_TO_IDX.get(a, 0) for a in cur])
        wt_lp = lp[np.arange(len(cur)), wt_idx][:, None]             # (L, 1)
        delta = lp - wt_lp                                            # Δlogp per (i,a)
        J = delta - plausibility_weight * lp
        for i in used:
            J[i] = np.inf                                            # don't re-edit a position
        J[np.arange(len(cur)), wt_idx] = np.inf                      # no-op edits excluded
        i, a = np.unravel_index(np.argmin(J), J.shape)
        edits.append((int(i), cur[i], AMINO_ACIDS[a]))
        plaus.append(float(lp[i, a]))
        cur[i] = AMINO_ACIDS[a]
        used.add(int(i))
        f = fitness("".join(cur), model_name, device)
        traj.append(f)
        if (f0 - f) >= target_drop:
            break
    return {
        "wt_seq": seq0, "final_seq": "".join(cur), "edits": edits,
        "trajectory": traj, "n_edits": len(edits),
        "flipped": (f0 - traj[-1]) >= target_drop,
        "fitness_drop": float(f0 - traj[-1]),
        "mean_plausibility": float(np.mean(plaus)) if plaus else float("nan"),
    }


def gradient_saliency(seq, model_name=DEFAULT_MODEL, device=None):
    """Per-position |∂fitness/∂embedding| — the gradient counterfactual's "where
    to edit" signal (one forward+backward). Complements the discrete search; the
    full Integrated-Gradients version reuses the machinery in ``common/ig_l1``."""
    from common import AA_TO_IDX
    tok, model, aa_ids = _load(model_name, device)
    device = device or get_device()
    enc = tok(seq, return_tensors="pt").to(device)
    emb_layer = model.get_input_embeddings()
    inp = emb_layer(enc["input_ids"]).detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=inp, attention_mask=enc["attention_mask"]).logits[0]
    logp = torch.log_softmax(logits, -1)[1:-1][:, aa_ids]
    idx = torch.tensor([AA_TO_IDX.get(a, 0) for a in seq], device=device)
    score = logp[torch.arange(len(seq)), idx].mean()
    model.zero_grad(set_to_none=True)
    score.backward()
    return inp.grad[0, 1:-1].norm(dim=-1).detach().cpu().numpy()      # (L,)


# ---------------------------------------------------------------------------
# Metrics + the (gated) real validation
# ---------------------------------------------------------------------------

def edit_metrics(result):
    """Counterfactual-quality metrics from a ``greedy_counterfactual`` result:
    validity (did it flip?), proximity (#edits, fewer is better), plausibility
    (mean logp of the edited residues, higher = more on-manifold)."""
    return {
        "validity": bool(result["flipped"]),
        "proximity_n_edits": result["n_edits"],
        "plausibility": result["mean_plausibility"],
        "fitness_drop": result["fitness_drop"],
    }


def run_edits_experiment(seq=DEMO_SEQ, target_drop=8.0, max_edits=8,
                         model_name=DEFAULT_MODEL, device=None):
    """Compare the OFF-manifold vs ON-manifold counterfactual search on the same
    target: do both flip the predictor, and does the plausibility penalty buy
    more realistic edits (higher plausibility) at the cost of a few more edits?"""
    out = {}
    for label, w in (("off_manifold", 0.0), ("on_manifold", 1.0)):
        m = edit_metrics(greedy_counterfactual(
            seq, target_drop=target_drop, max_edits=max_edits,
            plausibility_weight=w, model_name=model_name, device=device))
        out.update({f"{label}_{k}": v for k, v in m.items()})
    return out


def validate_against_ddg(*args, **kwargs):
    """Reserved scale-up: validate counterfactual edits against measured ΔΔG.

    Intentionally not wired - per this folder's ``CLAUDE.md`` the headline is only
    valid on a leakage-free split. Wiring this means: reuse ThermoMPNN's published
    ``dataset_splits/`` + FireProt-HF, run BOTH ESM-1v (mild leakage) and
    ThermoMPNN (severe), report each predictor's baseline on that split, and
    stratify by identity-to-training. Raises until that is done deliberately."""
    raise NotImplementedError(
        "ΔΔG validation is the scale-up; it must run on a split disjoint from the "
        "predictor's training data. See edits/CLAUDE.md for the required protocol "
        "(ThermoMPNN dataset_splits/ + FireProt-HF, both predictors, baselines, "
        "identity stratification).")
