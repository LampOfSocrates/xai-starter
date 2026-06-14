"""
plm_common.py - shared helpers for the plm/ lesson notebooks.

Right now this holds the ProteinGym Deep-Mutational-Scanning (DMS) loader used by
``plm_l2_zero_shot_variants`` to benchmark zero-shot variant scores against real
experimental fitness.

Placement rationale: the repo-root ``common/`` package is for primitives shared
across the research tracks (pinch/clasp/edits). pLM-only plumbing lives here,
mirroring ``edits/edits_common.py``. If a second track ever needs DMS loading,
promote ``load_dms_assay`` up to ``common/``.
"""

from __future__ import annotations

import re

# ProteinGym v1 on the HuggingFace Hub. The DMS_substitutions config is stored as
# a handful of parquet shards with all assays concatenated, with columns:
#   mutated_sequence, target_seq, mutant, DMS_score, DMS_score_bin, DMS_id
PROTEINGYM_HF = "OATML-Markslab/ProteinGym_v1"

# --- Offline fallback -------------------------------------------------------
# A handful of single substitutions on the ESM example sequence with SYNTHETIC
# fitness values. These are NOT real measurements - they exist only so the
# notebook's code path runs end-to-end when the Hub is unreachable. Any Spearman
# computed against them is meaningless and the notebook says so loudly.
_FALLBACK_WT = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
_FALLBACK_VARIANTS = [
    ("M1A", -3.10), ("M1V", -2.40), ("K2R", -0.20), ("K2D", -1.80),
    ("T3S", 0.10), ("T3P", -2.90), ("V4I", 0.30), ("V4D", -2.10),
    ("R5K", -0.40), ("R5E", -1.60), ("Q6A", 0.50), ("Q6P", -2.20),
    ("E7D", -0.10), ("E7K", -1.30), ("R8K", -0.30), ("L9I", 0.20),
    ("L9P", -3.40), ("K10R", -0.25), ("S11A", 0.05), ("S11D", -0.30),
]


def parse_single_substitution(mutant: str):
    """'M1A' -> ('M', 0, 'A')  (0-based position). Returns None for multi-mutants
    like 'A1P:D2N' or malformed codes, so callers can skip them."""
    m = re.fullmatch(r"\s*([A-Z])(\d+)([A-Z])\s*", mutant)
    if not m:
        return None
    return m.group(1), int(m.group(2)) - 1, m.group(3)


def _collect_one_assay(stream, assay_id, max_variants, scan_limit):
    """Walk the streamed rows and keep single substitutions for ONE assay.
    Rows are contiguous per DMS_id (the shards concatenate per-assay files), so
    once we've seen our assay and the id changes, we stop."""
    chosen = assay_id
    wt_seq = None
    variants = []
    scanned = 0
    for row in stream:
        scanned += 1
        if scanned > scan_limit:
            break
        did = row["DMS_id"]
        if chosen is None:        # no assay requested -> take the first one we meet
            chosen = did
        if did != chosen:
            if variants:          # passed the end of our (contiguous) assay block
                break
            continue              # requested assay not reached yet; keep scanning
        parsed = parse_single_substitution(row["mutant"])
        if parsed is None:
            continue              # skip multi-substitution rows
        wt_aa, pos0, mut_aa = parsed
        score = row["DMS_score"]
        if score is None:
            continue
        if wt_seq is None:
            wt_seq = row["target_seq"]
        # Guard against any indexing offset: the WT residue in the code must match
        # the reference sequence at that position.
        if pos0 < 0 or pos0 >= len(wt_seq) or wt_seq[pos0] != wt_aa:
            continue
        variants.append((pos0, wt_aa, mut_aa, float(score)))
        if len(variants) >= max_variants:
            break
    return chosen, wt_seq, variants


def load_dms_assay(assay_id: str | None = None, max_variants: int = 200,
                   scan_limit: int = 60000):
    """Load one real ProteinGym DMS assay (single substitutions only).

    Args:
        assay_id: a ProteinGym ``DMS_id`` (e.g. "PABP_YEAST_Melamed_2013"). If
                  None, the first assay encountered in the stream is used.
        max_variants: cap on how many single-substitution variants to return
                      (keeps the downstream masked-LM scoring affordable).
        scan_limit: safety cap on rows scanned while looking for ``assay_id``.

    Returns:
        (assay_name, wt_seq, variants) where variants is a list of
        (pos0, wt_aa, mut_aa, dms_score) tuples.

    Raises:
        RuntimeError on any failure (no network, datasets missing, assay not
        found, ...) so the notebook can fall back to ``fallback_dms_assay``.
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001 - turn any import issue into a fallback signal
        raise RuntimeError(f"`datasets` unavailable: {e}")

    try:
        stream = load_dataset(PROTEINGYM_HF, "DMS_substitutions",
                              split="train", streaming=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not open ProteinGym stream from the Hub: {e}")

    try:
        chosen, wt_seq, variants = _collect_one_assay(
            stream, assay_id, max_variants, scan_limit)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"error while streaming ProteinGym: {e}")

    if not variants or wt_seq is None:
        raise RuntimeError(
            f"no single-substitution rows found for assay {chosen!r} "
            f"within {scan_limit} scanned rows")
    return chosen, wt_seq, variants


def fallback_dms_assay():
    """Tiny offline stand-in for :func:`load_dms_assay`. SYNTHETIC scores - for
    exercising the code path only, never for real conclusions."""
    variants = []
    for mut, score in _FALLBACK_VARIANTS:
        wt_aa, pos0, mut_aa = parse_single_substitution(mut)
        if _FALLBACK_WT[pos0] == wt_aa:
            variants.append((pos0, wt_aa, mut_aa, float(score)))
    return "FALLBACK_synthetic", _FALLBACK_WT, variants
