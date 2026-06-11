"""Scoring helpers shared across tracks.

``rank01`` puts signals that live on completely different scales (attention, IG,
GNNExplainer; or attention vs DCA in `clasp/`) onto a common [0, 1] footing
before they are averaged or fused. Lifted from the original combined research module.
"""

import numpy as np


def rank01(values):
    """Rank-normalise a 1-D array to [0, 1] (ties broken by order).

    Used to put signals that live on completely different scales onto a common
    footing before averaging into a consensus or fusing two methods.
    """
    v = np.asarray(values, dtype=float)
    order = v.argsort()
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v))
    return ranks / max(len(v) - 1, 1)
