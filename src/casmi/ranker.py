"""Ranker v0: weighted sum of per-molecule rank-normalised channel scores.

Each channel's scores over a molecule's candidate window are mapped to their
percentile among those candidates (a candidate with no evidence — score ≤ 0 —
gets 0), so channels with different scales can be added. Weights are tuned on
one half of the validation molecules and checked on the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from casmi.config import TOP_K

CHANNELS = ("library", "analog", "fragment")


def rank_normalise(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fraction of candidates scoring ≤ each positive score; non-positive scores map to 0."""
    out = np.zeros(len(scores))
    pos = scores > 0
    if pos.any():
        out[pos] = np.searchsorted(np.sort(scores), scores[pos], side="right") / len(scores)
    return out


def combine(keys: Sequence[str], channel_scores: Mapping[str, Mapping[str, float]],
            weights: Mapping[str, float]) -> NDArray[np.float64]:
    """Weighted sum of rank-normalised channel scores for ``keys``."""
    total = np.zeros(len(keys))
    for channel, w in weights.items():
        if w:
            raw = np.array([channel_scores.get(channel, {}).get(k, 0.0) for k in keys])
            total += w * rank_normalise(raw)
    return total


def rank(keys: Sequence[str], combined: NDArray[np.float64], tiebreak: Mapping[str, float] | None = None,
         k: int = TOP_K) -> list[str]:
    """Top-``k`` keys by combined score (ties: tiebreak score, then key); zero scores are not ranked."""
    tb = tiebreak or {}
    order = sorted((i for i in range(len(keys)) if combined[i] > 0),
                   key=lambda i: (-combined[i], -tb.get(keys[i], 0.0), keys[i]))
    return [keys[i] for i in order[:k]]
