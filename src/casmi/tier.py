"""PubChem tier: the last N of the 25 slots go to PubChem-only candidates when they look strong enough.

The main list ranks the base pool (COCONUT ∪ train) as before. Candidates that only the PubChem
pool has (same mass window) are scored on a common scale with the main candidates — the summed
rank-normalised channel scores over the union window — times a popularity prior

    prior(c) = min(1, log1p(substances(c)) / log1p(sid_cap)) ** beta

(PubChem substance records per structure; beta = 0 switches it off). The tier is used only when
the best PubChem-only score beats the main list's ``gate_k``-th candidate; then the main list keeps
its first 25 − N entries and the N best PubChem-only candidates follow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from casmi.config import TOP_K
from casmi.ranker import rank_normalise


@dataclass(frozen=True)
class TierConfig:
    n_slots: int = 5
    gate_k: int = 5
    beta: float = 1.0
    sid_cap: float = 1000.0


def union_scores(keys: Sequence[str], per: Mapping[str, Mapping[str, float]],
                 weights: Mapping[str, float]) -> dict[str, float]:
    """Weighted sum of channel scores rank-normalised over the whole (union) window."""
    total = np.zeros(len(keys))
    for channel, w in weights.items():
        if w:
            total += w * rank_normalise(np.array([per.get(channel, {}).get(k, 0.0) for k in keys]))
    return dict(zip(keys, total.tolist()))


def prior(sid: float, cfg: TierConfig) -> float:
    if cfg.beta == 0:
        return 1.0
    return min(1.0, float(np.log1p(max(sid, 0.0)) / np.log1p(cfg.sid_cap))) ** cfg.beta


def tier_merge(main: Sequence[str], pub: Mapping[str, float], common: Mapping[str, float], cfg: TierConfig,
               k: int = TOP_K) -> list[str]:
    """Final top-``k``: ``main`` (ranked base-pool keys) with the tier applied.

    ``pub``: prior-weighted scores of the PubChem-only candidates; ``common``: union-window scores of
    the main candidates (the gate compares the best ``pub`` score with ``common[main[gate_k - 1]]``).
    """
    if not pub or cfg.n_slots <= 0:
        return list(main[:k])
    best = sorted(pub, key=lambda c: (-pub[c], c))
    gate = common.get(main[cfg.gate_k - 1], 0.0) if len(main) >= cfg.gate_k else 0.0
    if pub[best[0]] <= gate:
        return list(main[:k])
    n = min(cfg.n_slots, len(best), k)
    head = list(main[: k - n])
    return head + best[:n]
