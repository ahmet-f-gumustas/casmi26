"""PubChem tier: the last N of the 25 slots go to PubChem-only candidates when they look strong enough.

The main list ranks the base pool (COCONUT ∪ train) as before. Candidates that only the PubChem
pool has (same mass window) are scored on a common scale with the main candidates — the summed
rank-normalised channel scores over the union window — times a popularity prior

    prior(c) = min(1, log1p(substances(c)) / log1p(sid_cap)) ** beta

(PubChem substance records per structure; beta = 0 switches it off). The tier is used only when
the best PubChem-only score beats the main list's ``gate_k``-th candidate; then the main list keeps
its first 25 − N entries and the N best PubChem-only candidates follow.

``order="popularity"`` (recall probe 2a) keeps the PubChem-only candidates whose union-window score
is at least ``min_common`` and fills the slots in order of PubChem substance count; the gate then
compares their best union-window score.

``slot_merge`` is the same idea for candidates from other mass windows (alternative adducts,
recall probe 2b): at most N of them, each beating the main list's ``gate_k``-th score, either take
the last slots (``tail``) or are merged in by score (``merit``).
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
    order: str = "score"  # or "popularity"
    min_common: float = 0.0  # popularity order: minimum union-window score of a PubChem candidate


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


def pub_candidates(keys: Sequence[str], common: Mapping[str, float], sid: Mapping[str, float],
                   cfg: TierConfig) -> list[tuple[str, float]]:
    """PubChem-only ``keys`` in slot order, each with the score the gate compares."""
    if cfg.order == "score":
        s = {c: common[c] * prior(sid.get(c, 0), cfg) for c in keys if common[c] >= cfg.min_common}
        return sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))
    if cfg.order == "popularity":
        kept = [c for c in keys if common[c] >= cfg.min_common and common[c] > 0]
        return [(c, common[c]) for c in sorted(kept, key=lambda c: (-sid.get(c, 0), -common[c], c))]
    raise ValueError(f"unknown tier order {cfg.order!r}")


def tier_merge(main: Sequence[str], pub: Mapping[str, float] | Sequence[tuple[str, float]],
               common: Mapping[str, float], cfg: TierConfig, k: int = TOP_K) -> list[str]:
    """Final top-``k``: ``main`` (ranked base-pool keys) with the tier applied.

    ``pub``: prior-weighted scores of the PubChem-only candidates (ranked by score), or
    ``pub_candidates`` output (already in slot order); ``common``: union-window scores of the main
    candidates (the gate compares the best ``pub`` score with ``common[main[gate_k - 1]]``).
    """
    if not pub or cfg.n_slots <= 0:
        return list(main[:k])
    if isinstance(pub, Mapping):
        pub = sorted(pub.items(), key=lambda kv: (-kv[1], kv[0]))
    gate = common.get(main[cfg.gate_k - 1], 0.0) if len(main) >= cfg.gate_k else 0.0
    if max(s for _, s in pub) <= gate:
        return list(main[:k])
    n = min(cfg.n_slots, len(pub), k)
    head = list(main[: k - n])
    return head + [c for c, _ in pub[:n]]


def slot_merge(main: Sequence[str], main_scores: Mapping[str, float], extra: Mapping[str, float], n_slots: int,
               gate_k: int, placement: str = "tail", k: int = TOP_K) -> list[str]:
    """Final top-``k`` from ``main`` (ranked) and up to ``n_slots`` ``extra`` candidates (scores on the same
    scale as ``main_scores``) that beat the main list's ``gate_k``-th score; ``tail`` puts them in the last
    slots, ``merit`` merges them in by score (ties: main first)."""
    gate = main_scores.get(main[gate_k - 1], 0.0) if len(main) >= gate_k else 0.0
    best = sorted((c for c, s in extra.items() if s > gate and s > 0), key=lambda c: (-extra[c], c))[:n_slots]
    if not best:
        return list(main[:k])
    if placement == "tail":
        return list(main[: k - len(best)]) + best
    if placement != "merit":
        raise ValueError(f"unknown placement {placement!r}")
    head = list(main[: k - len(best)])
    order = {c: i for i, c in enumerate(head)}
    merged = sorted(head + best, key=lambda c: (-(main_scores.get(c, 0.0) if c in order else extra[c]),
                                                0 if c in order else 1, order.get(c, 0), c))
    return merged[:k]
