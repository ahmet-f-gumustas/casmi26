"""Analog difficulty: how close is a molecule's nearest structure in the reference library?

``nn_tanimoto`` is the highest Morgan (r2, 2048) Tanimoto between a molecule and any
library structure other than itself; ``n_analogs`` counts library structures with
Tanimoto ≥ ``min_tanimoto`` whose neutral mass is within ±``max_shift_da`` (the analog
channel's search window). Both are computed with blocked matrix products on unpacked
fingerprints.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

NN_BINS = (0.0, 0.3, 0.5, 0.7, 1.0001)
BIN_LABELS = ("0–0.3", "0.3–0.5", "0.5–0.7", "0.7+")


def unpack(fps: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Packed fingerprints [n, 256] → 0/1 float32 [n, 2048]."""
    return np.unpackbits(np.asarray(fps), axis=1).astype(np.float32)


def nn_tanimoto(query_fps: NDArray[np.uint8], query_keys: Sequence[str], query_masses: NDArray[np.float64],
                ref_fps: NDArray[np.uint8], ref_keys: Sequence[str], ref_masses: NDArray[np.float64],
                min_tanimoto: float = 0.6, max_shift_da: float = 200.0, chunk: int = 1000,
                ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Per query: nearest-neighbour Tanimoto among references (its own key excluded) and analog count."""
    ref = unpack(ref_fps)
    ref_bits = ref.sum(axis=1)
    ref_index = pd.Index(ref_keys)
    ref_m = np.asarray(ref_masses, dtype=np.float64)
    nn = np.zeros(len(query_keys))
    count = np.zeros(len(query_keys), dtype=np.int64)
    for start in range(0, len(query_keys), chunk):
        stop = min(start + chunk, len(query_keys))
        q = unpack(query_fps[start:stop])
        inter = q @ ref.T
        union = q.sum(axis=1)[:, None] + ref_bits[None, :] - inter
        t = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        self_pos = ref_index.get_indexer(list(query_keys[start:stop]))
        has_self = self_pos >= 0
        t[np.flatnonzero(has_self), self_pos[has_self]] = -1.0
        nn[start:stop] = t.max(axis=1)
        near = np.abs(ref_m[None, :] - np.asarray(query_masses[start:stop])[:, None]) <= max_shift_da
        count[start:stop] = ((t >= min_tanimoto) & near).sum(axis=1)
    return np.maximum(nn, 0.0), count


def nn_bin(values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Bin index (0..3) for ``BIN_LABELS``."""
    return np.clip(np.searchsorted(np.asarray(NN_BINS), values, side="right") - 1, 0, len(BIN_LABELS) - 1)


def tilt_to_target(shares: NDArray[np.float64], bin_mrr: NDArray[np.float64], target: float) -> tuple[NDArray, float]:
    """Exponentially tilted ``shares`` (closest in KL) whose mixture of ``bin_mrr`` equals ``target``.

    Returns ``(profile, lambda)``; raises ``ValueError`` if ``target`` is outside the bin-MRR range.
    """
    p = np.asarray(shares, dtype=np.float64)
    m = np.asarray(bin_mrr, dtype=np.float64)
    ok = p > 0
    if not (m[ok].min() <= target <= m[ok].max()):
        raise ValueError(f"target {target} outside per-bin MRR range [{m[ok].min():.3f}, {m[ok].max():.3f}]")

    def mix(lam: float) -> tuple[NDArray, float]:
        w = np.where(ok, p * np.exp(lam * (m - m.mean())), 0.0)
        w = w / w.sum()
        return w, float(w @ m)

    lo, hi = -200.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mix(mid)[1] < target:
            lo = mid
        else:
            hi = mid
    w, _ = mix((lo + hi) / 2)
    return w, (lo + hi) / 2


def library_nn(keys: Sequence[str], fps: NDArray[np.uint8], masses: NDArray[np.float64], key_row: dict[str, int],
               library: Sequence[str], cache=None, chunk: int = 1000) -> pd.DataFrame:
    """``key, nn_tanimoto, n_analogs`` of ``keys`` against ``library`` structures (pool fingerprints).

    ``cache`` (a parquet path) keeps earlier results; only missing keys are computed.
    """
    cached = pd.read_parquet(cache) if cache is not None and cache.exists() else pd.DataFrame(
        columns=["key", "nn_tanimoto", "n_analogs"])
    done = set(cached["key"])
    todo = [k for k in dict.fromkeys(keys) if k in key_row and k not in done]
    if todo:
        lib = [k for k in library if k in key_row]
        q_idx = np.array([key_row[k] for k in todo])
        r_idx = np.array([key_row[k] for k in lib])
        nn, n_an = nn_tanimoto(np.asarray(fps[q_idx]), todo, masses[q_idx], np.asarray(fps[r_idx]), lib,
                               masses[r_idx], chunk=chunk)
        new = pd.DataFrame({"key": todo, "nn_tanimoto": nn, "n_analogs": n_an})
        cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(cache, index=False)
    return cached[cached["key"].isin(set(keys))].reset_index(drop=True)


def neighbours_at_least(query_fps: NDArray[np.uint8], caps: Sequence[float], ref_fps: NDArray[np.uint8],
                        chunk: int = 200) -> list[NDArray[np.int64]]:
    """Per query: positions of the references whose Tanimoto to it is at least its cap (``inf``: none)."""
    ref = unpack(ref_fps)
    ref_bits = ref.sum(axis=1)
    caps = np.asarray(caps, dtype=np.float64)
    out = []
    for start in range(0, len(caps), chunk):
        stop = min(start + chunk, len(caps))
        q = unpack(query_fps[start:stop])
        inter = q @ ref.T
        union = q.sum(axis=1)[:, None] + ref_bits[None, :] - inter
        t = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        out.extend(np.flatnonzero(t[i] >= cap) if np.isfinite(cap) else np.zeros(0, dtype=np.int64)
                   for i, cap in enumerate(caps[start:stop]))
    return out
