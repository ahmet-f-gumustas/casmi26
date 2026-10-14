"""Library channel: spectral-entropy match of query spectra against reference spectra.

For a molecule, each query spectrum is compared with every reference spectrum of
the same ionization mode whose structure lies in the molecule's candidate mass
window (entropy similarity, ``ms_entropy``). A structure's score for one query
spectrum is its best-matching reference spectrum; scores are aggregated over the
molecule's query spectra by ``max`` or ``mean`` (in the mean, a query spectrum
with no comparable reference spectrum of that structure counts as 0). Structures
without reference spectra are not ranked by this channel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ms_entropy import calculate_entropy_similarity, clean_spectrum
from numpy.typing import NDArray

from casmi.config import TOP_K

Spectrum = tuple[str, NDArray[np.float32]]  # (ionization_mode, cleaned peaks [n, 2])
AGGREGATIONS = ("max", "mean")


@dataclass(frozen=True)
class LibraryConfig:
    ms2_tol_da: float = 0.02
    noise_threshold: float = 0.01
    precursor_removal_da: float = 1.6  # drop the precursor ion and anything heavier
    min_ms2_difference_da: float = 0.05


def prepare(mz: NDArray[np.float64], intensity: NDArray[np.float64], precursor_mz: float,
            cfg: LibraryConfig = LibraryConfig()) -> NDArray[np.float32]:
    """Cleaned, intensity-normalised peaks ready for entropy similarity (may be empty)."""
    peaks = np.column_stack([mz, intensity]).astype(np.float32)
    if not len(peaks):
        return peaks.reshape(0, 2)
    return clean_spectrum(peaks, max_mz=precursor_mz - cfg.precursor_removal_da, noise_threshold=cfg.noise_threshold,
                          min_ms2_difference_in_da=cfg.min_ms2_difference_da, normalize_intensity=True)


def score_molecule(queries: Sequence[Spectrum], candidate_keys: Sequence[str],
                   references: Mapping[str, Sequence[Spectrum]], cfg: LibraryConfig = LibraryConfig()) -> pd.DataFrame:
    """Per candidate structure with references: ``max`` / ``mean`` over query spectra and ``n_refs``."""
    keys = [k for k in candidate_keys if references.get(k)]
    if not keys or not queries:
        return pd.DataFrame(columns=["key", *AGGREGATIONS, "n_refs"])
    sims = np.zeros((len(queries), len(keys)))
    for j, key in enumerate(keys):
        for mode_r, ref in references[key]:
            if not len(ref):
                continue
            for i, (mode_q, q) in enumerate(queries):
                if mode_q != mode_r or not len(q):
                    continue
                s = calculate_entropy_similarity(q, ref, ms2_tolerance_in_da=cfg.ms2_tol_da, clean_spectra=False)
                if s > sims[i, j]:
                    sims[i, j] = s
    return pd.DataFrame({"key": keys, "max": sims.max(axis=0), "mean": sims.mean(axis=0),
                         "n_refs": [len(references[k]) for k in keys]})


def rank(scores: pd.DataFrame, by: str = "max", k: int = TOP_K) -> list[str]:
    """Top-``k`` structure keys by ``by`` (ties: more references, then key), positive scores only."""
    if by not in AGGREGATIONS:
        raise ValueError(f"by must be one of {AGGREGATIONS}")
    hits = scores[scores[by] > 0].sort_values([by, "n_refs", "key"], ascending=[False, False, True])
    return hits["key"].head(k).tolist()
