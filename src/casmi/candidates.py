"""Candidate structure pool: mass-sorted arrays on disk, ppm-window lookups.

Layout of a pool directory (built by ``scripts/build_pool.py``), all in the same
ascending-mass order:

- ``masses.npy``: float64 neutral-atom monoisotopic mass
- ``fps.npy``: uint8 ``[N, 256]``, Morgan radius-2 2048-bit fingerprint (``np.packbits``)
  of the stereo-free canonical tautomer (or ``fps.npz``, compressed, in uploaded datasets)
- ``meta.parquet``: ``key`` (scorer key), ``smiles``, ``exact_mass``, ``formula``,
  ``formal_charge``, ``in_train``, ``in_coconut``, ``coconut_id``
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from casmi.chem import neutral_mass
from casmi.config import EXTERNAL_DIR

POOL_DIR = EXTERNAL_DIR / "pool"
# timsTOF precursors read ~+2 ppm high in positive mode (notes/ADDUCTS.md), so the
# neutral mass derived from them is shifted by the same amount.
TIMSTOF_OFFSET_PPM = 2.0


@dataclass
class CandidatePool:
    masses: NDArray[np.float64]
    meta: pd.DataFrame
    path: Path | None = None
    _fps: NDArray[np.uint8] | None = field(default=None, repr=False)
    _excluded: NDArray[np.bool_] | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path = POOL_DIR) -> CandidatePool:
        masses = np.load(path / "masses.npy")
        meta = pd.read_parquet(path / "meta.parquet")
        if len(meta) != len(masses):
            raise ValueError(f"{path}: meta and masses disagree in length")
        return cls(masses=masses, meta=meta, path=path)

    @property
    def fps(self) -> NDArray[np.uint8]:
        """Packed fingerprints: ``fps.npy`` memory-mapped, or ``fps.npz`` (compressed, for upload) in memory."""
        if self._fps is None:
            if (self.path / "fps.npy").exists():
                self._fps = np.load(self.path / "fps.npy", mmap_mode="r")
            else:
                with np.load(self.path / "fps.npz") as z:
                    self._fps = z["fps"]
        return self._fps

    def __len__(self) -> int:
        return len(self.masses)

    def exclude(self, keys: Iterable[str]) -> CandidatePool:
        """Pool that never returns structures whose scorer key is in ``keys`` (Class 3 simulation)."""
        excluded = self.meta["key"].isin(set(keys)).to_numpy()
        if self._excluded is not None:
            excluded = excluded | self._excluded
        return CandidatePool(self.masses, self.meta, self.path, self._fps, excluded)

    def window(self, mass: float, ppm: float = 10.0, offset_ppm: float = TIMSTOF_OFFSET_PPM) -> NDArray[np.intp]:
        """Pool indices within ±``ppm`` of ``mass`` corrected for an instrument offset."""
        centre = mass / (1.0 + offset_ppm * 1e-6)
        delta = centre * ppm * 1e-6
        lo = np.searchsorted(self.masses, centre - delta, side="left")
        hi = np.searchsorted(self.masses, centre + delta, side="right")
        idx = np.arange(lo, hi)
        return idx if self._excluded is None else idx[~self._excluded[idx]]


def molecule_masses(queries: pd.DataFrame, key: str = "molecule_id") -> pd.Series:
    """Median neutral mass per molecule from its spectra's ``precursor_mz`` and ``adduct``."""
    m = pd.Series(np.nan, index=queries.index)
    for adduct, idx in queries.groupby("adduct").groups.items():
        m.loc[idx] = neutral_mass(queries.loc[idx, "precursor_mz"].to_numpy(), adduct)
    return m.groupby(queries[key]).median()
