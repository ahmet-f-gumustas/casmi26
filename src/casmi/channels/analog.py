"""Analog channel: mass-shifted analog propagation.

For each query spectrum, the library is searched with Flash entropy search (hybrid
mode: fragments match directly or shifted by the precursor difference) and the
top-K analogs within ±``max_shift_da`` of the query precursor are kept. A candidate
structure ``c`` then scores, for that spectrum,

    max over analogs a of  sim(a) ** p  ×  Tanimoto(FP(c), FP(structure of a))

and the per-spectrum scores are aggregated over the molecule's spectra (max/mean).
Unlike the library channel this ranks candidates without reference spectra (Class 2).
A candidate's own library spectra count as its analogs only at a similarity of at least
``self_min_sim`` (0 = always, > 1 = never): a Tanimoto-1 self match at moderate similarity
otherwise gives every structure with library spectra a head start over one without.

One Flash index per ionization mode is built over the whole cleaned library; a
split's library mask is applied at query time, so the same index serves every
validation seed and the final run.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ms_entropy import FlashEntropySearch
from numpy.typing import NDArray

MODES = ("positive", "negative")


@dataclass(frozen=True)
class AnalogConfig:
    top_k: int = 100
    max_shift_da: float = 200.0
    power: float = 3.0
    method: str = "hybrid"  # flash search mode: "open" or "hybrid"
    ms2_tol_da: float = 0.02
    noise_threshold: float = 0.01
    max_peaks: int = 100  # per indexed spectrum, keeps the index small
    self_min_sim: float = 0.0  # own library spectra count as analogs only at this similarity or above


@dataclass
class AnalogIndex:
    """Flash entropy index per ionization mode, with library rows / precursors in index order."""

    searchers: dict[str, FlashEntropySearch]
    rows: dict[str, NDArray[np.int64]]
    precursors: dict[str, NDArray[np.float64]]

    @classmethod
    def build(cls, library: Path, cfg: AnalogConfig = AnalogConfig()) -> AnalogIndex:
        spectra: dict[str, list[dict]] = {m: [] for m in MODES}
        pf = pq.ParquetFile(library)
        cols = ["row", "ionization_mode", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=cols).to_pandas()
            for row, mode, prec, mz, it in zip(t["row"], t["ionization_mode"], t["precursor_mz"], t["ms2_mzs"],
                                               t["ms2_normalized_intensities"]):
                if mode in spectra and len(mz):
                    peaks = np.column_stack([mz, it]).astype(np.float32)
                    spectra[mode].append({"precursor_mz": float(prec), "peaks": peaks, "row": int(row)})
        searchers, rows, precs = {}, {}, {}
        for mode, lst in spectra.items():
            es = FlashEntropySearch(max_ms2_tolerance_in_da=cfg.ms2_tol_da * 1.2)
            ordered = es.build_index(lst, noise_threshold=cfg.noise_threshold, max_peak_num=cfg.max_peaks)
            searchers[mode] = es
            rows[mode] = np.array([s["row"] for s in ordered], dtype=np.int64)
            precs[mode] = np.array([s["precursor_mz"] for s in ordered], dtype=np.float64)
        return cls(searchers, rows, precs)

    def save(self, path: Path) -> None:
        """Version-independent layout: ms_entropy's raw arrays per mode plus .npy row/precursor maps."""
        path.mkdir(parents=True, exist_ok=True)
        tolerances = {}
        for mode, es in self.searchers.items():
            es.write(path / mode)
            np.save(path / f"{mode}_rows.npy", self.rows[mode])
            np.save(path / f"{mode}_precursors.npy", self.precursors[mode])
            tolerances[mode] = float(es.entropy_search.max_ms2_tolerance_in_da)
        (path / "index.json").write_text(json.dumps({"modes": list(self.searchers), "max_ms2_tolerance_in_da": tolerances}))

    @classmethod
    def load(cls, path: Path) -> AnalogIndex:
        info = json.loads((path / "index.json").read_text())
        searchers, rows, precs = {}, {}, {}
        for mode in info["modes"]:
            es = FlashEntropySearch(max_ms2_tolerance_in_da=info["max_ms2_tolerance_in_da"][mode])
            es.read(_mode_dir(path, mode))
            searchers[mode] = es
            rows[mode] = np.load(path / f"{mode}_rows.npy")
            precs[mode] = np.load(path / f"{mode}_precursors.npy")
        return cls(searchers, rows, precs)

    def search(self, mode: str, precursor_mz: float, peaks: NDArray[np.float32], cfg: AnalogConfig,
               allowed: NDArray[np.bool_] | None = None) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Top-K library rows (and similarities) within ±max_shift_da; ``allowed`` is indexed by library row."""
        es = self.searchers.get(mode)
        if es is None or not len(peaks):
            return np.zeros(0, np.int64), np.zeros(0)
        res = es.search(precursor_mz=precursor_mz, peaks=peaks, ms2_tolerance_in_da=cfg.ms2_tol_da,
                        method=cfg.method, noise_threshold=cfg.noise_threshold)
        sims = np.asarray(res[f"{cfg.method}_search"], dtype=np.float64)
        lo = np.searchsorted(self.precursors[mode], precursor_mz - cfg.max_shift_da, side="left")
        hi = np.searchsorted(self.precursors[mode], precursor_mz + cfg.max_shift_da, side="right")
        sims, rows = sims[lo:hi], self.rows[mode][lo:hi]
        if allowed is not None:
            sims = np.where(allowed[rows], sims, 0.0)
        k = min(cfg.top_k, int((sims > 0).sum()))
        if k == 0:
            return np.zeros(0, np.int64), np.zeros(0)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return rows[top], sims[top]


def _mode_dir(path: Path, mode: str) -> Path:
    """The index directory of ``mode``; a flattened copy (``<mode>__<file>``, as uploaded to a Kaggle
    dataset, which cannot hold sub-folders) is re-assembled with symlinks in a temporary folder."""
    if (path / mode).is_dir():
        return path / mode
    flat = sorted(path.glob(f"{mode}__*"))
    if not flat:
        raise FileNotFoundError(f"no {mode} index under {path}")
    target = Path(tempfile.mkdtemp(prefix=f"analog_{mode}_"))
    for f in flat:
        (target / f.name.split("__", 1)[1]).symlink_to(f.resolve())
    return target


def tanimoto(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Pairwise Tanimoto between packed fingerprints ``a`` [n, B] and ``b`` [m, B]."""
    a64 = np.ascontiguousarray(a).view(np.uint64)
    b64 = np.ascontiguousarray(b).view(np.uint64)
    inter = np.bitwise_count(a64[:, None, :] & b64[None, :, :]).sum(axis=2, dtype=np.int64)
    na = np.bitwise_count(a64).sum(axis=1, dtype=np.int64)
    nb = np.bitwise_count(b64).sum(axis=1, dtype=np.int64)
    union = na[:, None] + nb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def score_molecule(analogs: Sequence[tuple[NDArray[np.int64], NDArray[np.float64]]], candidate_fps: NDArray[np.uint8],
                   row_fp: Mapping[int, int], fps: NDArray[np.uint8], powers: Sequence[float] = (3.0,),
                   top_k: int | None = None, candidate_rows: NDArray[np.int64] | None = None,
                   self_min_sim: float = 0.0) -> dict[float, NDArray[np.float64]]:
    """Per power p: matrix [n_spectra, n_candidates] of max_a sim^p × Tanimoto(candidate, analog).

    ``analogs`` holds (library rows, similarities) per query spectrum; ``row_fp`` maps a
    library row to its structure's fingerprint row in ``fps`` (rows without one are skipped).
    With ``candidate_rows`` (the candidates' rows in ``fps``) a candidate's own spectra count only
    where their similarity is at least ``self_min_sim``.
    """
    n_q, n_c = len(analogs), len(candidate_fps)
    out = {p: np.zeros((n_q, n_c)) for p in powers}
    used = [(i, r, s) for i, (rows, sims) in enumerate(analogs)
            for r, s in zip(rows[:top_k], sims[:top_k]) if int(r) in row_fp]
    if not used or not n_c:
        return out
    fp_rows = sorted({row_fp[int(r)] for _, r, _ in used})
    col = {f: j for j, f in enumerate(fp_rows)}
    t = tanimoto(candidate_fps, fps[fp_rows])  # [n_c, n_analog_structures]
    own = None
    if candidate_rows is not None and self_min_sim > 0:
        own = np.asarray(candidate_rows)[:, None] == np.asarray(fp_rows)[None, :]
        t_own, t = np.where(own, t, 0.0), np.where(own, 0.0, t)
    for p in powers:
        w = np.zeros((n_q, len(fp_rows)))
        for i, r, s in used:
            j = col[row_fp[int(r)]]
            w[i, j] = max(w[i, j], s ** p)
        out[p] = (w[:, None, :] * t[None, :, :]).max(axis=2)
        if own is not None and self_min_sim <= 1:
            w_own = np.where(w >= self_min_sim ** p, w, 0.0)
            out[p] = np.maximum(out[p], (w_own[:, None, :] * t_own[None, :, :]).max(axis=2))
    return out


def aggregate(per_spectrum: NDArray[np.float64], how: str) -> NDArray[np.float64]:
    """Molecule-level candidate scores from the per-spectrum matrix."""
    if per_spectrum.size == 0:
        return np.zeros(per_spectrum.shape[1] if per_spectrum.ndim == 2 else 0)
    return per_spectrum.max(axis=0) if how == "max" else per_spectrum.mean(axis=0)


def rank_candidates(keys: Sequence[str], scores: NDArray[np.float64], k: int = 25) -> list[str]:
    """Top-``k`` keys by score (positive only), ties broken by key."""
    order = sorted((i for i in range(len(keys)) if scores[i] > 0), key=lambda i: (-scores[i], keys[i]))
    return [keys[i] for i in order[:k]]


def library_row_keys(library: Path) -> pd.Series:
    """Scorer key per library row (index = train row)."""
    t = pq.read_table(library, columns=["row", "key"]).to_pandas()
    return t.set_index("row")["key"]
