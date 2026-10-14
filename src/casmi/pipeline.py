"""End-to-end ranking pipeline shared by local validation and the Kaggle notebook.

queries (one row per spectrum, with ``molecule_id``) → neutral mass per molecule →
candidate structures from the pool's ppm window → channel scores → top-25 SMILES.
Validation passes its split's queries, library mask and Class-3 exclusions; the
Kaggle notebook passes ``test.parquet`` and the full cleaned library.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from casmi.candidates import CandidatePool, molecule_masses
from casmi.channels.library import LibraryConfig, Spectrum, prepare, rank, score_molecule
from casmi.clean import strip_peaks

QUERY_COLUMNS = ["molecule_id", "ionization_mode", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
LIBRARY_COLUMNS = ["row", "key", "ionization_mode", "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"]


@dataclass(frozen=True)
class PipelineConfig:
    ppm: float = 10.0
    offset_ppm: float = 2.0
    aggregation: str = "mean"
    workers: int = max(1, len(os.sched_getaffinity(0)))  # CPUs this process may use (container-aware)
    library: LibraryConfig = LibraryConfig()


def clean_queries(queries: pd.DataFrame) -> pd.DataFrame:
    """Apply the test-side peak rule (drop peaks above precursor + 2 Da) to query spectra."""
    lengths = queries["ms2_mzs"].map(len).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    mz = np.concatenate([np.asarray(a, dtype=np.float64) for a in queries["ms2_mzs"]]) if lengths.sum() else np.zeros(0)
    it = (np.concatenate([np.asarray(a, dtype=np.float64) for a in queries["ms2_normalized_intensities"]])
          if lengths.sum() else np.zeros(0))
    new_off, new_mz, new_it, _ = strip_peaks(offsets, mz, it, queries["precursor_mz"].to_numpy(np.float64))
    out = queries.copy()
    out["ms2_mzs"] = [new_mz[a:b] for a, b in zip(new_off[:-1], new_off[1:])]
    out["ms2_normalized_intensities"] = [new_it[a:b] for a, b in zip(new_off[:-1], new_off[1:])]
    return out


def to_spectra(df: pd.DataFrame, cfg: LibraryConfig) -> list[Spectrum]:
    return [(mode, prepare(np.asarray(mz), np.asarray(it), prec, cfg))
            for mode, mz, it, prec in zip(df["ionization_mode"], df["ms2_mzs"], df["ms2_normalized_intensities"],
                                          df["precursor_mz"])]


def candidate_windows(queries: pd.DataFrame, pool: CandidatePool, cfg: PipelineConfig) -> dict[str, list[str]]:
    """Scorer keys of the pool structures in each molecule's mass window."""
    keys = pool.meta["key"].to_numpy()
    masses = molecule_masses(queries)
    return {m: keys[pool.window(masses[m], cfg.ppm, cfg.offset_ppm)].tolist() for m in masses.index}


def load_references(library: Path, keys: Iterable[str], in_library: pd.Series | None = None,
                    cfg: LibraryConfig = LibraryConfig()) -> dict[str, list[Spectrum]]:
    """Prepared reference spectra per scorer key from the cleaned library parquet.

    ``in_library`` (indexed by train ``row``) restricts the library to a split's mask.
    """
    table = ds.dataset(library).to_table(columns=LIBRARY_COLUMNS, filter=pc.field("key").isin(list(set(keys))))
    lib = table.to_pandas()
    if in_library is not None:
        lib = lib[lib["row"].map(in_library).fillna(False).to_numpy(dtype=bool)]
    return {key: to_spectra(grp, cfg) for key, grp in lib.groupby("key")}


# Worker state, inherited through fork.
_STATE: dict = {}


def _score(mol_id: str) -> tuple[str, pd.DataFrame]:
    s = _STATE
    return mol_id, score_molecule(s["queries"].get(mol_id, []), s["windows"].get(mol_id, []), s["refs"], s["cfg"])


def library_scores(queries: pd.DataFrame, windows: Mapping[str, list[str]], refs: Mapping[str, list[Spectrum]],
                   cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Library-channel score table per molecule."""
    _STATE.update(queries={m: to_spectra(g, cfg.library) for m, g in queries.groupby("molecule_id")},
                  windows=windows, refs=refs, cfg=cfg.library)
    ids = list(windows)
    if cfg.workers > 1:
        with Pool(cfg.workers) as pool:
            out = dict(pool.map(_score, ids, chunksize=1))
    else:
        out = dict(map(_score, ids))
    _STATE.clear()
    return out


def predictions(scores: Mapping[str, pd.DataFrame], smiles_of: Mapping[str, str], by: str = "mean") -> dict[str, list[str]]:
    """Top-25 SMILES per molecule from its score table."""
    return {m: [smiles_of[k] for k in rank(s, by)] for m, s in scores.items()}


def run_library_pipeline(queries: pd.DataFrame, pool: CandidatePool, library: Path, cfg: PipelineConfig = PipelineConfig(),
                         in_library: pd.Series | None = None) -> tuple[dict[str, list[str]], dict[str, pd.DataFrame]]:
    """Library-only pipeline: returns ``(predictions, score tables)``."""
    queries = clean_queries(queries)
    windows = candidate_windows(queries, pool, cfg)
    refs = load_references(library, {k for ks in windows.values() for k in ks}, in_library, cfg.library)
    scores = library_scores(queries, windows, refs, cfg)
    smiles_of = dict(zip(pool.meta["key"], pool.meta["smiles"]))
    return predictions(scores, smiles_of, cfg.aggregation), scores


# --------------------------------------------------------------------------------------
# Analog and fragment channels, and the ranker-v0 combination.
# --------------------------------------------------------------------------------------
ChannelScores = dict[str, dict[str, float]]  # molecule_id -> {key: score}


def _analog_search(i: int) -> tuple[np.ndarray, np.ndarray]:
    s = _STATE
    mode, prec, peaks = s["raw"][i]
    return s["index"].search(mode, prec, peaks, s["cfg"], s["allowed"])


def analog_channel(queries: pd.DataFrame, windows: Mapping[str, list[str]], pool: CandidatePool, index,
                   row_key: Mapping[int, str], cfg=None, allowed: np.ndarray | None = None,
                   workers: int = 1) -> ChannelScores:
    """Mass-shifted analog propagation scores (see ``casmi.channels.analog``), mean over spectra."""
    from casmi.channels.analog import AnalogConfig, score_molecule as analog_score

    cfg = cfg or AnalogConfig()
    queries = queries.reset_index(drop=True)
    raw = [(m, float(p), np.column_stack([mz, it]).astype(np.float32))
           for m, p, mz, it in zip(queries["ionization_mode"], queries["precursor_mz"], queries["ms2_mzs"],
                                   queries["ms2_normalized_intensities"])]
    _STATE.update(raw=raw, index=index, cfg=cfg, allowed=allowed)
    if workers > 1:
        with Pool(workers) as p:
            hits = p.map(_analog_search, range(len(raw)), chunksize=4)
    else:
        hits = [_analog_search(i) for i in range(len(raw))]
    _STATE.clear()
    fp_row_of_key = dict(zip(pool.meta["key"], range(len(pool.meta))))
    used = {int(r) for rows, _ in hits for r in rows}
    row_fp = {r: fp_row_of_key[row_key[r]] for r in used if row_key.get(r) in fp_row_of_key}
    fps = pool.fps
    out: ChannelScores = {}
    for mol_id, idx in queries.groupby("molecule_id").indices.items():
        keys = windows.get(mol_id, [])
        if not keys:
            continue
        cand = fps[[fp_row_of_key[k] for k in keys]]
        per = analog_score([hits[i] for i in idx], cand, row_fp, fps, (cfg.power,), top_k=cfg.top_k)[cfg.power]
        out[mol_id] = dict(zip(keys, per.mean(axis=0).tolist()))
    return out


def _fragments(smiles: str):
    from casmi.channels.fragment import fragment_masses

    return fragment_masses(smiles, _STATE["fcfg"])


def fragment_channel(queries: pd.DataFrame, windows: Mapping[str, list[str]], smiles_of: Mapping[str, str],
                     cfg=None, workers: int = 1, cache: dict | None = None) -> ChannelScores:
    """Explained-intensity scores from 1–2 bond-break fragments, mean over spectra."""
    from casmi.channels.fragment import FragmentConfig, explained_intensity, ion_masses

    cfg = cfg or FragmentConfig()
    cache = {} if cache is None else cache
    todo = sorted({k for ks in windows.values() for k in ks} - set(cache))
    _STATE.update(fcfg=cfg)
    if todo:
        if workers > 1:
            with Pool(workers) as p:
                frags = p.map(_fragments, [smiles_of[k] for k in todo], chunksize=50)
        else:
            frags = [_fragments(smiles_of[k]) for k in todo]
        cache.update(zip(todo, frags))
    _STATE.clear()
    out: ChannelScores = {}
    for mol_id, grp in queries.groupby("molecule_id"):
        keys = windows.get(mol_id, [])
        specs = to_spectra(grp, LibraryConfig())
        if not keys or not specs:
            continue
        mat = np.array([[explained_intensity(ion_masses(cache[k], mode, cfg), peaks, cfg.tol_da) for k in keys]
                        for mode, peaks in specs])
        out[mol_id] = dict(zip(keys, mat.mean(axis=0).tolist()))
    return out


def library_channel(queries: pd.DataFrame, windows: Mapping[str, list[str]], library: Path, cfg: PipelineConfig,
                    in_library: pd.Series | None = None) -> ChannelScores:
    """Library-channel scores (aggregation from ``cfg``) per molecule."""
    refs = load_references(library, {k for ks in windows.values() for k in ks}, in_library, cfg.library)
    scores = library_scores(queries, windows, refs, cfg)
    return {m: dict(zip(s["key"], s[cfg.aggregation].astype(float))) for m, s in scores.items() if len(s)}


def run_combined_pipeline(queries: pd.DataFrame, pool: CandidatePool, library: Path, index, row_key: Mapping[int, str],
                          weights: Mapping[str, float], cfg: PipelineConfig = PipelineConfig(),
                          in_library: pd.Series | None = None, allowed: np.ndarray | None = None,
                          ) -> tuple[dict[str, list[str]], dict[str, ChannelScores]]:
    """Library + analog + fragment channels combined by ranker v0."""
    from casmi.ranker import combine, rank as rank_keys

    queries = clean_queries(queries)
    windows = candidate_windows(queries, pool, cfg)
    smiles_of = dict(zip(pool.meta["key"], pool.meta["smiles"]))
    channels = {
        "library": library_channel(queries, windows, library, cfg, in_library) if weights.get("library") else {},
        "analog": analog_channel(queries, windows, pool, index, row_key, allowed=allowed, workers=cfg.workers)
        if weights.get("analog") else {},
        "fragment": fragment_channel(queries, windows, smiles_of, workers=cfg.workers) if weights.get("fragment") else {},
    }
    preds = {}
    for mol_id, keys in windows.items():
        per_channel = {c: channels[c].get(mol_id, {}) for c in channels}
        total = combine(keys, per_channel, weights)
        preds[mol_id] = [smiles_of[k] for k in rank_keys(keys, total, per_channel.get("analog"))]
    return preds, channels
