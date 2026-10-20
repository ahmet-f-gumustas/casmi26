"""Forward-model scores: the query spectra against spectra predicted for the ranker's top candidates.

Every candidate in a molecule's top ``TOP_K`` (by the ranker) gets one predicted spectrum per query spectrum:
the query's adduct, each of its collision energies (eV) and the instrument class of the query
(``instrument_of``: QTOF for timsTOF). A merged spectrum (20,40,60 eV) is compared with the sum of the
predictions at its energies, each normalised to unit total. Query and prediction both go through the library
channel's cleaning (``casmi.channels.library.prepare``: precursor removal, noise floor, centroiding) and are
compared by entropy similarity. A candidate's score is the mean over the molecule's scorable spectra, weight 4
for a query spectrum with at least 5 peaks after cleaning, else 1 (ms-pred's retrieval weighting).

Only adducts the model was trained on are scored (``SUPPORTED``: [M+H]+ and [M+Na]+ in MassSpecGym). A molecule
without such a spectrum keeps the ranker's order; so does a candidate the model cannot handle (nan). When the
time budget runs out, the molecules not reached keep the ranker's order and ``casmi.degraded`` records how many.

Fusion with the ranker: ``rrf`` re-orders the top ``TOP_K`` by reciprocal-rank fusion of the ranker's rank and
the forward rank; ``feature_columns`` gives the score as ranker features (nan outside the top ``TOP_K``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from ms_entropy import calculate_entropy_similarity
from numpy.typing import NDArray

from casmi import degraded
from casmi.channels.library import LibraryConfig, prepare

TOP_K = 50
RRF_K = 10
SUPPORTED = ("[M+H]+", "[M+Na]+")
MIN_PEAKS_FULL_WEIGHT = 5
FEATURES = ("fw", "fw_rank", "fw_minus_max")

Peaks = tuple[NDArray[np.float64], NDArray[np.float32]]


class Predictor(Protocol):
    def predict(self, smiles: Sequence[str], adduct: str, energies: Sequence[float], instrument: str = "QTOF",
                top: int = 100) -> list[list[Peaks] | None]: ...


def instrument_of(instrument_type: str | None) -> str:
    """The model's instrument class for a query instrument string: Orbitrap and FT (QFT, ITFT, Hybrid FT) →
    Orbitrap, everything else (timsTOF, QTOF, TOF, ion traps, QqQ) → QTOF, the two classes in its training data."""
    name = str(instrument_type or "").lower()
    return "Orbitrap" if "orbitrap" in name or name.endswith("ft") else "QTOF"


def query_spectra(queries: pd.DataFrame, supported: Sequence[str] = SUPPORTED,
                  cfg: LibraryConfig = LibraryConfig()) -> dict[str, list[dict]]:
    """Scorable spectra per molecule: adduct, energies, instrument class, cleaned peaks and weight."""
    out: dict[str, list[dict]] = {}
    for r in queries.itertuples(index=False):
        energies = [float(e) for e in np.atleast_1d(r.collision_energy_ev) if np.isfinite(e) and e > 0]
        if r.adduct not in supported or not energies:
            continue
        peaks = prepare(np.asarray(r.ms2_mzs, np.float64), np.asarray(r.ms2_normalized_intensities, np.float64),
                        float(r.precursor_mz), cfg)
        if not len(peaks):
            continue
        out.setdefault(r.molecule_id, []).append({
            "adduct": r.adduct, "energies": tuple(energies), "instrument": instrument_of(r.instrument_type),
            "precursor_mz": float(r.precursor_mz), "peaks": peaks,
            "weight": 4.0 if len(peaks) >= MIN_PEAKS_FULL_WEIGHT else 1.0})
    return out


def _merge(spectra: Sequence[Peaks]) -> NDArray[np.float64]:
    parts = [np.column_stack([mz, it / it.sum()]) for mz, it in spectra if len(mz) and it.sum() > 0]
    return np.concatenate(parts) if parts else np.zeros((0, 2))


def forward_scores(model: Predictor, queries: pd.DataFrame, top: Mapping[str, Sequence[str]],
                   smiles_of: Mapping[str, str], order: Sequence[str] | None = None, budget_s: float | None = None,
                   chunk: int = 16, cache: dict | None = None, supported: Sequence[str] = SUPPORTED,
                   cfg: LibraryConfig = LibraryConfig()) -> tuple[dict[tuple[str, str], float], dict]:
    """``{(molecule_id, key): score}`` for the candidates in ``top`` (molecule → keys), and run counts.

    Molecules are scored in ``order`` (default: ``top``'s order) in chunks of ``chunk``; predictions are
    batched per (adduct, energy, instrument) over a chunk and kept in ``cache`` by (key, adduct, energy,
    instrument). With ``budget_s``, no new chunk starts once the budget is spent.
    """
    t0 = time.time()
    cache = {} if cache is None else cache
    spectra = query_spectra(queries, supported, cfg)
    mols = [m for m in (order if order is not None else list(top)) if top.get(m) and spectra.get(m)]
    scores: dict[tuple[str, str], float] = {}
    info = {"molecules": len(mols), "scored": 0, "unscorable": len([m for m in top if not spectra.get(m)]),
            "predictions": 0, "failed": 0}
    for start in range(0, len(mols), chunk):
        if budget_s is not None and time.time() - t0 > budget_s:
            left = len(mols) - start
            degraded.note("forward model out of time: molecules left in ranker order", left)
            print(f"DEGRADED: forward model budget {budget_s:.0f}s spent after {start} molecules; "
                  f"{left} keep the ranker's order", flush=True)
            break
        part = mols[start:start + chunk]
        need: dict[tuple[str, float, str], set[str]] = {}
        for m in part:
            for s in spectra[m]:
                for e in s["energies"]:
                    keys = {k for k in top[m] if (k, s["adduct"], e, s["instrument"]) not in cache}
                    need.setdefault((s["adduct"], e, s["instrument"]), set()).update(keys)
        for (adduct, e, inst), keys in need.items():
            if not keys:
                continue
            keys = sorted(keys)
            preds = model.predict([smiles_of.get(k, "") for k in keys], adduct, [e], instrument=inst)
            for k, p in zip(keys, preds):
                cache[(k, adduct, e, inst)] = None if p is None else p[0]
            info["predictions"] += len(keys)
            info["failed"] += sum(p is None for p in preds)
        for m in part:
            for k in top[m]:
                num = den = 0.0
                for s in spectra[m]:
                    pred = [cache[(k, s["adduct"], e, s["instrument"])] for e in s["energies"]]
                    if any(p is None for p in pred):
                        continue
                    merged = _merge(pred)
                    ref = prepare(merged[:, 0], merged[:, 1], s["precursor_mz"], cfg) if len(merged) else merged
                    sim = (calculate_entropy_similarity(s["peaks"], ref.astype(np.float32),
                                                        ms2_tolerance_in_da=cfg.ms2_tol_da, clean_spectra=False)
                           if len(ref) else 0.0)
                    num, den = num + s["weight"] * sim, den + s["weight"]
                if den > 0:
                    scores[(m, k)] = num / den
        info["scored"] += len(part)
    info["seconds"] = round(time.time() - t0, 1)
    return scores, info


def head(molecule_id: NDArray, score: NDArray[np.float64]) -> NDArray[np.int64]:
    """1-based rank of every row by ``score`` within its molecule (ties: row order)."""
    d = pd.DataFrame({"m": molecule_id, "s": -np.asarray(score, np.float64)})
    return d.groupby("m", sort=False)["s"].rank(method="first").to_numpy(np.int64)


def top_keys(molecule_id: NDArray, keys: NDArray, score: NDArray[np.float64], k: int = TOP_K) -> dict[str, list[str]]:
    """Molecule → its top-``k`` keys by ``score``, best first."""
    r = head(molecule_id, score)
    d = pd.DataFrame({"m": molecule_id, "key": keys, "r": r})[r <= k].sort_values(["m", "r"])
    return d.groupby("m", sort=False)["key"].agg(list).to_dict()


def column(molecule_id: NDArray, keys: NDArray, scores: Mapping[tuple[str, str], float]) -> NDArray[np.float64]:
    """The forward score per row (nan without one)."""
    return np.array([scores.get((m, k), np.nan) for m, k in zip(molecule_id, keys)], np.float64)


def rrf(molecule_id: NDArray, base: NDArray[np.float64], fws: Sequence[NDArray[np.float64]], k: int = RRF_K,
        top: int = TOP_K) -> NDArray[np.float64]:
    """Final score: the top ``top`` by ``base`` re-ordered by 1/(k + base rank) + Σ 1/(k + forward rank).

    Each forward rank counts among the head rows with that model's score; a head row without one takes its base
    rank in that term. Rows below the head keep the base order, under every head row.
    """
    rb = head(molecule_id, base)
    in_head = rb <= top
    fused = 1.0 / (k + rb)
    for fw in fws:
        fw_head = np.where(in_head & np.isfinite(fw), fw, -np.inf)
        rf = np.where(np.isfinite(fw_head), head(molecule_id, fw_head), rb)
        fused = fused + 1.0 / (k + rf)
    return np.where(in_head, 2.0 + fused, 1.0 / (1.0 + rb))


def feature_columns(molecule_id: NDArray, fw: NDArray[np.float64]) -> pd.DataFrame:
    """``FEATURES``: the score, its rank among the molecule's scored rows and its gap to their best (nan where
    there is no score)."""
    d = pd.DataFrame({"m": molecule_id, "fw": fw})
    g = d.groupby("m", sort=False)["fw"]
    return pd.DataFrame({"fw": fw, "fw_rank": g.rank(ascending=False, method="min").to_numpy(),
                         "fw_minus_max": fw - g.transform("max").to_numpy()})
