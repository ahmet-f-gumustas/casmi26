"""Class 2 failure clusters of the analog-only ranking (``scripts/error_analysis.py``).

Per Class 2 molecule: the answer's rank in the full analog ranking, ``t_top1`` (Tanimoto of the wrong
top-1 to the answer), ``t_analog10`` (best Tanimoto of the answer to the top-10 spectral analogs of
any spectrum), ``sim1`` (best analog spectral similarity) and query-side descriptors.

Failures (answer not at rank 1) are put in two clusters and a remainder:

- (a) near-isomer confusion: ``t_top1`` > 0.8;
- (b) retrieval failure: ``t_top1`` < 0.5 and ``t_analog10`` < 0.5 (no retrieved analog resembles the
  answer), split by ``nn_tanimoto``: a library analog exists (≥ 0.5) but was not retrieved (b1), or
  none exists (b2);
- (c) intermediate: the rest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from casmi.channels.analog import tanimoto
from casmi.config import TOP_K
from casmi.data import mean_ce
from casmi.ranker import combine, rank

CLOSE, FAR, ANALOG_FAR, NN_EXISTS = 0.8, 0.5, 0.5, 0.5
HIT = "hit (rank 1)"
CLUSTERS = [HIT, "(a) near-isomer", "(b1) retrieval: analog exists, not retrieved",
            "(b2) retrieval: no library analog", "(c) intermediate"]


def cluster(df: pd.DataFrame) -> pd.Series:
    """Cluster label per row of ``c2_error_rows`` output."""
    out = pd.Series(HIT, index=df.index)
    miss = df["rank"] != 1
    near = miss & (df["t_top1"] > CLOSE)
    far = miss & ~near & (df["t_top1"].fillna(0) < FAR) & (df["t_analog10"] < ANALOG_FAR)
    out[near] = CLUSTERS[1]
    out[far & (df["nn_tanimoto"] >= NN_EXISTS)] = CLUSTERS[2]
    out[far & (df["nn_tanimoto"] < NN_EXISTS)] = CLUSTERS[3]
    out[miss & ~near & ~far] = CLUSTERS[4]
    return out


def c2_error_rows(split: str, val: pd.DataFrame, queries: pd.DataFrame, windows: Mapping[str, list[str]],
                  scores: Mapping[str, Mapping[str, float]], hits: Sequence, pool_meta: pd.DataFrame,
                  fps: np.ndarray, row_key: Mapping[int, str], row_tims: Mapping[int, bool]) -> pd.DataFrame:
    """One row per Class 2 molecule of ``val`` with the cluster inputs and label.

    ``queries`` must be in the row order ``hits`` (``casmi.pipeline.analog_hits``) was computed in;
    ``scores`` are the analog channel's; ``val["nn_tanimoto"]`` decides b1 / b2.
    """
    queries = queries.assign(n_peaks=queries["ms2_mzs"].map(len))
    fp_row = dict(zip(pool_meta["key"], range(len(pool_meta))))
    rows = []
    c2 = val[val["class"] == 2]
    for m, answer, src, nn in zip(c2["molecule_id"], c2["inchikey14"], c2["source"], c2["nn_tanimoto"]):
        keys = windows.get(m, [])
        s = scores.get(m, {})
        ranked = rank(keys, combine(keys, {"analog": s}, {"analog": 1.0}), s, k=len(keys))
        r = ranked.index(answer) + 1 if answer in ranked else 0
        a_fp = fps[[fp_row[answer]]] if answer in fp_row else None
        t_cand = dict(zip(keys, tanimoto(a_fp, fps[[fp_row[k] for k in keys]])[0])) if a_fp is not None and keys else {}
        wrong = [k for k in ranked if k != answer]
        idx = queries.index[queries["molecule_id"] == m]
        spec_hits = [hits[i] for i in idx]
        best = max(((s_[0], r_[0]) for r_, s_ in spec_hits if len(r_)), default=(0.0, None))
        top10 = {int(x) for r_, _ in spec_hits for x in r_[:10]}
        tims10 = float(np.mean([row_tims.get(x, False) for x in top10])) if top10 else np.nan
        an_rows = [fp_row[row_key[x]] for x in top10 if row_key.get(x) in fp_row]
        t_an = tanimoto(a_fp, fps[sorted(set(an_rows))])[0] if a_fp is not None and an_rows else np.zeros(0)
        b_key = row_key.get(best[1]) if best[1] is not None else None
        q = queries.loc[idx]
        ces = [c for c in map(mean_ce, q["collision_energy_ev"]) if np.isfinite(c)]
        rows.append({
            "split": split, "molecule_id": m, "source": src, "nn_tanimoto": nn,
            "rank": r, "rr": 1.0 / r if 0 < r <= TOP_K else 0.0,
            "n_candidates": len(keys), "n_scored": len(ranked),
            "n_close": sum(v > CLOSE for k, v in t_cand.items() if k != answer),
            "t_top1": t_cand.get(ranked[0]) if ranked and ranked[0] != answer else np.nan,
            "t_wrong1": t_cand.get(wrong[0]) if wrong else np.nan,
            "sim1": float(best[0]),
            "t_analog1": float(tanimoto(a_fp, fps[[fp_row[b_key]]])[0, 0]) if a_fp is not None and b_key in fp_row
            else np.nan,
            "t_analog10": float(t_an.max()) if len(t_an) else 0.0, "tims_share10": tims10,
            "n_spectra": len(idx), "ce": float(np.mean(ces)) if len(ces) else np.nan, "n_peaks": q["n_peaks"].mean(),
            "instrument": q["instrument_type"].mode().iat[0] if q["instrument_type"].notna().any() else "?",
        })
    df = pd.DataFrame(rows)
    df["cluster"] = cluster(df) if len(df) else pd.Series(dtype=str)
    return df


def row_instruments(library) -> dict[int, bool]:
    """Train row → is the spectrum timsTOF."""
    inst = pd.read_parquet(library, columns=["row", "instrument_type"])
    return dict(zip(inst["row"], inst["instrument_type"].fillna("").str.contains("timsTOF", case=False)))
