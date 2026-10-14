"""Training arrays for the fingerprint model (``scripts/train_fp.py``) → ``outputs/fp_data/``.

- ``pool_fp.npy``: packed casmi6311 fingerprints of every pool structure (pool row order);
  ``pool_ok.npy`` marks the ones RDKit could parse.
- Spectra of the cleaned library, prepared with ``casmi.fpmodel.prep_peaks`` (m/z rounded to
  0.01 Da, ≤ 256 peaks, sqrt intensity): ``mz.npy`` / ``it.npy`` (float32 / float16, concatenated),
  ``off.npy``, and per spectrum ``prec``, ``adduct``, ``instr``, ``ce`` (NaN = unknown), ``mode``,
  ``struct`` (pool row of its structure; -1 outside the pool's mass range), ``row`` (train row).
- ``allowed_<name>.npy`` per split family: spectra that are library rows in every split of the
  family (``v2`` = v2_seed0/1/2, so no validation molecule of any seed is trained on; ``rank`` adds
  the ranker-training splits r_seed10–14).
- ``decoys.npy`` [n_struct, 64] with ``decoy_struct.npy``: per training structure, pool rows in its
  ±10 ppm window (nearest masses first when the window has fewer than 64).

    python scripts/build_fp_data.py [--workers 24]
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.candidates import CandidatePool
from casmi.channels.fingerprint import descriptors
from casmi.config import WORK_DIR
from casmi.data import mean_ce
from casmi.fpmodel import ADDUCT_IX, instr_family, prep_peaks
from casmi.harness import CLEAN_LIBRARY

OUT = WORK_DIR / "fp_data"
V2 = ["v2_seed0", "v2_seed1", "v2_seed2"]
# "rank": also the ranker-training splits, so the fingerprint scores there are out-of-sample too
SPLIT_FAMILIES = {"v2": V2, "rank": V2 + [f"r_seed{s}" for s in range(10, 15)]}
N_DECOYS = 64
COLS = ["row", "key", "precursor_mz", "adduct", "instrument_type", "collision_energy_ev", "ionization_mode",
        "ms2_mzs", "ms2_normalized_intensities"]


def _row_group(i: int) -> dict:
    t = pq.ParquetFile(CLEAN_LIBRARY).read_row_group(i, columns=COLS).to_pandas()
    peaks = [prep_peaks(mz, it) for mz, it in zip(t["ms2_mzs"], t["ms2_normalized_intensities"])]
    return {"row": t["row"].to_numpy(np.int64), "key": t["key"].to_numpy(),
            "prec": t["precursor_mz"].to_numpy(np.float32),
            "adduct": np.array([ADDUCT_IX.get(a, ADDUCT_IX["<unk>"]) for a in t["adduct"]], np.int16),
            "instr": np.array([instr_family(s) for s in t["instrument_type"]], np.int8),
            "ce": np.array([mean_ce(v) for v in t["collision_energy_ev"]], np.float32),
            "mode": np.where(t["ionization_mode"] == "positive", 1, -1).astype(np.int8),
            "n": np.array([len(p[0]) for p in peaks], np.int64),
            "mz": np.concatenate([p[0] for p in peaks]) if peaks else np.zeros(0, np.float32),
            "it": np.concatenate([p[1] for p in peaks]).astype(np.float16) if peaks else np.zeros(0, np.float16)}


def decoy_table(masses: np.ndarray, structs: np.ndarray, ppm: float = 10.0, k: int = N_DECOYS) -> np.ndarray:
    """[len(structs), k] pool rows: the ±ppm window around each struct (itself excluded), then nearest masses."""
    out = np.empty((len(structs), k), np.int32)
    n = len(masses)
    for j, s in enumerate(structs):
        m = masses[s]
        lo = np.searchsorted(masses, m * (1 - ppm * 1e-6), side="left")
        hi = np.searchsorted(masses, m * (1 + ppm * 1e-6), side="right")
        win = np.arange(lo, hi)
        win = win[win != s]
        if len(win) >= k:
            near = win[np.argsort(np.abs(masses[win] - m), kind="stable")[:k]]
        else:
            a, b = max(0, s - k), min(n, s + k + 1)
            cand = np.arange(a, b)
            cand = cand[cand != s]
            near = cand[np.argsort(np.abs(masses[cand] - m), kind="stable")[:k]]
        out[j] = near
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    pool = CandidatePool.load()
    if not (OUT / "pool_fp.npy").exists() or len(np.load(OUT / "pool_ok.npy")) != len(pool):
        packed = descriptors(pool.meta["smiles"].tolist(), "casmi6311", workers=args.workers)
        width = next(len(p) for p in packed if p is not None)
        ok = np.array([p is not None for p in packed])
        fp = np.zeros((len(packed), width), np.uint8)
        fp[ok] = np.stack([p for p in packed if p is not None])
        np.save(OUT / "pool_fp.npy", fp)
        np.save(OUT / "pool_ok.npy", ok)
        print(f"pool fingerprints: {len(fp):,} ({(~ok).sum()} unparsable), {time.time() - t0:.0f}s", flush=True)

    n_groups = pq.ParquetFile(CLEAN_LIBRARY).num_row_groups
    with Pool(args.workers) as p:
        parts = p.map(_row_group, range(n_groups), chunksize=1)
    cat = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    key_row = pd.Series(np.arange(len(pool.meta)), index=pool.meta["key"])
    # structures outside the pool's mass range (100–1300 Da) get -1 and are not trained on
    cat["struct"] = key_row.reindex(cat.pop("key")).fillna(-1).to_numpy().astype(np.int32)
    print(f"spectra without a pool structure: {(cat['struct'] < 0).sum():,}", flush=True)
    cat["off"] = np.concatenate([[0], np.cumsum(cat.pop("n"))]).astype(np.int64)
    for name, arr in cat.items():
        np.save(OUT / f"{name}.npy", arr)
    print(f"spectra: {len(cat['row']):,}, peaks {len(cat['mz']):,} (mean {len(cat['mz']) / len(cat['row']):.1f}), "
          f"{time.time() - t0:.0f}s", flush=True)

    for family, splits in SPLIT_FAMILIES.items():
        allowed = np.ones(len(cat["row"]), bool)
        for split in splits:
            mask = pd.read_parquet(WORK_DIR / "splits" / split / "library_mask.parquet", columns=["row", "in_library"])
            allowed &= pd.Series(mask["in_library"].to_numpy(), index=mask["row"]).reindex(cat["row"]).fillna(
                False).to_numpy(bool)
        np.save(OUT / f"allowed_{family}.npy", allowed)
        print(f"allowed_{family}: {allowed.sum():,} spectra, {len(np.unique(cat['struct'][allowed])):,} structures")

    structs = np.unique(cat["struct"][cat["struct"] >= 0])
    np.save(OUT / "decoy_struct.npy", structs)
    np.save(OUT / "decoys.npy", decoy_table(pool.masses, structs))
    print(f"decoys for {len(structs):,} structures; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
