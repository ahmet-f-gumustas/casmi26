"""Per-(molecule, candidate) features for the learned ranker (ranker v2) → ``outputs/ranker_features/<split>.parquet``.

Splits: the validation splits (v2_seed0–2) and the ranker-training splits (r_seed10–14). The
fingerprint model must not have seen any of their held-out molecules (``train_fp.py --family rank``).

Features per candidate (ranks: 1 = best within the molecule's window, ``*_pct``: rank-normalised):
analog score / rank / pct, fingerprint f·z / rank / pct and soft-Tanimoto, fragment score / pct,
library max / mean similarity, ``t_analog1`` (best Tanimoto to the top-1 spectral analog of any
spectrum), window size, molecule mass, PubChem popularity (log1p substances, log1p PubMed) and
source flags. ``in_library`` is computed from the split's library mask, not from the pool's
``in_train`` flag: a simulated Class 2 answer is a training structure whose spectra were removed,
so the pool flag would mark every answer; the masked flag mirrors the test, where a Class 2 answer
has no training spectra.

    python validation/ranker_features.py --model outputs/fp_models/rank_full [--splits ...] [--out <dir>]
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.candidates import POOL_DIR
from casmi.channels.analog import AnalogIndex, library_row_keys
from casmi.channels.fingerprint import fingerprint_channel
from casmi.config import WORK_DIR
from casmi.fpmodel import CasmiBundle
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, library_keys, load_val
from casmi.pipeline import (
    PipelineConfig,
    analog_channel,
    analog_hits,
    candidate_windows,
    clean_queries,
    fingerprint_logits,
    fragment_channel,
    library_channel,
)
from casmi.ranker_v2 import candidate_features

SPLITS = [f"v2_seed{s}" for s in range(3)] + [f"r_seed{s}" for s in range(10, 15)]
FRAG_CACHE = WORK_DIR / "fragment_cache.pkl"
OUT = WORK_DIR / "ranker_features"


def split_features(split: str, model: CasmiBundle, index: AnalogIndex, row_key: dict, pop: pd.DataFrame,
                   frag_cache: dict, workers: int) -> pd.DataFrame:
    data = load_val(WORK_DIR / "splits" / split)
    queries = clean_queries(data.queries).reset_index(drop=True)
    cfg = PipelineConfig(workers=workers)
    windows = candidate_windows(queries, data.pool, cfg)
    allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    meta = data.pool.meta
    smiles_of = dict(zip(meta["key"], meta["smiles"]))

    hits = analog_hits(queries, index, allowed=allowed, workers=workers)
    logits = fingerprint_logits(queries, windows, model)
    channels = {
        "analog": analog_channel(queries, windows, data.pool, index, row_key, allowed=allowed, hits=hits),
        "fingerprint": fingerprint_channel(logits, windows, smiles_of, model.layout, workers=workers),
        "fingerprint_st": fingerprint_channel(logits, windows, smiles_of, model.layout, workers=workers,
                                              how="soft_tanimoto"),
        "fragment": fragment_channel(queries, windows, smiles_of, workers=workers, cache=frag_cache),
    }
    channels["library"], channels["library_max"] = library_channel(queries, windows, CLEAN_LIBRARY, cfg,
                                                                   data.in_library, both=True)
    df = candidate_features(queries, windows, channels, hits, row_key, meta, data.pool.fps,
                            library_keys(data.in_library), pop)
    truth = dict(zip(data.val["molecule_id"], data.val["inchikey14"]))
    df["label"] = df["key"].to_numpy() == df["molecule_id"].map(truth).to_numpy()
    df["class"] = df["molecule_id"].map(dict(zip(data.val["molecule_id"], data.val["class"])))
    df["source"] = df["molecule_id"].map(dict(zip(data.val["molecule_id"], data.val["source"])))
    df["split"] = split
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--device", default=None, help="fingerprint model device (default: cuda if available)")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default: outputs/ranker_features)")
    args = ap.parse_args()
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    model = CasmiBundle.load(sorted(args.model.glob("fp_casmi*.pt")), device=args.device)
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    pop = pd.read_parquet(POOL_DIR / "popularity.parquet")
    frag_cache = pickle.loads(FRAG_CACHE.read_bytes()) if FRAG_CACHE.exists() else {}
    n_cache = len(frag_cache)
    for split in args.splits:
        df = split_features(split, model, index, row_key, pop, frag_cache, args.workers)
        df.to_parquet(args.out / f"{split}.parquet", index=False)
        hit = df.groupby("molecule_id")["label"].any()
        print(f"{split}: {df['molecule_id'].nunique()} molecules, {len(df):,} rows, answer in window "
              f"{hit.mean():.1%} ({time.time() - t0:.0f}s)", flush=True)
    if len(frag_cache) > n_cache:
        FRAG_CACHE.write_bytes(pickle.dumps(frag_cache))


if __name__ == "__main__":
    main()
