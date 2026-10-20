"""The 31 V33 ranker features (``casmi.rank31``) on our simulated splits → ``outputs/rank31_features/``.

Every (molecule, candidate) of the split's validation molecules, with ``label`` (the candidate is the
answer), ``class`` and ``source``. The fingerprint model must not have seen the splits' held-out
molecules (FP v2 epoch 5, family ``rank``, for the r3 / v3 splits). ``--rows`` also writes the training
rows in ``rank_train.npz`` format (X in ``FEATURES`` order, Y, M = 0 for Class 1 / 1 for Class 2, G per
molecule; molecules without the answer in the window and Class 3 left out) from the splits given there.

    python validation/rank31_features.py --model outputs/fp_models/np_v2_e5 \\
        --splits r3_seed10 r3_seed11 r3_seed12 r3_seed13 r3_seed14 v3_seed0 v3_seed1 v3_seed2 \\
        --rows r3_seed10 r3_seed11 r3_seed12 r3_seed13 r3_seed14
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import rank31
from casmi.channels.analog import AnalogConfig, AnalogIndex, library_row_keys
from casmi.config import WORK_DIR
from casmi.fpmodel import CasmiBundle
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val
from casmi.pipeline import PipelineConfig, analog_hits, candidate_windows, clean_queries, fingerprint_logits, \
    library_channel

OUT = WORK_DIR / "rank31_features"


def split_frame(split: str, model: CasmiBundle, index: AnalogIndex, row_key: dict, caches: dict,
                workers: int) -> pd.DataFrame:
    data = load_val(WORK_DIR / "splits" / split)
    queries = clean_queries(data.queries).reset_index(drop=True)
    cfg = PipelineConfig(workers=workers)
    windows = candidate_windows(queries, data.pool, cfg)
    allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
    hits = analog_hits(queries, index, cfg=AnalogConfig(top_k=rank31.ANALOG_TOP_K), allowed=allowed, workers=workers)
    _, lib_max = library_channel(queries, windows, CLEAN_LIBRARY, cfg, data.in_library, both=True)
    logits = fingerprint_logits(queries, windows, model)
    df = rank31.frame(queries, windows, smiles_of, hits, row_key, lib_max, logits, model.layout, model.bits,
                      workers=workers, caches=caches)
    val = data.val.set_index("molecule_id")
    df["label"] = df["key"].to_numpy() == df["molecule_id"].map(val["inchikey14"]).to_numpy()
    df["class"] = df["molecule_id"].map(val["class"]).to_numpy()
    df["source"] = df["molecule_id"].map(val["source"]).to_numpy()
    df["split"] = split
    return df


def training_rows(frames: list[pd.DataFrame]) -> dict[str, np.ndarray]:
    """``rank_train.npz`` arrays from Class 1 / 2 molecules whose answer is in the window."""
    df = pd.concat(frames, ignore_index=True)
    df = df[df["class"].isin([1, 2])]
    df = df[df.groupby(["split", "molecule_id"])["label"].transform("any")]
    g = df.groupby(["split", "molecule_id"], sort=False).ngroup().to_numpy()
    return {"X": df[rank31.FEATURES].to_numpy(np.float32), "Y": df["label"].to_numpy(np.float64),
            "M": (df["class"] == 2).to_numpy(np.float64), "G": g.astype(np.int64)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--rows", nargs="*", default=[], help="splits whose rows form the training npz")
    ap.add_argument("--rows-out", type=Path, default=WORK_DIR / "rank31_rows" / "rank31_rows.npz")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    model = CasmiBundle.load(sorted(args.model.glob("fp_casmi*.pt")))
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    caches: dict = {}
    frames = {}
    for split in args.splits:
        df = split_frame(split, model, index, row_key, caches, args.workers)
        df.to_parquet(OUT / f"{split}.parquet", index=False)
        frames[split] = df
        hit = df.groupby("molecule_id")["label"].any()
        print(f"{split}: {df['molecule_id'].nunique()} molecules, {len(df):,} rows, answer in window {hit.mean():.1%} "
              f"({time.time() - t0:.0f}s)", flush=True)
    if args.rows:
        rows = training_rows([frames[s] for s in args.rows])
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.rows_out, **rows)
        print(f"training rows {args.rows_out}: {len(rows['Y']):,} rows, {len(np.unique(rows['G'])):,} groups, "
              f"Class 2 share {rows['M'].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
