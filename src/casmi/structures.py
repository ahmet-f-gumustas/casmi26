"""Per-structure scorer keys and masses, cached for the train SMILES."""

from __future__ import annotations

from collections.abc import Iterable
from multiprocessing import Pool

import numpy as np
import pandas as pd

from casmi.chem import exact_mass
from casmi.config import WORK_DIR
from casmi.metric import scorer_key

CACHE = WORK_DIR / "structure_keys.parquet"


def _key_and_mass(smiles: str) -> tuple[str | None, float | None]:
    return scorer_key(smiles), exact_mass(smiles)


def structure_table(smiles: Iterable[str], workers: int = 24) -> pd.DataFrame:
    """``normalized_smiles, key, exact_mass`` per unique SMILES (scorer key = official metric key).

    Cached in ``WORK_DIR/structure_keys.parquet``; SMILES missing from the cache are computed and added.
    """
    uniq = sorted({s for s in smiles if isinstance(s, str)})
    cached = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["normalized_smiles", "key", "exact_mass"])
    todo = sorted(set(uniq) - set(cached["normalized_smiles"]))
    if todo:
        with Pool(workers) as pool:
            res = pool.map(_key_and_mass, todo, chunksize=200)
        new = pd.DataFrame({"normalized_smiles": todo, "key": [k for k, _ in res],
                            "exact_mass": np.array([m for _, m in res], dtype=np.float64)})
        cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(CACHE, index=False)
    return cached[cached["normalized_smiles"].isin(uniq)].reset_index(drop=True)
