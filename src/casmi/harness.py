"""Validation harness: load a split's queries / library mask / pool and score predictions per class."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from casmi.candidates import POOL_DIR, CandidatePool
from casmi.config import WORK_DIR
from casmi.metric import evaluate
from casmi.pipeline import QUERY_COLUMNS

CLEAN_LIBRARY = WORK_DIR / "train_clean.parquet"
NOISE = 0.015  # seed noise on total val MRR; smaller differences are "not measurable"


@dataclass
class ValData:
    split_dir: Path
    val: pd.DataFrame
    queries: pd.DataFrame  # one row per query spectrum, with molecule_id
    in_library: pd.Series  # bool, indexed by train row
    pool: CandidatePool  # Class 3 answers excluded

    @property
    def truth(self) -> dict[str, str]:
        return dict(zip(self.val["molecule_id"], self.val["smiles"]))


def load_val(split_dir: Path, pool_dir: Path = POOL_DIR, library: Path = CLEAN_LIBRARY) -> ValData:
    val = pd.read_parquet(split_dir / "val_ids.parquet")
    mask = pd.read_parquet(split_dir / "library_mask.parquet")
    q = mask[mask["query_molecule_id"].notna()]
    query_of = dict(zip(q["row"].tolist(), q["query_molecule_id"]))
    cols = [c for c in QUERY_COLUMNS if c != "molecule_id"] + ["row"]
    queries = ds.dataset(library).to_table(columns=cols, filter=pc.field("row").isin(list(query_of))).to_pandas()
    queries["molecule_id"] = queries["row"].map(query_of)
    pool = CandidatePool.load(pool_dir).exclude(val.loc[val["class"] == 3, "inchikey14"])
    return ValData(split_dir, val, queries, mask.set_index("row")["in_library"], pool)


def class_mrr(preds: Mapping[str, Sequence[str]], val: pd.DataFrame, label: str) -> dict:
    """``{label, C1, C2, C3, total, hits@1, n}`` for one prediction set."""
    res = evaluate(preds, dict(zip(val["molecule_id"], val["smiles"])),
                   classes=dict(zip(val["molecule_id"], val["class"])))
    return {"run": label, **{f"C{c}": res.get(f"mrr_class{c}", np.nan) for c in (1, 2, 3)},
            "total": res["mrr@25"], "hits@1": res["hits@1"], "n": int(res["n"])}


def compare(a: float, b: float, noise: float = NOISE) -> str:
    """Label a total-MRR difference against the seed-noise threshold."""
    d = b - a
    return f"{d:+.3f} ({'not measurable' if abs(d) <= noise else 'measurable'})"

