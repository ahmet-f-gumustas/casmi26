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
from casmi.channels.analog import AnalogIndex, library_row_keys
from casmi.config import WORK_DIR
from casmi.metric import evaluate
from casmi.pipeline import (
    QUERY_COLUMNS,
    PipelineConfig,
    analog_channel,
    candidate_windows,
    clean_queries,
    fragment_channel,
    library_channel,
)
from casmi.ranker import combine, rank

CLEAN_LIBRARY = WORK_DIR / "train_clean.parquet"
ANALOG_INDEX = WORK_DIR / "analog_index"
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


def compute_channels(data: ValData, index_dir: Path = ANALOG_INDEX, channels: Sequence[str] = ("library", "analog", "fragment"),
                     workers: int = 24) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Candidate windows and per-channel scores for every validation molecule (pipeline code path)."""
    cfg = PipelineConfig(workers=workers)
    queries = clean_queries(data.queries)
    windows = candidate_windows(queries, data.pool, cfg)
    smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
    out: dict[str, dict] = {}
    if "library" in channels:
        out["library"] = library_channel(queries, windows, CLEAN_LIBRARY, cfg, data.in_library)
    if "analog" in channels:
        allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
        allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
        row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
        out["analog"] = analog_channel(queries, windows, data.pool, AnalogIndex.load(index_dir), row_key,
                                       allowed=allowed, workers=workers)
    if "fragment" in channels:
        out["fragment"] = fragment_channel(queries, windows, smiles_of, workers=workers)
    return windows, out


def predict(windows: Mapping[str, list[str]], channels: Mapping[str, dict], weights: Mapping[str, float],
            smiles_of: Mapping[str, str], ids) -> dict[str, list[str]]:
    """Ranker-v0 predictions for molecules ``ids`` from precomputed channel scores."""
    preds = {}
    for m in ids:
        keys = windows.get(m, [])
        per = {c: channels[c].get(m, {}) for c in channels}
        preds[m] = [smiles_of[k] for k in rank(keys, combine(keys, per, weights), per.get("analog"))]
    return preds


def reciprocal_ranks(windows: Mapping[str, list[str]], channels: Mapping[str, dict], weights: Mapping[str, float],
                     val: pd.DataFrame) -> pd.Series:
    """Per-molecule reciprocal rank of the answer's scorer key (pool keys are unique, so this equals the metric)."""
    rr = {}
    for m, truth in zip(val["molecule_id"], val["inchikey14"]):
        keys = windows.get(m, [])
        per = {c: channels[c].get(m, {}) for c in channels}
        ranked = rank(keys, combine(keys, per, weights), per.get("analog"))
        rr[m] = 1.0 / (ranked.index(truth) + 1) if truth in ranked else 0.0
    return pd.Series(rr, name="rr")


def library_keys(in_library: pd.Series) -> set[str]:
    """Scorer keys with at least one spectrum in the (masked) cleaned library."""
    row_key = library_row_keys(CLEAN_LIBRARY)
    rows = in_library.index[in_library.to_numpy()]
    return set(row_key.reindex(rows).dropna())
