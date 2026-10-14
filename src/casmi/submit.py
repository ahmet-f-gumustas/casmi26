"""Build and validate ``submission.csv`` in the competition format.

    molecule_id,smiles
    m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C

One row per test ``molecule_id``, up to 25 SMILES joined by ``;``, best first.
Validation mirrors the rejections of the official scorer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from casmi.config import TOP_K, WORK_DIR
from casmi.metric import dedup_by_key, split_guesses

COLUMNS = ["molecule_id", "smiles"]
# The scorer rejects a row without guesses, so a molecule with no usable
# candidate still gets one (it simply never matches).
FILLER_SMILES = "C"


def format_guesses(smiles: Sequence[str], k: int = TOP_K, dedup: bool = True) -> str:
    """``;``-joined top-``k`` guesses; with ``dedup``, one per scorer key and unparseable ones dropped."""
    guesses = dedup_by_key(split_guesses(smiles), k) if dedup else split_guesses(smiles)[:k]
    return ";".join(guesses or [FILLER_SMILES])


def validate_submission(df: pd.DataFrame, molecule_ids: Sequence[str] | None = None, k: int = TOP_K) -> None:
    """Raise ``ValueError`` for anything the official scorer would reject.

    With ``molecule_ids``, also require exactly that set of ids.
    """
    if not all(c in df.columns for c in COLUMNS):
        raise ValueError(f"Missing columns: {COLUMNS}")
    if len(df) == 0:
        raise ValueError("Submission is empty")
    if df[COLUMNS].isna().any().any():
        raise ValueError("NaN in required columns")
    if df["molecule_id"].duplicated().any():
        raise ValueError("Each molecule_id must appear exactly once")
    for molecule_id, raw in zip(df["molecule_id"], df["smiles"]):
        n = len(split_guesses(str(raw)))
        if n == 0:
            raise ValueError(f"No guesses for molecule_id {molecule_id}")
        if n > k:
            raise ValueError(f"At most {k} semicolon-separated guesses per molecule_id, got {n} for {molecule_id}")
    if molecule_ids is not None:
        missing = set(molecule_ids) - set(df["molecule_id"])
        unknown = set(df["molecule_id"]) - set(molecule_ids)
        if missing or unknown:
            raise ValueError(f"molecule_id mismatch: {len(missing)} missing, {len(unknown)} unknown")


def make_submission(
    predictions: Mapping[str, Sequence[str]],
    molecule_ids: Sequence[str],
    k: int = TOP_K,
    dedup: bool = True,
) -> pd.DataFrame:
    """Submission frame for ``molecule_ids`` (in that order) from ranked SMILES lists."""
    rows = [(mid, format_guesses(predictions.get(mid, []), k, dedup)) for mid in molecule_ids]
    df = pd.DataFrame(rows, columns=COLUMNS)
    validate_submission(df, molecule_ids, k)
    return df


def write_submission(df: pd.DataFrame, path: Path = WORK_DIR / "submission.csv") -> Path:
    """Validate and write ``df`` as CSV; returns the path."""
    validate_submission(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
