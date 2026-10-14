import pandas as pd
import pytest

from casmi.metric import evaluate, load_official_scorer
from casmi.submit import FILLER_SMILES, make_submission, validate_submission, write_submission

ETHANOL = "CCO"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
PYRIDONE = "O=C1C=CC=CN1"
HYDROXYPYRIDINE = "Oc1ccccn1"


def test_make_submission_format() -> None:
    preds = {
        "m_b": [PYRIDONE, HYDROXYPYRIDINE, "xx", ETHANOL],
        "m_a": [f"C{'C' * i}O" for i in range(40)],
    }
    df = make_submission(preds, ["m_a", "m_b", "m_c"])
    assert list(df.columns) == ["molecule_id", "smiles"]
    assert df["molecule_id"].tolist() == ["m_a", "m_b", "m_c"]
    rows = dict(zip(df["molecule_id"], df["smiles"].str.split(";")))
    assert len(rows["m_a"]) == 25
    assert rows["m_b"] == [PYRIDONE, ETHANOL]  # tautomer duplicate and invalid SMILES removed
    assert rows["m_c"] == [FILLER_SMILES]


def test_csv_round_trip_scores(tmp_path) -> None:
    df = make_submission({"m_1": [ETHANOL, CAFFEINE], "m_2": [CAFFEINE]}, ["m_1", "m_2"])
    path = write_submission(df, tmp_path / "submission.csv")
    back = pd.read_csv(path)
    validate_submission(back, ["m_1", "m_2"])
    res = evaluate(dict(zip(back["molecule_id"], back["smiles"])), {"m_1": CAFFEINE, "m_2": CAFFEINE})
    assert res["mrr@25"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (pd.DataFrame({"molecule_id": ["m"]}), "Missing columns"),
        (pd.DataFrame({"molecule_id": [], "smiles": []}), "empty"),
        (pd.DataFrame({"molecule_id": ["m"], "smiles": [None]}), "NaN"),
        (pd.DataFrame({"molecule_id": ["m", "m"], "smiles": ["C", "C"]}), "exactly once"),
        (pd.DataFrame({"molecule_id": ["m"], "smiles": [";;"]}), "No guesses"),
        (pd.DataFrame({"molecule_id": ["m"], "smiles": [";".join(["C"] * 26)]}), "At most 25"),
    ],
)
def test_validate_rejects_like_scorer(df: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_submission(df)


def test_validate_checks_id_set() -> None:
    df = pd.DataFrame({"molecule_id": ["m_1"], "smiles": ["C"]})
    with pytest.raises(ValueError, match="1 missing"):
        validate_submission(df, ["m_1", "m_2"])


def test_official_scorer_accepts_our_file(tmp_path) -> None:
    official = load_official_scorer()
    if official is None:
        pytest.skip("official metric notebook not downloaded to external/official_metric")
    df = make_submission({"m_1": [ETHANOL, CAFFEINE], "m_2": [HYDROXYPYRIDINE]}, ["m_1", "m_2"])
    solution = pd.DataFrame({"molecule_id": ["m_1", "m_2"], "smiles": [CAFFEINE, PYRIDONE], "Usage": "Public"})
    back = pd.read_csv(write_submission(df, tmp_path / "submission.csv"))
    assert official.score(solution, back, "molecule_id") == pytest.approx(0.75)
