import numpy as np
import pytest

from casmi.chem import ADDUCTS, mass_window_indices, neutral_mass, precursor_mz
from casmi.metric import dedup_by_key, evaluate, scorer_key

ETHANOL = "CCO"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
PYRIDONE = "O=C1C=CC=CN1"
HYDROXYPYRIDINE = "Oc1ccccn1"


def test_tautomers_share_scorer_key() -> None:
    key = scorer_key(PYRIDONE)
    assert key is not None and len(key) == 14
    assert key == scorer_key(HYDROXYPYRIDINE)


@pytest.mark.parametrize("smiles", ["xx", "", "C1CC"])
def test_invalid_smiles_has_no_key(smiles: str) -> None:
    assert scorer_key(smiles) is None


def test_raw_rank_is_preserved() -> None:
    res = evaluate({"m": [ETHANOL, ETHANOL, CAFFEINE]}, {"m": CAFFEINE})
    assert res["mrr@25"] == pytest.approx(1 / 3)
    assert res["hits@1"] == 0.0 and res["hits@5"] == 1.0
    assert res["median_rank"] == 3.0


def test_answer_beyond_top_k_scores_zero() -> None:
    preds = [ETHANOL] * 25 + [CAFFEINE]
    res = evaluate({"m": preds}, {"m": CAFFEINE})
    assert res["mrr@25"] == 0.0
    assert res["missing_frac"] == 1.0


def test_per_class_mrr() -> None:
    preds = {"a": [CAFFEINE], "b": [ETHANOL, CAFFEINE], "c": []}
    truth = {"a": CAFFEINE, "b": CAFFEINE, "c": CAFFEINE}
    res = evaluate(preds, truth, classes={"a": 1, "b": 2, "c": 3})
    assert res["mrr_class1"] == 1.0
    assert res["mrr_class2"] == 0.5
    assert res["mrr_class3"] == 0.0
    assert res["mrr@25"] == pytest.approx(0.5)


def test_dedup_by_key() -> None:
    out = dedup_by_key([PYRIDONE, HYDROXYPYRIDINE, ETHANOL, "xx"])
    assert out == [PYRIDONE, ETHANOL]


@pytest.mark.parametrize("adduct", sorted(ADDUCTS))
def test_adduct_round_trip(adduct: str) -> None:
    m = 194.080376
    assert neutral_mass(precursor_mz(m, adduct), adduct) == pytest.approx(m, abs=1e-9)


def test_caffeine_protonated_mz() -> None:
    assert precursor_mz(194.080376, "[M+H]+") == pytest.approx(195.087652, abs=1e-5)


def test_unknown_adduct_raises() -> None:
    with pytest.raises(KeyError, match="unknown adduct"):
        neutral_mass(100.0, "[M+Xx]+")


def test_mass_window_indices() -> None:
    masses = np.array([100.0, 150.0, 194.08, 194.081, 194.09, 300.0])
    idx = mass_window_indices(masses, 194.0804, ppm=10)
    assert idx.tolist() == [2, 3]
    assert mass_window_indices(masses, 250.0, ppm=10).size == 0
