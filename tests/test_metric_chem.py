import numpy as np
import pytest

from casmi.chem import (
    ADDUCTS,
    ELECTRON,
    FORMATE,
    H,
    H2O,
    NA,
    PROTON,
    TEST_ADDUCTS,
    exact_mass,
    mass_window_indices,
    neutral_mass,
    parse_adduct,
    precursor_mz,
)
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


# Long-tail train adducts that are not in ADDUCTS but must still resolve via the parser.
PARSED_ONLY = ["[M+HNa]2+", "[M-H5O3]+", "[4M+Ca]2+", "[M+C2F3O2]-", "[2M+Na3-H4]-"]


@pytest.mark.parametrize("adduct", sorted(ADDUCTS) + PARSED_ONLY)
def test_adduct_round_trip(adduct: str) -> None:
    m = 194.080376
    assert neutral_mass(precursor_mz(m, adduct), adduct) == pytest.approx(m, abs=1e-9)


def test_caffeine_protonated_mz() -> None:
    assert precursor_mz(194.080376, "[M+H]+") == pytest.approx(195.087652, abs=1e-5)


@pytest.mark.parametrize("adduct", ["[M+Xx]+", "[Cat]2+", "M+H", ""])
def test_unknown_adduct_raises(adduct: str) -> None:
    with pytest.raises(KeyError, match="unknown adduct"):
        neutral_mass(100.0, adduct)


def test_documented_test_adducts_are_known() -> None:
    documented = {"[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                  "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"}
    assert set(TEST_ADDUCTS) == documented
    assert documented <= set(ADDUCTS)


@pytest.mark.parametrize(
    ("adduct", "expected"),
    [
        ("[M+H]+", (1, PROTON, 1)),
        ("[M-H]-", (1, -PROTON, -1)),
        ("[M+Na]+", (1, NA - ELECTRON, 1)),
        ("[M-H2O+H]+", (1, PROTON - H2O, 1)),
        ("[M+CH2O2-H]-", (1, FORMATE + ELECTRON, -1)),
        ("[M+2H]2+", (1, 2 * PROTON, 2)),
        ("[2M+Na-2H]-", (2, NA - 2 * H + ELECTRON, -1)),
    ],
)
def test_parsed_adduct_shift(adduct: str, expected: tuple[int, float, int]) -> None:
    n_m, shift, charge = parse_adduct(adduct)
    assert (n_m, charge) == (expected[0], expected[2])
    # H atom - electron differs from the proton mass by the 13.6 eV binding energy (~1.5e-8 Da)
    assert shift == pytest.approx(expected[1], abs=1e-7)


def test_aliases_match_competition_names() -> None:
    for alias, name in [("[M+H-H2O]+", "[M-H2O+H]+"), ("[M+FA-H]-", "[M+CH2O2-H]-"), ("[M+HCOO]-", "[M+CH2O2-H]-"),
                        ("[M-H-H2O]-", "[M-H2O-H]-"), ("[M+CH3COO]-", "[M+C2H4O2-H]-")]:
        assert ADDUCTS[alias][1] == pytest.approx(ADDUCTS[name][1], abs=1e-9)


def test_exact_mass_of_permanent_cation() -> None:
    # tetramethylammonium C4H12N+ observed at m/z 74.096426 under [M]+
    assert precursor_mz(exact_mass("C[N+](C)(C)C"), "[M]+") == pytest.approx(74.096426, abs=1e-6)
    assert exact_mass("CN1C=NC2=C1C(=O)N(C(=O)N2C)C") == pytest.approx(194.080376, abs=1e-6)
    assert exact_mass("xx") is None


def test_mass_window_indices() -> None:
    masses = np.array([100.0, 150.0, 194.08, 194.081, 194.09, 300.0])
    idx = mass_window_indices(masses, 194.0804, ppm=10)
    assert idx.tolist() == [2, 3]
    assert mass_window_indices(masses, 250.0, ppm=10).size == 0
