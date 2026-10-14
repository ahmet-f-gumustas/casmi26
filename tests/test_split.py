import numpy as np
import pandas as pd
import pytest

from casmi.split import Split, SplitConfig, candidate_pool_keys, check_leakage, make_split

N_MOL = 300


def synthetic_rows(seed: int = 0) -> pd.DataFrame:
    """Train-like rows: every molecule has timsTOF spectra, every third also other-library ones."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_MOL):
        key = f"KEY{i:011d}"
        lib = "enveda-np-examples" if i < 20 else "enveda-180"
        for _ in range(rng.integers(1, 8)):
            rows.append({"ingest_lib": lib, "key": key, "normalized_smiles": f"C{i}",
                         "adduct": rng.choice(["[M+H]+", "[M-H]-", "[M+Na]+", "[2M+Na]+"]),
                         "ppm": rng.normal(0, 3), "base_peak_intensity": rng.choice([500.0, 5e4]),
                         "n_ce": rng.choice([1, 3]), "exact_mass": 200.0 + i})
        if i % 3 == 0:
            rows.append({"ingest_lib": "gnps", "key": key, "normalized_smiles": f"C{i}", "adduct": "[M+H]+",
                         "ppm": 1.0, "base_peak_intensity": np.nan, "n_ce": 1, "exact_mass": 200.0 + i})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # interleave molecules


TEST = pd.DataFrame({"molecule_id": ["a", "a", "a", "b", "c", "c"],
                     "adduct": ["[M+H]+", "[M+H]+", "[M-H]-", "[M+H]+", "[M+Na]+", "[M+H]+"],
                     "n_ce": [1, 1, 3, 1, 1, 3]})
CFG = SplitConfig(n_val=100, seed=7)


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return synthetic_rows()


@pytest.fixture(scope="module")
def split(rows: pd.DataFrame) -> Split:
    return make_split(rows, TEST, CFG)


def test_class_counts(split: Split) -> None:
    counts = split.val["class"].value_counts().to_dict()
    assert counts == {1: 16, 2: 45, 3: 39}


def test_no_leakage(rows: pd.DataFrame, split: Split) -> None:
    check_leakage(rows, split, candidate_pool_keys(rows["key"], split))


def test_all_spectra_of_a_val_molecule_leave_timstof_library(rows: pd.DataFrame, split: Split) -> None:
    val_keys = set(split.val["inchikey14"])
    sub = rows["key"].isin(val_keys) & rows["ingest_lib"].str.startswith("enveda")
    assert not split.library_mask.loc[sub.to_numpy(), "in_library"].any()


def test_np_examples_forced_into_val(rows: pd.DataFrame, split: Split) -> None:
    eligible = rows[(rows["ingest_lib"] == "enveda-np-examples") & (rows["base_peak_intensity"] >= 1000)
                    & (rows["adduct"] != "[2M+Na]+") & (rows["ppm"].abs() <= 20)]
    np_val = set(split.val.loc[split.val["source"] == "enveda-np-examples", "inchikey14"])
    assert np_val == set(eligible["key"]) and len(np_val) > 10


def test_queries_respect_test_filters(rows: pd.DataFrame, split: Split) -> None:
    q = rows[split.library_mask["query_molecule_id"].notna().to_numpy()]
    assert q["adduct"].isin(["[M+H]+", "[M-H]-", "[M+Na]+"]).all()  # [2M+Na]+ is not a test adduct
    assert (q["base_peak_intensity"] >= 1000).all()
    assert q["ingest_lib"].str.startswith("enveda").all()
    assert split.val["n_query"].between(1, 3).all()  # visible test has 1-3 spectra per molecule here


def test_deterministic(rows: pd.DataFrame, split: Split) -> None:
    again = make_split(rows, TEST, CFG)
    pd.testing.assert_frame_equal(split.val, again.val)
    pd.testing.assert_frame_equal(split.library_mask, again.library_mask)
    other = make_split(rows, TEST, SplitConfig(n_val=100, seed=8))
    assert set(other.val["inchikey14"]) != set(split.val["inchikey14"])


def test_leakage_check_catches_violations(rows: pd.DataFrame, split: Split) -> None:
    mask = split.library_mask.copy()
    non_query = mask["query_molecule_id"].isna().to_numpy()
    c2_keys = set(split.val.loc[split.val["class"] == 2, "inchikey14"])
    leak = rows["key"].isin(c2_keys).to_numpy() & non_query
    assert leak.any()
    mask.loc[leak, "in_library"] = True
    with pytest.raises(AssertionError, match="C2"):
        check_leakage(rows, Split(split.val, mask), candidate_pool_keys(rows["key"], split))
    with pytest.raises(AssertionError, match="C3 structures"):
        check_leakage(rows, split, set(rows["key"]))


def test_too_few_c1_candidates_raises(rows: pd.DataFrame) -> None:
    no_refs = rows[rows["ingest_lib"] != "gnps"]
    with pytest.raises(ValueError, match="C1-eligible"):
        make_split(no_refs, TEST, CFG)
