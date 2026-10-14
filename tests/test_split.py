import numpy as np
import pandas as pd
import pytest

from casmi.split import Split, SplitConfig, candidate_pool_keys, check_leakage, final_split, make_split

N_MOL = 400
N_NP = 40


def synthetic_rows(seed: int = 0) -> pd.DataFrame:
    """Train-like rows: every molecule has timsTOF spectra, every third also other-library ones.
    NP molecules are heavier than enveda-180 ones, like the real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_MOL):
        key = f"KEY{i:011d}"
        is_np = i < N_NP
        lib = "enveda-np-examples" if is_np else "enveda-180"
        mass = float(rng.uniform(160, 900) if is_np else rng.uniform(200, 450))
        for _ in range(rng.integers(1, 8)):
            rows.append({"ingest_lib": lib, "key": key, "normalized_smiles": f"C{i}",
                         "adduct": rng.choice(["[M+H]+", "[M-H]-", "[M+NH4]+" if is_np else "[M+Na]+", "[2M+Na]+"]),
                         "ppm": rng.normal(0, 3), "base_peak_intensity": rng.choice([500.0, 5e4, 5e4]),
                         "n_ce": rng.choice([1, 3]), "exact_mass": mass})
        if is_np or i % 3 == 0:
            rows.append({"ingest_lib": "gnps", "key": key, "normalized_smiles": f"C{i}", "adduct": "[M+H]+",
                         "ppm": 1.0, "base_peak_intensity": np.nan, "n_ce": 1, "exact_mass": mass})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # interleave molecules


CFG = SplitConfig(max_drug_share=0.5, seed=7)


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return synthetic_rows()


@pytest.fixture(scope="module")
def split(rows: pd.DataFrame) -> Split:
    return make_split(rows, CFG)


def test_sizes_and_class_counts(split: Split) -> None:
    n_np = int((split.val["source"] == "enveda-np-examples").sum())
    n = len(split.val)
    assert n == round(n_np / (1 - CFG.max_drug_share))
    assert (split.val["source"] == "enveda-180").mean() <= CFG.max_drug_share + 1e-9
    counts = split.val["class"].value_counts().to_dict()
    assert counts[1] == round(0.16 * n) and counts[3] == round(0.39 * n) and sum(counts.values()) == n


def test_no_leakage(rows: pd.DataFrame, split: Split) -> None:
    check_leakage(rows, split, candidate_pool_keys(rows["key"], split))


def test_all_spectra_of_a_val_molecule_leave_timstof_library(rows: pd.DataFrame, split: Split) -> None:
    val_keys = set(split.val["inchikey14"])
    sub = rows["key"].isin(val_keys) & rows["ingest_lib"].str.startswith("enveda")
    assert not split.library_mask.loc[sub.to_numpy(), "in_library"].any()


def test_every_eligible_np_example_is_in_val(rows: pd.DataFrame, split: Split) -> None:
    eligible = rows[(rows["ingest_lib"] == "enveda-np-examples") & (rows["base_peak_intensity"] >= 1000)
                    & (rows["adduct"] != "[2M+Na]+") & (rows["ppm"].abs() <= 20)]
    np_val = set(split.val.loc[split.val["source"] == "enveda-np-examples", "inchikey14"])
    assert np_val == set(eligible["key"]) and len(np_val) > 20


def test_drug_molecules_follow_np_mass_bins(split: Split) -> None:
    drug = split.val[split.val["source"] == "enveda-180"]
    assert drug["exact_mass"].max() > 350  # pulled towards the heavier NP distribution


def test_queries_respect_test_filters(rows: pd.DataFrame, split: Split) -> None:
    q = rows[split.library_mask["query_molecule_id"].notna().to_numpy()]
    assert q["adduct"].isin(["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+"]).all()  # [2M+Na]+ is not a test adduct
    assert (q["base_peak_intensity"] >= 1000).all()
    assert q["ingest_lib"].str.startswith("enveda").all()
    assert split.val["n_query"].between(1, 16).all()


def test_deterministic(rows: pd.DataFrame, split: Split) -> None:
    again = make_split(rows, CFG)
    pd.testing.assert_frame_equal(split.val, again.val)
    pd.testing.assert_frame_equal(split.library_mask, again.library_mask)
    other = make_split(rows, SplitConfig(max_drug_share=0.5, seed=8))
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
    no_refs = rows[~((rows["ingest_lib"] == "gnps") & ~rows["key"].isin([f"KEY{i:011d}" for i in range(N_NP)]))]
    with pytest.raises(ValueError, match="C1-eligible"):
        make_split(no_refs, SplitConfig(max_drug_share=0.9, seed=7))


def test_final_split_keeps_everything(rows: pd.DataFrame) -> None:
    final = final_split(len(rows))
    assert final.library_mask["in_library"].all() and final.library_mask["query_molecule_id"].isna().all()
    assert final.val.empty and list(final.val.columns)[:3] == ["molecule_id", "inchikey14", "class"]


def test_split_v2_follows_c2_nn_profile(rows: pd.DataFrame) -> None:
    rng = np.random.default_rng(3)
    keys = rows["key"].unique()
    nn = pd.Series(rng.uniform(0.3, 0.95, len(keys)), index=keys)
    cfg = SplitConfig(max_drug_share=0.5, seed=7, c2_profile=(0.0, 0.6, 0.3, 0.1))
    v2 = make_split(rows, cfg, nn)
    check_leakage(rows, v2, candidate_pool_keys(rows["key"], v2))
    c2 = v2.val[v2.val["class"] == 2]
    bins = pd.cut(c2["nn_tanimoto"], [0.0, 0.3, 0.5, 0.7, 1.0001], right=False).value_counts(sort=False).to_numpy()
    assert bins[0] == 0 and bins[1] >= bins[2] >= bins[3]
    assert (c2["nn_tanimoto"] >= 0.7).mean() <= 0.1 + 2 / len(c2)
    assert v2.val["class"].value_counts().to_dict() == make_split(rows, CFG).val["class"].value_counts().to_dict()


def test_split_v2_needs_nn(rows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="nn_tanimoto"):
        make_split(rows, SplitConfig(seed=7, c2_profile=(0.0, 0.5, 0.35, 0.15)))
