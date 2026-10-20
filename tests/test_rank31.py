import numpy as np
import pandas as pd

from casmi import rank31


def test_rank_features_shape_and_definitions() -> None:
    lv = np.array([0.0, 0.9, 0.2], np.float32)
    tan = np.array([[0.2, 0.1], [0.9, 0.3], [0.5, 0.8]], np.float32)
    sims = np.array([0.8, 0.5], np.float32)
    raw = np.array([1.0, 3.0, 2.0], np.float32)
    X = rank31.rank_features(lv, tan, sims, raw, np.array([4, 9, 1]), np.array([0.1, 0.5, 0.3]))
    f = dict(zip(rank31.FEATURES, X.T))
    assert X.shape == (3, 31) and len(rank31.FEATURES) == 31
    assert np.allclose(f["ap"], (tan * sims ** 3).max(1)) and np.allclose(f["top_tan"], tan[:, 0])
    assert f["lv_rank"][1] == 0.0 and f["m_is_max"].tolist() == [0, 1, 0]
    assert np.allclose(f["x_agree"], 1.0)  # the library's best candidate is also the model's best
    assert np.allclose(f["log_nc"], np.log(3))


def test_rank_features_without_analogs_or_model() -> None:
    X = rank31.rank_features(np.zeros(2), None, np.zeros(0), None, None, None)
    assert X.shape == (2, 31) and np.allclose(X[:, 14], np.log(2))
    nonzero = {n for n, col in zip(rank31.FEATURES, X.T) if col.any()}
    assert nonzero == {"lv_rank", "ap_rank", "log_nc"}  # ranks of ties are positional, as in the original


def test_metfrag_lite_explains_a_cleavage_fragment() -> None:
    frags = rank31.fragment_masses("CCOC(=O)c1ccccc1")  # ethyl benzoate
    assert len(frags) > 3
    ion = frags.min() + rank31._PROTON
    assert rank31.explain_score(frags, np.array([ion]), np.array([1.0])) == 1.0
    assert rank31.explain_score(frags, np.array([ion + 0.5]), np.array([1.0])) == 0.0


def test_clean_peaks_keeps_the_most_intense() -> None:
    mz, it = rank31.clean_peaks(np.arange(5, dtype=float), np.array([1.0, 0.001, 5.0, 3.0, 2.0]), topk=3)
    assert mz.tolist() == [2.0, 3.0, 4.0] and np.isclose(it.sum(), 1.0)


def test_analog_list_takes_the_best_per_structure() -> None:
    hits = [(np.array([1, 2]), np.array([0.9, 0.4])), (np.array([3, 2]), np.array([0.5, 0.7]))]
    keys, sims = rank31.analog_list(hits, {1: "A", 2: "B", 3: "A"}, n=5)
    assert keys == ["A", "B"] and np.allclose(sims, [0.9, 0.7])


def test_fit_and_proba_follow_the_label() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    Y = (X[:, 0] > 1).astype(float)
    M = (np.arange(400) % 2).astype(float)
    models = rank31.fit(X, Y, M, priors=(0.3,), seeds=(0,), params=dict(max_iter=20))
    p = rank31.proba(models, X)
    assert p[Y == 1].mean() > p[Y == 0].mean() and rank31.columns(["lv", "f_z"]) == [0, 24]


def test_frame_builds_one_row_per_candidate() -> None:
    q = pd.DataFrame({"molecule_id": ["m"], "ionization_mode": ["positive"], "ms2_mzs": [np.array([77.04, 105.03])],
                      "ms2_normalized_intensities": [np.array([0.5, 1.0])]})
    smiles = {"A": "CCOC(=O)c1ccccc1", "B": "CC(=O)Oc1ccccc1", "L": "COC(=O)c1ccccc1"}
    df = rank31.frame(q, {"m": ["A", "B"]}, smiles, [(np.array([7]), np.array([0.8]))], {7: "L"},
                      {"m": {"A": 0.6}}, None, None)
    assert len(df) == 2 and df.loc[0, "lv"] > 0 and df.loc[1, "lv"] == 0 and (df["top_sim"] > 0.79).all()
    assert df.loc[0, "best_tan"] > df.loc[1, "best_tan"]  # ethyl vs methyl benzoate are closer than the acetate

