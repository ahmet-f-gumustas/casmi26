import numpy as np
import pandas as pd

from casmi.combiners import Logistic, combine_rows, insert_top1, mixed, product, topk_rerank, zscore
from casmi.ranker_v2 import rank_all


def rows() -> pd.DataFrame:
    df = pd.DataFrame({
        "molecule_id": ["m1"] * 4 + ["m2"] * 3,
        "key": ["A", "B", "C", "D", "E", "F", "G"],
        "analog": [0.9, 0.8, 0.7, 0.1, 0.5, 0.5, 0.2],
        "fingerprint": [10.0, 50.0, 30.0, 100.0, 1.0, 2.0, 3.0],
        "fragment": [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0],
        "library_max": [0.2, 0.85, 0.1, 0.0, 0.5, 0.6, 0.7],
        "t_analog1": [0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1],
        "sim1": [0.6] * 4 + [0.3] * 3,
        "n_candidates": [4] * 4 + [3] * 3,
        "label": [False, False, True, False, False, True, False],
    })
    g = df.groupby("molecule_id")
    for col in ("analog", "fingerprint", "fragment"):
        df[f"{col}_rank"] = g[col].rank(method="min", ascending=False)
        df[f"{col}_pct"] = g[col].rank(pct=True)
    return df


def test_zscore_is_within_window() -> None:
    z = zscore(rows(), "analog")
    assert abs(z[:4].mean()) < 1e-9 and abs(z[:4].std() - 1) < 1e-9
    assert (zscore(rows().assign(analog=1.0), "analog") == 0).all()


def test_topk_rerank_keeps_the_tail_in_analog_order() -> None:
    df = rows()
    top = rank_all(df, topk_rerank(df, k=2, alpha=0.5))
    assert top["m1"][:2] == ["B", "A"] and top["m1"][2:] == ["C", "D"]  # D has the best fp but is outside the top 2
    assert rank_all(df, topk_rerank(df, k=4, alpha=0.0))["m1"] == ["D", "B", "C", "A"]
    assert rank_all(df, topk_rerank(df, k=4, alpha=1.0))["m1"] == ["A", "B", "C", "D"]


def test_product_depends_on_the_temperature_ratio() -> None:
    df = rows()
    a = rank_all(df, product(df, 0.1, 100.0))
    assert a == rank_all(df, product(df, 0.2, 200.0))
    assert rank_all(df, product(df, 1e-6, 1.0))["m1"] == ["A", "B", "C", "D"]


def test_insert_top1_moves_only_the_strong_library_match() -> None:
    df = rows()
    s = insert_top1(df, df["analog"].to_numpy(), tau=0.8)
    assert rank_all(df, s)["m1"] == ["B", "A", "C", "D"]
    assert rank_all(df, s)["m2"] == ["E", "F", "G"]  # best library match 0.7 < τ: unchanged


def test_logistic_round_trips_through_its_dict() -> None:
    df = pd.concat([rows().assign(molecule_id=lambda d, i=i: d["molecule_id"] + str(i)) for i in range(5)],
                   ignore_index=True)
    cols = ["analog_z", "fp_z", "analog_logrank", "fp_logrank", "t_analog1", "fragment_z"]
    m = Logistic(cols).fit(df)
    s = combine_rows(df, {"kind": "logistic", "model": m.to_dict()})
    assert np.allclose(s, m.score(df)) and len(m.coef) == len(cols)
    assert np.isinf(combine_rows(df, {"kind": "analog", "insert_tau": 0.8})).sum() == 5


def test_mixed_is_scale_free_in_the_fingerprint() -> None:
    df = rows()
    a = rank_all(df, mixed(df, 0.1, 1.0))
    scaled = df.assign(fingerprint=df["fingerprint"] * 37 + 5)
    assert a == rank_all(scaled, mixed(scaled, 0.1, 1.0))
    assert rank_all(df, combine_rows(df, {"kind": "mixed", "t1": 0.1, "w": 1.0})) == a
    assert rank_all(df, mixed(df, 1e-6, 1.0))["m1"] == ["A", "B", "C", "D"]


def test_topk_without_fingerprint_ignores_it() -> None:
    df = rows()
    spec = {"kind": "topk", "k": 100, "alpha": 0.5, "fp": None, "extra": {"fragment": 0.3}}
    a = rank_all(df, combine_rows(df, spec))
    assert a == rank_all(df.assign(fingerprint=-df["fingerprint"]), combine_rows(df.assign(fingerprint=0.0), spec))
    assert a["m1"] == rank_all(df, 0.5 * zscore(df, "analog") + 0.3 * zscore(df, "fragment"))["m1"]
