import numpy as np
import pandas as pd

from casmi.ranker_v2 import fit_bags, predict, rank_all


def test_rank_all_orders_by_score_then_analog_then_key() -> None:
    df = pd.DataFrame({"molecule_id": ["m"] * 4 + ["n"], "key": ["D", "C", "B", "A", "E"],
                       "analog": [0.1, 0.5, 0.5, 0.9, 0.0]})
    out = rank_all(df, np.array([0.2, 0.7, 0.7, 0.1, 0.3]), k=3)
    assert out == {"m": ["B", "C", "D"], "n": ["E"]}


def test_fit_bags_learns_a_separable_label() -> None:
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"x": rng.normal(size=n), "noise": rng.normal(size=n)})
    df["label"] = df["x"] > 1.0
    df["answer"] = [f"g{i // 6}" for i in range(n)]
    models = fit_bags(df, ["x", "noise"], bags=2, params={"max_iter": 50})
    p = predict(models, df, ["x", "noise"])
    assert len(models) == 2 and p[df["label"]].mean() > 0.8 > 0.2 > p[~df["label"]].mean()
