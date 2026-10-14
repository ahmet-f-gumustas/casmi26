import numpy as np
import pytest

from casmi.ranker import combine, rank, rank_normalise


def test_rank_normalise_percentiles_and_zero() -> None:
    out = rank_normalise(np.array([0.0, 0.2, 0.9, 0.2]))
    assert out.tolist() == pytest.approx([0.0, 0.75, 1.0, 0.75])
    assert rank_normalise(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_combine_is_weighted_sum_of_channels() -> None:
    keys = ["A", "B", "C"]
    channels = {"analog": {"A": 0.9, "B": 0.1}, "library": {"C": 0.5}}
    total = combine(keys, channels, {"analog": 1.0, "library": 2.0})
    # analog percentiles: A 1.0, B 2/3, C 0; library: C 1.0
    assert total.tolist() == pytest.approx([1.0, 2 / 3, 2.0])
    assert combine(keys, channels, {"analog": 1.0, "library": 0.0}).tolist() == pytest.approx([1.0, 2 / 3, 0.0])


def test_rank_drops_zero_and_breaks_ties() -> None:
    keys = ["B", "A", "C", "D"]
    combined = np.array([0.5, 0.5, 0.9, 0.0])
    assert rank(keys, combined) == ["C", "A", "B"]  # tie → key order
    assert rank(keys, combined, tiebreak={"B": 0.3, "A": 0.1}) == ["C", "B", "A"]
    assert rank(keys, combined, k=1) == ["C"]
