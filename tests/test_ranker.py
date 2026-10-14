import numpy as np
import pytest

from casmi.ranker import combine, needed_channels, rank, rank_molecule, rank_normalise


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


KEYS = ["A", "B", "C", "D"]
PER = {"analog": {"A": 0.9, "B": 0.5, "C": 0.2}, "library_max": {"C": 0.85, "D": 0.4}}


def test_gate_promotes_only_strong_library_matches() -> None:
    assert rank_molecule(KEYS, PER, {"mode": "gate", "tau": 0.7, "w_lib": 2.0})[0] == "C"
    # τ above every library match: identical to analog only
    assert rank_molecule(KEYS, PER, {"mode": "gate", "tau": 0.9, "w_lib": 2.0}) == ["A", "B", "C"]


def test_two_stage_puts_library_top1_first_and_fills_from_analog() -> None:
    assert rank_molecule(KEYS, PER, {"mode": "two_stage", "tau": 0.8}) == ["C", "A", "B"]
    assert rank_molecule(KEYS, PER, {"mode": "two_stage", "tau": 0.9}) == ["A", "B", "C"]
    assert rank_molecule(KEYS, {"analog": PER["analog"]}, {"mode": "two_stage", "tau": 0.5}) == ["A", "B", "C"]


def test_weighted_mode_matches_v0_and_needed_channels() -> None:
    v0 = rank_molecule(KEYS, PER, {"mode": "weighted", "weights": {"analog": 1.0}})
    assert v0 == ["A", "B", "C"]
    assert needed_channels({"mode": "weighted", "weights": {"analog": 1.0, "fragment": 0.0}}) == {"analog"}
    assert needed_channels({"mode": "gate", "tau": 0.7, "w_lib": 1.0}) == {"analog", "library_max"}
    with pytest.raises(ValueError):
        rank_molecule(KEYS, PER, {"mode": "nope", "tau": 0.5})
