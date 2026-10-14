import numpy as np
import pytest

from casmi.channels.library import prepare, rank, score_molecule

PREC = 300.0


def spec(*peaks: tuple[float, float]) -> np.ndarray:
    mz, it = zip(*peaks)
    return prepare(np.array(mz), np.array(it), PREC)


A = spec((100.0, 1.0), (150.0, 0.5), (200.0, 0.2))
B = spec((100.0, 1.0), (175.0, 0.8))
C = spec((50.0, 1.0), (60.0, 1.0))


def test_prepare_drops_precursor_region_and_noise() -> None:
    peaks = spec((100.0, 1.0), (150.0, 0.005), (299.0, 0.9), (310.0, 1.0))
    assert peaks[:, 0].tolist() == [100.0]


def test_identical_spectrum_ranks_first() -> None:
    refs = {"KA": [("positive", A)], "KB": [("positive", B)], "KC": [("positive", C)]}
    scores = score_molecule([("positive", A)], ["KB", "KA", "KC", "KD"], refs)
    assert rank(scores, "max") == ["KA", "KB"]  # KC shares no peak, KD has no references
    assert scores.set_index("key").loc["KA", "max"] == pytest.approx(1.0, abs=1e-6)


def test_ionization_mode_must_match() -> None:
    refs = {"KA": [("negative", A)]}
    assert rank(score_molecule([("positive", A)], ["KA"], refs)) == []


def test_mean_penalises_unmatched_query_spectra() -> None:
    refs = {"KA": [("positive", A)], "KB": [("positive", B), ("negative", C)]}
    queries = [("positive", A), ("negative", C)]
    scores = score_molecule(queries, ["KA", "KB"], refs).set_index("key")
    assert scores.loc["KA", "max"] == pytest.approx(1.0, abs=1e-6)
    assert scores.loc["KA", "mean"] == pytest.approx(0.5, abs=1e-6)  # no negative reference for KA
    # KB matches the negative spectrum exactly and the positive one partly: better on mean, tied on max
    assert scores.loc["KB", "max"] == pytest.approx(1.0, abs=1e-6)
    assert scores.loc["KB", "mean"] > scores.loc["KA", "mean"]
    assert rank(scores.reset_index(), "mean")[0] == "KB"


def test_rank_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValueError):
        rank(score_molecule([("positive", A)], ["KA"], {"KA": [("positive", A)]}), "median")
