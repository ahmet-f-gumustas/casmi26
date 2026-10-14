import numpy as np

from casmi.data import mean_ce


def test_mean_ce() -> None:
    assert mean_ce([20.0, 40.0, 60.0]) == 40.0
    assert np.isnan(mean_ce(None)) and mean_ce(None, 25.0) == 25.0
    assert mean_ce(np.array([np.nan, 10.0])) == 10.0
