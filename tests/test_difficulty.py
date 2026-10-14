import numpy as np
import pytest

from casmi.difficulty import nn_bin, nn_tanimoto, tilt_to_target


def packed(*on_bits: int) -> np.ndarray:
    bits = np.zeros(2048, dtype=np.uint8)
    bits[list(on_bits)] = 1
    return np.packbits(bits)


def test_nn_tanimoto_excludes_self_and_counts_analogs_in_mass_window() -> None:
    ref = np.stack([packed(1, 2, 3, 4), packed(1, 2, 3), packed(1, 2, 3, 4, 5), packed(9)])
    ref_keys = ["Q", "A", "B", "C"]
    ref_mass = np.array([300.0, 310.0, 600.0, 300.0])
    nn, count = nn_tanimoto(np.stack([packed(1, 2, 3, 4)]), ["Q"], np.array([300.0]), ref, ref_keys, ref_mass,
                            min_tanimoto=0.6, max_shift_da=200.0, chunk=1)
    assert nn[0] == pytest.approx(4 / 5)  # B (4/5) beats A (3/4); its own key Q is excluded
    assert count[0] == 1  # A within ±200 Da; B is 300 Da away; C dissimilar


def test_nn_bin_edges() -> None:
    assert nn_bin(np.array([0.0, 0.29, 0.3, 0.5, 0.69, 0.7, 1.0])).tolist() == [0, 0, 1, 2, 2, 3, 3]


def test_tilt_hits_target_and_stays_close() -> None:
    shares = np.array([0.1, 0.2, 0.3, 0.4])
    mrr = np.array([0.2, 0.4, 0.6, 0.8])
    w, _ = tilt_to_target(shares, mrr, 0.5)
    assert w.sum() == pytest.approx(1.0) and w @ mrr == pytest.approx(0.5, abs=1e-6)
    same, lam = tilt_to_target(shares, mrr, float(shares @ mrr))
    assert np.allclose(same, shares, atol=1e-6) and abs(lam) < 1e-6
    with pytest.raises(ValueError, match="outside"):
        tilt_to_target(shares, mrr, 0.1)
