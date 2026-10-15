import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from casmi.channels.analog import AnalogConfig, AnalogIndex, aggregate, rank_candidates, score_molecule, tanimoto


def packed(*on_bits: int, n_bits: int = 2048) -> np.ndarray:
    bits = np.zeros(n_bits, dtype=np.uint8)
    bits[list(on_bits)] = 1
    return np.packbits(bits)


def test_tanimoto_matches_set_definition() -> None:
    a = np.stack([packed(1, 2, 3), packed(10)])
    b = np.stack([packed(2, 3, 4, 5), packed(10), packed()])
    t = tanimoto(a, b)
    assert t[0, 0] == pytest.approx(2 / 5)
    assert t[1, 1] == 1.0 and t[0, 1] == 0.0 and t[1, 2] == 0.0


def test_score_is_max_of_sim_power_times_tanimoto() -> None:
    fps = np.stack([packed(1, 2), packed(1, 2, 3, 4), packed(9)])  # analog structures (rows 0, 1, 2)
    cands = np.stack([packed(1, 2), packed(9)])
    row_fp = {100: 0, 101: 1, 102: 2}
    analogs = [(np.array([100, 101]), np.array([0.5, 0.9])), (np.array([102]), np.array([0.8]))]
    out = score_molecule(analogs, cands, row_fp, fps, powers=(2.0,))[2.0]
    # spectrum 0, candidate 0: max(0.5^2 * 1, 0.9^2 * 0.5) = 0.405
    assert out[0, 0] == pytest.approx(max(0.25, 0.81 * 0.5))
    assert out[0, 1] == 0.0 and out[1, 1] == pytest.approx(0.64)
    assert aggregate(out, "max").tolist() == pytest.approx([0.405, 0.64])
    assert aggregate(out, "mean").tolist() == pytest.approx([0.2025, 0.32])
    assert rank_candidates(["KA", "KB"], aggregate(out, "max")) == ["KB", "KA"]


def test_exclude_self_drops_a_candidates_own_spectra() -> None:
    fps = np.stack([packed(1, 2), packed(1, 2, 3, 4), packed(9)])
    row_fp = {100: 0, 101: 1, 102: 2}
    analogs = [(np.array([100, 101]), np.array([0.95, 0.9]))]
    rows = np.array([0, 2])  # candidate 0 is the structure of library row 100
    out = score_molecule(analogs, fps[rows], row_fp, fps, powers=(2.0,), candidate_rows=rows, self_min_sim=2.0)[2.0]
    assert out[0, 0] == pytest.approx(0.81 * 0.5)  # only the other analog counts
    assert out[0, 1] == 0.0
    with_self = score_molecule(analogs, fps[rows], row_fp, fps, powers=(2.0,))[2.0]
    assert with_self[0, 0] == pytest.approx(0.95**2)
    for thr, expected in ((0.9, 0.95**2), (0.96, 0.81 * 0.5), (0.0, 0.95**2)):
        got = score_molecule(analogs, fps[rows], row_fp, fps, powers=(2.0,), candidate_rows=rows, self_min_sim=thr)
        assert got[2.0][0, 0] == pytest.approx(expected)


def test_top_k_prefix_and_unknown_rows() -> None:
    fps = np.stack([packed(1)])
    analogs = [(np.array([5, 7]), np.array([0.9, 0.8]))]  # row 7 has no fingerprint
    out = score_molecule(analogs, fps, {5: 0}, fps, powers=(1.0,), top_k=1)[1.0]
    assert out[0, 0] == pytest.approx(0.9)
    assert score_molecule([(np.array([7]), np.array([0.9]))], fps, {5: 0}, fps)[3.0].sum() == 0.0


def test_index_search_respects_shift_window_and_mask(tmp_path) -> None:
    peaks = [[50.0, 70.0, 90.0], [1.0, 0.5, 0.3]]
    rows = pa.table({
        "row": [0, 1, 2, 3],
        "ionization_mode": ["positive", "positive", "positive", "negative"],
        "precursor_mz": [200.0, 380.0, 450.0, 200.0],
        "ms2_mzs": [peaks[0]] * 4,
        "ms2_normalized_intensities": [peaks[1]] * 4,
    })
    pq.write_table(rows, tmp_path / "lib.parquet")
    cfg = AnalogConfig(max_shift_da=200.0, top_k=10)
    index = AnalogIndex.build(tmp_path / "lib.parquet", cfg)
    query = np.array([[50.0, 1.0], [70.0, 0.5], [90.0, 0.3]], dtype=np.float32)
    found, sims = index.search("positive", 210.0, query, cfg)
    assert sorted(found.tolist()) == [0, 1]  # row 2 is 240 Da away, row 3 is negative mode
    assert sims.max() == pytest.approx(1.0, abs=1e-6)
    allowed = np.array([False, True, True, True])
    found, _ = index.search("positive", 210.0, query, cfg, allowed)
    assert found.tolist() == [1]
