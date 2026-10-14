import numpy as np
import pandas as pd
import pytest

from casmi.candidates import CandidatePool
from casmi.chem import precursor_mz
from casmi.pipeline import PipelineConfig, candidate_windows, clean_queries


def test_clean_queries_applies_test_peak_rule() -> None:
    q = pd.DataFrame({
        "molecule_id": ["m", "m"], "precursor_mz": [200.0, 300.0],
        "ms2_mzs": [np.array([50.0, 150.0, 202.5]), np.array([100.0])],
        "ms2_normalized_intensities": [np.array([0.5, 0.25, 1.0]), np.array([1.0])],
    })
    out = clean_queries(q)
    assert out["ms2_mzs"].iloc[0].tolist() == [50.0, 150.0]
    assert out["ms2_normalized_intensities"].iloc[0].tolist() == pytest.approx([1.0, 0.5])
    assert out["ms2_mzs"].iloc[1].tolist() == [100.0]
    assert q["ms2_mzs"].iloc[0].tolist() == [50.0, 150.0, 202.5]  # input untouched


def test_candidate_windows_use_offset_and_ppm() -> None:
    masses = np.array([194.0790, 194.0804, 194.0820])
    pool = CandidatePool(masses=masses, meta=pd.DataFrame({"key": ["A", "B", "C"], "smiles": ["C"] * 3}))
    q = pd.DataFrame({"molecule_id": ["m"], "adduct": ["[M+H]+"],
                      "precursor_mz": [precursor_mz(194.0804, "[M+H]+") * (1 + 2e-6)]})
    assert candidate_windows(q, pool, PipelineConfig(ppm=3, offset_ppm=2)) == {"m": ["B"]}
    assert candidate_windows(q, pool, PipelineConfig(ppm=10, offset_ppm=2)) == {"m": ["A", "B", "C"]}
