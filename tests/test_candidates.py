import numpy as np
import pandas as pd
import pytest

from casmi.candidates import CandidatePool, molecule_masses
from casmi.chem import precursor_mz

MASSES = np.array([100.0, 194.0800, 194.0804, 194.0808, 194.0830, 300.0])
META = pd.DataFrame({"key": [f"K{i}" for i in range(len(MASSES))], "smiles": ["C"] * len(MASSES)})


@pytest.fixture
def pool() -> CandidatePool:
    return CandidatePool(masses=MASSES, meta=META)


def test_window_without_offset(pool: CandidatePool) -> None:
    assert pool.window(194.0804, ppm=3, offset_ppm=0).tolist() == [1, 2, 3]


def test_window_corrects_instrument_offset(pool: CandidatePool) -> None:
    observed = 194.0830 * (1 + 2e-6)  # reads 2 ppm high
    assert pool.window(observed, ppm=1, offset_ppm=2).tolist() == [4]
    assert pool.window(observed, ppm=1, offset_ppm=0).tolist() == []


def test_exclude_hides_keys_and_composes(pool: CandidatePool) -> None:
    sub = pool.exclude(["K2"]).exclude(["K3"])
    assert sub.window(194.0804, ppm=3, offset_ppm=0).tolist() == [1]
    assert pool.window(194.0804, ppm=3, offset_ppm=0).tolist() == [1, 2, 3]  # original untouched


def test_molecule_masses_median_over_adducts() -> None:
    m = 194.080376
    q = pd.DataFrame({
        "molecule_id": ["a", "a", "a", "b"],
        "adduct": ["[M+H]+", "[M+Na]+", "[M-H]-", "[M+H]+"],
        "precursor_mz": [precursor_mz(m, "[M+H]+"), precursor_mz(m, "[M+Na]+"), precursor_mz(m, "[M-H]-"),
                         precursor_mz(300.0, "[M+H]+")],
    })
    out = molecule_masses(q)
    assert out["a"] == pytest.approx(m, abs=1e-9) and out["b"] == pytest.approx(300.0, abs=1e-9)
