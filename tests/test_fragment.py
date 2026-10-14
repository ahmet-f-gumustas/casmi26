import numpy as np
import pytest

from casmi.channels.fragment import FragmentConfig, explained_intensity, fragment_masses, ion_masses
from casmi.chem import ELEMENTS, PROTON

H, C, O = ELEMENTS["H"], ELEMENTS["C"], ELEMENTS["O"]
CFG = FragmentConfig(min_fragment_mass=0.0)


def test_ethanol_fragments() -> None:
    got = fragment_masses("CCO", CFG)
    expected = {C + 3 * H, C + 2 * H + O + H, 2 * C + 5 * H, O + H, C + 2 * H}  # CH3, CH2OH, C2H5, OH, CH2
    assert len(got) == len(expected)
    for m in expected:
        assert np.isclose(got, m, atol=1e-5).any()


def test_ring_needs_two_cuts() -> None:
    single = fragment_masses("C1CCCCC1", FragmentConfig(max_cuts=1, min_fragment_mass=0.0))
    assert len(single) == 0  # no bridge in cyclohexane
    double = fragment_masses("C1CCCCC1", CFG)
    assert np.isclose(double, C + 2 * H, atol=1e-5).any() and np.isclose(double, 5 * (C + 2 * H), atol=1e-5).any()


def test_ion_masses_and_explained_intensity() -> None:
    ions = ion_masses(np.array([100.0]), "positive", FragmentConfig(h_shifts=(0,)))
    assert ions.tolist() == pytest.approx([100.0 + PROTON])
    peaks = np.array([[100.0 + PROTON + 0.005, 0.6], [150.0, 0.4]], dtype=np.float32)
    assert explained_intensity(ions, peaks, 0.01) == pytest.approx(0.6, abs=1e-6)
    assert explained_intensity(ions, peaks, 0.001) == 0.0
    neg = ion_masses(np.array([100.0]), "negative", FragmentConfig(h_shifts=(0,)))
    assert neg.tolist() == pytest.approx([100.0 - PROTON])


def test_invalid_smiles_has_no_fragments() -> None:
    assert fragment_masses("xx").size == 0
