import numpy as np
import pytest

from casmi.chem import precursor_mz
from casmi.clean import C13_SHIFT, FIXED, OK, mass_verdict, strip_peaks

CAFFEINE = 194.080376


def test_strip_peaks_cuts_at_precursor_plus_2_and_renormalises() -> None:
    # spectrum 0: base peak above the cut; spectrum 1: untouched; spectrum 2: nothing left; spectrum 3: empty
    offsets = np.array([0, 3, 5, 6, 6])
    mz = np.array([50.0, 100.0, 102.5, 60.0, 101.9, 300.0])
    inten = np.array([0.5, 0.25, 1.0, 1.0, 0.4, 1.0])
    prec = np.array([100.0, 100.0, 100.0, 100.0])
    off, mz2, int2, scale = strip_peaks(offsets, mz, inten, prec)
    assert off.tolist() == [0, 2, 4, 4, 4]
    assert mz2.tolist() == [50.0, 100.0, 60.0, 101.9]
    assert int2.tolist() == [1.0, 0.5, 1.0, 0.4]
    assert scale.tolist() == [0.5, 1.0, 0.0, 0.0]


def test_boundary_is_inclusive() -> None:
    off, mz2, _, _ = strip_peaks(np.array([0, 2]), np.array([102.0, 102.0001]), np.array([1.0, 0.5]), np.array([100.0]))
    assert mz2.tolist() == [102.0]


def test_mass_verdict_classes() -> None:
    m = np.full(6, CAFFEINE)
    labelled = np.array(["[M+H]+", "[M+H]+", "[M+H]+", "[M+H]+", "[M+H]+", "[Cat]2+"], dtype=object)
    obs = np.array([
        precursor_mz(CAFFEINE, "[M+H]+") * (1 + 3e-6),  # fine
        precursor_mz(CAFFEINE, "[M+Na]+"),  # mislabelled Na adduct
        precursor_mz(CAFFEINE, "[M+H]+") + C13_SHIFT,  # 13C precursor
        precursor_mz(CAFFEINE, "[M+H]+") + 0.05,  # nominal-mass library
        precursor_mz(CAFFEINE, "[M+H]+") + 7.3,  # wrong structure
        100.0,
    ])
    ppm, adduct, status = mass_verdict(obs, m, labelled)
    assert status.tolist() == [OK, FIXED, "13C precursor", "low mass accuracy", "mass mismatch", "unknown adduct"]
    assert adduct[1] == "[M+Na]+" and abs(ppm[1]) < 1e-6
    assert ppm[0] == pytest.approx(3.0, abs=1e-6)


def test_missing_structure_is_rejected() -> None:
    _, _, status = mass_verdict(np.array([195.0]), np.array([np.nan]), np.array(["[M+H]+"], dtype=object))
    assert status.tolist() == ["no structure"]
