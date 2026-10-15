import numpy as np
import pandas as pd
import pytest

from casmi.adducts import alt_masses, same_ion
from casmi.chem import neutral_mass, precursor_mz


def test_same_ion_spellings() -> None:
    assert same_ion("[M+HCOO]-", "[M+CH2O2-H]-") and same_ion("[M+H-H2O]+", "[M-H2O+H]+")
    assert not same_ion("[M+H]+", "[M+Na]+")


def test_alt_masses_per_polarity_and_skips_confirmed_labels() -> None:
    m = 300.1
    q = pd.DataFrame({
        "molecule_id": ["a", "a", "b", "c", "c"],
        "ionization_mode": ["positive", "positive", "negative", "positive", "positive"],
        "adduct": ["[M+H]+", "[M+H]+", "[M+CH2O2-H]-", "[M+H]+", "[M+Na]+"],
        "precursor_mz": [precursor_mz(m, "[M+H]+")] * 2 + [precursor_mz(m, "[M+CH2O2-H]-"),
                                                           precursor_mz(m, "[M+H]+"), precursor_mz(m, "[M+Na]+")],
    })
    out = alt_masses(q)
    assert set(out) == {"a", "b"}  # c has two adduct types that agree
    assert [a for a, _ in out["a"]] == ["[M+Na]+", "[M+NH4]+", "[M+H-H2O]+"]
    na = dict(out["a"])["[M+Na]+"]
    assert na == pytest.approx(neutral_mass(precursor_mz(m, "[M+H]+"), "[M+Na]+"))
    assert [a for a, _ in out["b"]] == ["[M-H]-", "[M+Cl]-"]
    assert dict(out["b"])["[M-H]-"] == pytest.approx(m + 46.0055, abs=1e-3)  # formate read as [M-H]-
    assert np.isfinite([v for _, v in out["a"]]).all()
