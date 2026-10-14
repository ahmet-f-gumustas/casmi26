"""Mass arithmetic: adducts, neutral mass, ppm windows.

All adduct conversions go through ``adduct_spec`` so there is a single source of
truth. A spec is ``(n_M, shift, charge)`` with
``precursor_mz = (n_M * M + shift) / |charge|``, where ``M`` is the neutral-atom
monoisotopic mass of the structure.

Adduct names follow the competition's standardised form, ``[nM±formula...]z±``
(e.g. ``[M+CH2O2-H]-``, ``[2M+Na-2H]-``). Any name of that form is parsed from
its formula; ``ADDUCTS`` lists the common ones explicitly plus a few aliases.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- Physical constants (Da) ---
PROTON = 1.007276467
ELECTRON = 0.000548580

# --- Monoisotopic masses of the most abundant isotope (Da) ---
ELEMENTS: dict[str, float] = {
    "H": 1.00782503207,
    "Li": 7.01600455,
    "B": 11.0093054,
    "C": 12.0,
    "N": 14.0030740048,
    "O": 15.99491461956,
    "F": 18.99840322,
    "Na": 22.9897692809,
    "Mg": 23.985041700,
    "Si": 27.9769265325,
    "P": 30.97376163,
    "S": 31.97207100,
    "Cl": 34.96885268,
    "K": 38.96370668,
    "Ca": 39.96259098,
    "Fe": 55.9349375,
    "Cu": 62.9295975,
    "Zn": 63.9291422,
    "Br": 78.9183371,
    "Se": 79.9165213,
    "Ag": 106.905097,
    "I": 126.904473,
}
H = ELEMENTS["H"]
C = ELEMENTS["C"]
N = ELEMENTS["N"]
O = ELEMENTS["O"]
NA = ELEMENTS["Na"]
CL = ELEMENTS["Cl"]
K = ELEMENTS["K"]

# --- Neutral groups (Da) ---
NH4 = N + 4 * H
H2O = 2 * H + O
FORMATE = C + H + 2 * O  # HCOO
ACETATE = 2 * C + 3 * H + 2 * O  # CH3COO

_ADDUCT_RE = re.compile(r"^\[(\d*)M((?:[+-]\d*[A-Z][A-Za-z0-9]*)*)\](\d*)([+-])$")
_TERM_RE = re.compile(r"([+-])(\d*)([A-Z][A-Za-z0-9]*)")
_ATOM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str) -> float | None:
    """Monoisotopic mass of a plain formula such as ``CH2O2``, or ``None`` if unknown."""
    if not formula or "".join(m.group(0) for m in _ATOM_RE.finditer(formula)) != formula:
        return None
    mass = 0.0
    for symbol, count in _ATOM_RE.findall(formula):
        if symbol not in ELEMENTS:
            return None
        mass += ELEMENTS[symbol] * int(count or 1)
    return mass


@lru_cache(maxsize=None)
def parse_adduct(adduct: str) -> tuple[int, float, int] | None:
    """``(n_M, shift, charge)`` for a ``[nM±formula...]z±`` name, or ``None``.

    Added/lost groups are neutral atoms; the ion's electron balance is
    ``-charge * ELECTRON``, so ``[M+H]+`` gives ``M + PROTON``.
    """
    match = _ADDUCT_RE.match(adduct.strip()) if isinstance(adduct, str) else None
    if match is None:
        return None
    n_m_str, terms, z_str, sign = match.groups()
    delta = 0.0
    for term_sign, mult, formula in _TERM_RE.findall(terms):
        mass = formula_mass(formula)
        if mass is None:
            return None
        delta += (1 if term_sign == "+" else -1) * int(mult or 1) * mass
    charge = int(z_str or 1) * (1 if sign == "+" else -1)
    return int(n_m_str or 1), delta - charge * ELECTRON, charge


_KNOWN_ADDUCTS = (
    # the ten adducts used in the test set
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
    # frequent in train
    "[M]+", "[M]-", "[M-H2O]+", "[M+2H]2+", "[M+3H]3+", "[M-2H]2-", "[M+2Na-H]+",
    "[2M+H]+", "[2M+Na]+", "[2M+K]+", "[2M+NH4]+", "[2M-H]-", "[2M+Na-2H]-",
    "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[M+C2H4O2-H]-", "[M-CH3]-", "[M+Br]-",
    "[M+Li]+",
    # same ions under other common spellings
    "[M+H-H2O]+", "[M-H-H2O]-", "[M+HCOO]-", "[M+CH3COO]-",
)
ADDUCTS: dict[str, tuple[int, float, int]] = {name: parse_adduct(name) for name in _KNOWN_ADDUCTS}
ADDUCTS["[M+FA-H]-"] = ADDUCTS["[M+CH2O2-H]-"]
TEST_ADDUCTS = _KNOWN_ADDUCTS[:10]


def adduct_spec(adduct: str) -> tuple[int, float, int]:
    """``(n_M, shift, charge)`` for ``adduct``; raises ``KeyError`` if unknown."""
    spec = ADDUCTS.get(adduct) or parse_adduct(adduct)
    if spec is None:
        raise KeyError(f"unknown adduct {adduct!r}; add it to casmi.chem.ADDUCTS")
    return spec


def neutral_mass(mz: ArrayLike, adduct: str) -> float | NDArray[np.float64]:
    """Neutral monoisotopic mass M from an observed precursor m/z."""
    n_m, shift, charge = adduct_spec(adduct)
    return (np.asarray(mz, dtype=np.float64) * abs(charge) - shift) / n_m


def precursor_mz(mass: ArrayLike, adduct: str) -> float | NDArray[np.float64]:
    """Theoretical precursor m/z for neutral mass ``mass`` under ``adduct``."""
    n_m, shift, charge = adduct_spec(adduct)
    return (n_m * np.asarray(mass, dtype=np.float64) + shift) / abs(charge)


def exact_mass(smiles: str) -> float | None:
    """Neutral-atom monoisotopic mass of ``smiles`` (the ``M`` in adduct formulas).

    RDKit's ``ExactMolWt`` subtracts electron mass for charged structures; that
    is added back so a permanently charged cation under ``[M]+`` gets the right m/z.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Descriptors.ExactMolWt(mol) + Chem.GetFormalCharge(mol) * ELECTRON


def ppm_window(mass: float, ppm: float) -> tuple[float, float]:
    """Inclusive ``(lo, hi)`` mass bounds for a ±``ppm`` tolerance."""
    delta = mass * ppm * 1e-6
    return mass - delta, mass + delta


def ppm_error(obs: ArrayLike, theo: ArrayLike) -> float | NDArray[np.float64]:
    """Signed mass error of ``obs`` relative to ``theo`` in ppm."""
    theo = np.asarray(theo, dtype=np.float64)
    return (np.asarray(obs, dtype=np.float64) - theo) / theo * 1e6


def mass_window_indices(sorted_masses: NDArray[np.float64], mass: float, ppm: float) -> NDArray[np.intp]:
    """Indices into ascending ``sorted_masses`` that fall within ±``ppm`` of ``mass``."""
    lo, hi = ppm_window(mass, ppm)
    left = np.searchsorted(sorted_masses, lo, side="left")
    right = np.searchsorted(sorted_masses, hi, side="right")
    return np.arange(left, right)
