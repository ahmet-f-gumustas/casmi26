"""Mass arithmetic: adducts, neutral mass, ppm windows.

All adduct conversions go through ``ADDUCTS`` so there is a single source of
truth. Each entry is ``(n_M, shift, charge)`` with
``precursor_mz = (n_M * M + shift) / |charge|``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# --- Physical constants (Da) ---
PROTON = 1.007276467
ELECTRON = 0.000548580

# --- Monoisotopic atom masses (Da) ---
H = 1.00782503207
C = 12.0
N = 14.0030740048
O = 15.99491461956
NA = 22.9897692809
CL = 34.96885268
K = 38.96370668

# --- Neutral groups (Da) ---
NH4 = N + 4 * H
H2O = 2 * H + O
FORMATE = C + H + 2 * O  # HCOO
ACETATE = 2 * C + 3 * H + 2 * O  # CH3COO

ADDUCTS: dict[str, tuple[int, float, int]] = {
    # positive mode
    "[M+H]+": (1, PROTON, 1),
    "[M+Na]+": (1, NA - ELECTRON, 1),
    "[M+K]+": (1, K - ELECTRON, 1),
    "[M+NH4]+": (1, NH4 - ELECTRON, 1),
    "[M+H-H2O]+": (1, PROTON - H2O, 1),
    "[M]+": (1, -ELECTRON, 1),
    "[M+2H]2+": (1, 2 * PROTON, 2),
    "[2M+H]+": (2, PROTON, 1),
    "[2M+Na]+": (2, NA - ELECTRON, 1),
    # negative mode
    "[M-H]-": (1, -PROTON, -1),
    "[M+Cl]-": (1, CL + ELECTRON, -1),
    "[M+HCOO]-": (1, FORMATE + ELECTRON, -1),
    "[M+FA-H]-": (1, FORMATE + ELECTRON, -1),  # == [M+HCOO]-
    "[M+CH3COO]-": (1, ACETATE + ELECTRON, -1),
    "[M-H-H2O]-": (1, -PROTON - H2O, -1),
    "[2M-H]-": (2, -PROTON, -1),
}


def _adduct(adduct: str) -> tuple[int, float, int]:
    try:
        return ADDUCTS[adduct]
    except KeyError:
        raise KeyError(f"unknown adduct {adduct!r}; add it to casmi.chem.ADDUCTS") from None


def neutral_mass(mz: ArrayLike, adduct: str) -> float | NDArray[np.float64]:
    """Neutral monoisotopic mass M from an observed precursor m/z."""
    n_m, shift, charge = _adduct(adduct)
    return (np.asarray(mz, dtype=np.float64) * abs(charge) - shift) / n_m


def precursor_mz(mass: ArrayLike, adduct: str) -> float | NDArray[np.float64]:
    """Theoretical precursor m/z for neutral mass ``mass`` under ``adduct``."""
    n_m, shift, charge = _adduct(adduct)
    return (n_m * np.asarray(mass, dtype=np.float64) + shift) / abs(charge)


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
