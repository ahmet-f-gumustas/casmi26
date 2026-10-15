"""Alternative adduct hypotheses (recall probe 2b).

If a spectrum's adduct label were wrong, its neutral mass — and so the candidate window — would
be wrong and the answer could never be ranked. Per molecule, the other adducts of its polarity
give alternative neutral masses; their windows (minus the main window) hold the alternative-adduct
candidates, which ``casmi.tier.slot_merge`` may let into a few gated slots.

A molecule whose spectra carry two or more adduct types is skipped: their masses agree (the
labels confirm each other), so a mislabel would have to hit every adduct consistently.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from casmi.candidates import CandidatePool
from casmi.chem import adduct_spec, neutral_mass

ALT_ADDUCTS = {"positive": ("[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+H-H2O]+"),
               "negative": ("[M-H]-", "[M+HCOO]-", "[M+Cl]-")}


def same_ion(a: str, b: str) -> bool:
    """True when two adduct names describe the same ion (e.g. ``[M+HCOO]-`` and ``[M+CH2O2-H]-``)."""
    (na, sa, za), (nb, sb, zb) = adduct_spec(a), adduct_spec(b)
    return na == nb and za == zb and abs(sa - sb) < 1e-4


def alt_masses(queries: pd.DataFrame, key: str = "molecule_id") -> dict[str, list[tuple[str, float]]]:
    """Per single-adduct molecule: ``(adduct, neutral mass)`` for each other adduct of its polarity."""
    out = {}
    for mol_id, g in queries.groupby(key):
        labels = g["adduct"].unique()
        if any(not same_ion(labels[0], a) for a in labels[1:]):
            continue
        prec = float(np.median(g["precursor_mz"]))
        alts = ALT_ADDUCTS.get(str(g["ionization_mode"].iloc[0]), ())
        out[mol_id] = [(a, float(neutral_mass(prec, a))) for a in alts if not same_ion(a, labels[0])]
    return out


def alt_windows(queries: pd.DataFrame, pool: CandidatePool, main: Mapping[str, list[str]], ppm: float,
                offset_ppm: float) -> dict[str, list[str]]:
    """Pool keys in the alternative-adduct windows of each molecule, minus its main window."""
    keys = pool.meta["key"].to_numpy()
    out = {}
    for mol_id, alts in alt_masses(queries).items():
        seen = set(main.get(mol_id, []))
        found = []
        for _, mass in alts:
            for k in keys[pool.window(mass, ppm, offset_ppm)]:
                if k not in seen:
                    seen.add(k)
                    found.append(k)
        out[mol_id] = found
    return out
