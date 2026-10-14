"""Train cleaning rules: make library spectra look like test spectra.

The data page says test spectra had only minimal cleaning: spectra with a
precursor mass inconsistent with the structure and adduct were dropped, spectra
with a base peak below 1,000 raw counts were dropped, and peaks above
precursor + 2 Da were removed. ``enveda-np-examples`` went through the same
pipeline and its largest peak excess is 1.9999 Da, so the cut is
``mz > precursor_mz + 2.0``; intensities are renormalised to a base peak of 1.

Mass consistency uses our own ppm (``casmi.chem``), not the host's
``precursor_error_ppm`` column, which is biased by ~1.6 ppm and broken for
formate adducts in some libraries (see notes/ADDUCTS.md).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from casmi.chem import ADDUCTS, parse_adduct, ppm_error, precursor_mz

PEAK_MARGIN_DA = 2.0
MAX_ABS_PPM = 20.0
MIN_BASE_PEAK = 1000.0
LOW_RES_DA = 0.5
C13_SHIFT = 1.0033548378

OK, FIXED = "ok", "fixed"
REJECT_REASONS = ("unknown adduct", "no structure", "ambiguous adduct", "13C precursor", "low mass accuracy",
                  "mass mismatch", "base peak < 1000", "no peaks left")

# One name per distinct ion, in ADDUCTS order (aliases dropped).
_ALT_IONS = list({ADDUCTS[name]: name for name in reversed(list(ADDUCTS))}.values())[::-1]


def strip_peaks(
    offsets: NDArray[np.integer], mz: NDArray[np.float64], intensity: NDArray[np.float64],
    precursor: NDArray[np.float64], margin: float = PEAK_MARGIN_DA,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Drop peaks with ``mz > precursor + margin`` and renormalise each spectrum to max 1.

    ``offsets`` are list offsets (len = n_spectra + 1). Returns new offsets, mz,
    intensity and ``scale`` = new base peak / old base peak per spectrum (0 when
    no peak is left), so raw base-peak intensity becomes ``bpi * scale``.
    """
    n = len(offsets) - 1
    parent = np.repeat(np.arange(n), np.diff(offsets))
    keep = mz <= precursor[parent] + margin
    counts = np.bincount(parent[keep], minlength=n)
    new_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    new_mz, new_int = mz[keep], intensity[keep]
    row_max = np.zeros(n)
    nonempty = counts > 0
    if nonempty.any():
        row_max[nonempty] = np.maximum.reduceat(new_int, new_offsets[:-1][nonempty])
    old_max = np.zeros(n)
    had = np.diff(offsets) > 0
    if had.any():
        old_max[had] = np.maximum.reduceat(intensity, np.asarray(offsets[:-1])[had])
    new_int = new_int / np.repeat(np.where(nonempty, row_max, 1.0), counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(old_max > 0, row_max / old_max, 0.0)
    return new_offsets, new_mz, new_int, scale


def mass_verdict(
    obs_mz: NDArray[np.float64], mass: NDArray[np.float64], adduct: NDArray[np.object_], max_ppm: float = MAX_ABS_PPM,
) -> tuple[NDArray[np.float64], NDArray[np.object_], NDArray[np.object_]]:
    """Check each precursor against its labelled structure and adduct.

    Returns ``(ppm, adduct_out, status)``. ``status`` is ``ok``, ``fixed`` (exactly
    one other same-polarity adduct explains the mass within ``max_ppm``; ``adduct_out``
    holds it and ``ppm`` is recomputed) or a reject reason from ``REJECT_REASONS``.
    """
    n = len(obs_mz)
    ppm = np.full(n, np.nan)
    out = np.asarray(adduct, dtype=object).copy()
    status = np.full(n, OK, dtype=object)
    status[~np.isfinite(mass)] = "no structure"
    for name in np.unique(out[status == OK].astype(str)):
        idx = np.flatnonzero((out == name) & (status == OK))
        spec = ADDUCTS.get(name) or parse_adduct(name)
        if spec is None:
            status[idx] = "unknown adduct"
            continue
        m, mz = mass[idx], obs_mz[idx]
        ppm[idx] = ppm_error(mz, precursor_mz(m, name))
        bad = idx[np.abs(ppm[idx]) > max_ppm]
        if not len(bad):
            continue
        m, mz = mass[bad], obs_mz[bad]
        alts = [a for a in _ALT_IONS if a != name and np.sign(ADDUCTS[a][2]) == np.sign(spec[2])]
        alt_ppm = np.vstack([ppm_error(mz, precursor_mz(m, a)) for a in alts]) if alts else np.zeros((1, len(bad)))
        hits = (np.abs(alt_ppm) <= max_ppm) & bool(alts)
        n_hits = hits.sum(axis=0)
        one = n_hits == 1
        best = np.argmax(hits, axis=0)
        out[bad[one]] = np.asarray(alts, dtype=object)[best[one]]
        ppm[bad[one]] = alt_ppm[best[one], np.flatnonzero(one)]
        status[bad[one]] = FIXED
        rest = ~one
        status[bad[rest & (n_hits > 1)]] = "ambiguous adduct"
        none = n_hits == 0
        theo = precursor_mz(m, name)
        c13 = none & (np.abs(ppm_error(mz, theo + C13_SHIFT / abs(spec[2]))) <= max_ppm)
        low = none & ~c13 & (np.abs(mz - theo) < LOW_RES_DA)
        status[bad[c13]] = "13C precursor"
        status[bad[low]] = "low mass accuracy"
        status[bad[none & ~c13 & ~low]] = "mass mismatch"
    return ppm, out, status
