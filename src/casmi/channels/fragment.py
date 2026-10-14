"""Fragmentation channel: how much of the observed spectrum a candidate's bond breaks explain.

Fragments come from cutting one bond (bridges only — a ring bond alone does not split
the molecule) or two bonds (two bridges, or two bonds of the same ring). Each fragment
is a set of heavy atoms with their hydrogens; ion m/z values allow ±2 hydrogen shifts
around the protonated (positive) or deprotonated (negative) fragment. A query spectrum's
score is the fraction of its (cleaned) intensity within ``tol_da`` of some fragment ion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from casmi.chem import ELEMENTS, PROTON

H_MASS = ELEMENTS["H"]
_PT = Chem.GetPeriodicTable()


@dataclass(frozen=True)
class FragmentConfig:
    max_cuts: int = 2
    h_shifts: tuple[int, ...] = (-2, -1, 0, 1, 2)
    tol_da: float = 0.01
    max_bonds: int = 160  # larger molecules: single cuts and same-ring pairs only
    min_fragment_mass: float = 40.0


def _components(adj: list[int], n: int, removed: Sequence[tuple[int, int]]) -> list[int]:
    adj = list(adj)
    for i, j in removed:
        adj[i] &= ~(1 << j)
        adj[j] &= ~(1 << i)
    unseen, comps = (1 << n) - 1, []
    while unseen:
        comp = frontier = unseen & -unseen
        while frontier:
            nxt, f = 0, frontier
            while f:
                low = f & -f
                nxt |= adj[low.bit_length() - 1]
                f ^= low
            frontier = nxt & ~comp
            comp |= frontier
        comps.append(comp)
        unseen &= ~comp
    return comps


def fragment_masses(smiles: str, cfg: FragmentConfig = FragmentConfig()) -> NDArray[np.float64]:
    """Sorted unique neutral fragment masses (heavy atoms + their hydrogens) from 1–2 bond cuts."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() < 2:
        return np.zeros(0)
    n = mol.GetNumAtoms()
    if n > 600:
        return np.zeros(0)
    atom_mass = [_PT.GetMostCommonIsotopeMass(a.GetAtomicNum()) + a.GetTotalNumHs() * H_MASS for a in mol.GetAtoms()]
    adj = [0] * n
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i] |= 1 << j
        adj[j] |= 1 << i
        edges.append((i, j, b.IsInRing(), b.GetIdx()))
    ring_info = mol.GetRingInfo()
    bond_rings = [set(r) for r in ring_info.BondRings()]

    def mass(mask: int) -> float:
        total, m = 0.0, mask
        while m:
            low = m & -m
            total += atom_mass[low.bit_length() - 1]
            m ^= low
        return total

    masses: set[float] = set()
    bridges = [(i, j) for i, j, ring, _ in edges if not ring]
    for e in bridges:
        for c in _components(adj, n, [e]):
            masses.add(mass(c))
    if cfg.max_cuts >= 2:
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if len(edges) <= cfg.max_bonds:
            pairs += [(bridges[a], bridges[b]) for a in range(len(bridges)) for b in range(a + 1, len(bridges))]
        by_idx = {idx: (i, j) for i, j, _, idx in edges}
        seen: set[tuple[int, int]] = set()
        for ring in bond_rings:
            r = sorted(ring)
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    if (r[a], r[b]) not in seen:
                        seen.add((r[a], r[b]))
                        pairs.append((by_idx[r[a]], by_idx[r[b]]))
        for e1, e2 in pairs:
            comps = _components(adj, n, [e1, e2])
            if len(comps) > 1:
                for c in comps:
                    masses.add(mass(c))
    out = np.array(sorted(m for m in masses if m >= cfg.min_fragment_mass))
    return np.unique(np.round(out, 6))


def ion_masses(fragments: NDArray[np.float64], mode: str, cfg: FragmentConfig = FragmentConfig()) -> NDArray[np.float64]:
    """Sorted fragment-ion m/z values for an ionization mode, with hydrogen shifts."""
    charge = PROTON if mode == "positive" else -PROTON
    shifts = np.array(cfg.h_shifts, dtype=np.float64) * H_MASS + charge
    return np.sort((fragments[:, None] + shifts[None, :]).ravel())


def explained_intensity(ions: NDArray[np.float64], peaks: NDArray[np.float32], tol_da: float) -> float:
    """Fraction of total peak intensity within ``tol_da`` of some ion in sorted ``ions``."""
    if not len(peaks) or not len(ions):
        return 0.0
    mz, it = peaks[:, 0].astype(np.float64), peaks[:, 1].astype(np.float64)
    pos = np.searchsorted(ions, mz)
    left = np.abs(mz - ions[np.clip(pos - 1, 0, len(ions) - 1)])
    right = np.abs(ions[np.clip(pos, 0, len(ions) - 1)] - mz)
    hit = np.minimum(left, right) <= tol_da
    total = it.sum()
    return float(it[hit].sum() / total) if total > 0 else 0.0
