"""Fragmentation channel v2 (MetFrag-lite): a fragment tree to depth 3 with neutral losses.

A break event cuts one bridge bond or two bonds of the same ring; fragments are the connected
pieces (heavy atoms with their hydrogens), and each piece can break again, up to ``max_depth``
events from the molecule. Common neutral losses (H2O, CO, CO2, NH3, CH2O, C2H4O2 and the hexose,
pentose and deoxyhexose sugar units) are taken from the molecule and from every fragment when its
elemental composition allows; the precursor also loses pairs of them. Ions allow ±2 hydrogen shifts
around the protonated (positive) or deprotonated (negative) neutral.

Each explanation has a weight (deeper breaks and losses weigh less). A spectrum's score is its
intensity-weighted explained fraction: every peak takes the weight of its best explanation within
the m/z tolerance, Σ intensity · weight / Σ intensity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from casmi.chem import ELEMENTS, PROTON

H_MASS = ELEMENTS["H"]
COUNTED = ("C", "H", "N", "O")  # composition checked before a neutral loss
_PT = Chem.GetPeriodicTable()


def _formula_mass(counts: dict[str, int]) -> float:
    return sum(ELEMENTS[e] * k for e, k in counts.items())


NEUTRAL_LOSSES: dict[str, dict[str, int]] = {
    "H2O": {"H": 2, "O": 1}, "CO": {"C": 1, "O": 1}, "CO2": {"C": 1, "O": 2}, "NH3": {"N": 1, "H": 3},
    "CH2O": {"C": 1, "H": 2, "O": 1}, "C2H4O2": {"C": 2, "H": 4, "O": 2},
    "hexose": {"C": 6, "H": 10, "O": 5}, "pentose": {"C": 5, "H": 8, "O": 4}, "deoxyhexose": {"C": 6, "H": 10, "O": 4},
}


@dataclass(frozen=True)
class Fragment2Config:
    max_depth: int = 3
    max_atoms_depth3: int = 60  # larger molecules stop at depth 2
    max_fragments: int = 3000
    min_fragment_mass: float = 40.0
    depth_weight: tuple[float, ...] = (1.0, 1.0, 0.85, 0.7)  # by depth (0 = the molecule)
    loss_weight: float = 0.9
    use_losses: bool = True
    h_shifts: tuple[int, ...] = (-2, -1, 0, 1, 2)
    tol_da: float = 0.01
    tol_ppm: float | None = None  # if set: tolerance = max(tol_da, ppm · m/z)
    sqrt_intensity: bool = False


def _components(adj: Sequence[int], within: int, removed: Sequence[tuple[int, int]]) -> list[int]:
    """Connected atom sets of the subgraph on ``within`` after removing ``removed`` bonds."""
    over = {}
    for i, j in removed:
        over[i] = over.get(i, adj[i]) & ~(1 << j)
        over[j] = over.get(j, adj[j]) & ~(1 << i)
    unseen, comps = within, []
    while unseen:
        comp = frontier = unseen & -unseen
        while frontier:
            nxt, f = 0, frontier
            while f:
                low = f & -f
                a = low.bit_length() - 1
                nxt |= over.get(a, adj[a])
                f ^= low
            frontier = nxt & within & ~comp
            comp |= frontier
        comps.append(comp)
        unseen &= ~comp
    return comps


def fragment_tree(smiles: str, cfg: Fragment2Config = Fragment2Config()) -> list[tuple[float, tuple[int, ...], int]]:
    """``(neutral mass, (C, H, N, O) counts, depth)`` for the molecule (depth 0) and its fragments."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() < 2 or mol.GetNumAtoms() > 600:
        return []
    n = mol.GetNumAtoms()
    atom_mass, atom_comp = [], []
    for a in mol.GetAtoms():
        h = a.GetTotalNumHs()
        atom_mass.append(_PT.GetMostCommonIsotopeMass(a.GetAtomicNum()) + h * H_MASS)
        sym = a.GetSymbol()
        atom_comp.append(tuple((1 if sym == e else 0) + (h if e == "H" else 0) for e in COUNTED))
    adj = [0] * n
    bonds = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i] |= 1 << j
        adj[j] |= 1 << i
        bonds[b.GetIdx()] = (i, j, b.IsInRing())
    events: list[tuple[tuple[tuple[int, int], ...], int]] = [
        (((i, j),), (1 << i) | (1 << j)) for i, j, ring in bonds.values() if not ring]
    seen_pairs = set()
    for ring in mol.GetRingInfo().BondRings():
        r = sorted(ring)
        for x in range(len(r)):
            for y in range(x + 1, len(r)):
                if (r[x], r[y]) in seen_pairs:
                    continue
                seen_pairs.add((r[x], r[y]))
                (i1, j1, _), (i2, j2, _) = bonds[r[x]], bonds[r[y]]
                events.append((((i1, j1), (i2, j2)), (1 << i1) | (1 << j1) | (1 << i2) | (1 << j2)))

    def mass_comp(mask: int) -> tuple[float, tuple[int, ...]]:
        total, comp, m = 0.0, [0, 0, 0, 0], mask
        while m:
            low = m & -m
            a = low.bit_length() - 1
            total += atom_mass[a]
            c = atom_comp[a]
            comp[0] += c[0]
            comp[1] += c[1]
            comp[2] += c[2]
            comp[3] += c[3]
            m ^= low
        return total, tuple(comp)

    full = (1 << n) - 1
    depth = {full: 0}
    frontier = [full]
    max_depth = cfg.max_depth if n <= cfg.max_atoms_depth3 else min(cfg.max_depth, 2)
    for d in range(1, max_depth + 1):
        nxt = []
        for frag in frontier:
            for removed, atoms in events:
                if atoms & frag != atoms:
                    continue
                comps = _components(adj, frag, removed)
                if len(comps) < 2:
                    continue
                for c in comps:
                    if c not in depth:
                        depth[c] = d
                        nxt.append(c)
            if len(depth) >= cfg.max_fragments:
                break
        if len(depth) >= cfg.max_fragments:
            break
        frontier = nxt
    out = []
    for mask, d in depth.items():
        m, comp = mass_comp(mask)
        if m >= cfg.min_fragment_mass or d == 0:
            out.append((m, comp, d))
    return out


_LOSS = [(_formula_mass(c), tuple(c.get(e, 0) for e in COUNTED)) for c in NEUTRAL_LOSSES.values()]


def explanations(tree: list[tuple[float, tuple[int, ...], int]], cfg: Fragment2Config = Fragment2Config()
                 ) -> list[tuple[float, NDArray[np.float64]]]:
    """``(weight, sorted neutral masses)`` per weight level, highest weight first."""
    by_w: dict[float, list[float]] = {}
    for m, comp, d in tree:
        w = cfg.depth_weight[min(d, len(cfg.depth_weight) - 1)]
        by_w.setdefault(w, []).append(m)
        if not cfg.use_losses:
            continue
        lw = w * cfg.loss_weight
        for lm, lc in _LOSS:
            if all(a >= b for a, b in zip(comp, lc)) and m - lm >= cfg.min_fragment_mass:
                by_w.setdefault(lw, []).append(m - lm)
                if d == 0:  # the precursor also loses pairs
                    rest = tuple(a - b for a, b in zip(comp, lc))
                    for lm2, lc2 in _LOSS:
                        if all(a >= b for a, b in zip(rest, lc2)) and m - lm - lm2 >= cfg.min_fragment_mass:
                            by_w.setdefault(lw * cfg.loss_weight, []).append(m - lm - lm2)
    return [(w, np.unique(np.round(np.array(v), 6))) for w, v in sorted(by_w.items(), reverse=True)]


def ion_levels(expl: list[tuple[float, NDArray[np.float64]]], mode: str, cfg: Fragment2Config = Fragment2Config()
               ) -> list[tuple[float, NDArray[np.float64]]]:
    """Sorted ion m/z per weight level for an ionization mode."""
    charge = PROTON if mode == "positive" else -PROTON
    shifts = np.array(cfg.h_shifts, dtype=np.float64) * H_MASS + charge
    return [(w, np.sort((m[:, None] + shifts[None, :]).ravel())) for w, m in expl]


def explained_weighted(levels: list[tuple[float, NDArray[np.float64]]], peaks: NDArray[np.float32],
                       cfg: Fragment2Config = Fragment2Config()) -> float:
    """Σ intensity · (weight of the best explanation within tolerance) / Σ intensity."""
    if not len(peaks) or not levels:
        return 0.0
    mz, it = peaks[:, 0].astype(np.float64), peaks[:, 1].astype(np.float64)
    if cfg.sqrt_intensity:
        it = np.sqrt(np.maximum(it, 0.0))
    tol = np.maximum(cfg.tol_da, cfg.tol_ppm * 1e-6 * mz) if cfg.tol_ppm else np.full(len(mz), cfg.tol_da)
    best = np.zeros(len(mz))
    open_ = np.ones(len(mz), bool)
    for w, ions in levels:
        if not open_.any():
            break
        if not len(ions):
            continue
        pos = np.searchsorted(ions, mz)
        left = np.abs(mz - ions[np.clip(pos - 1, 0, len(ions) - 1)])
        right = np.abs(ions[np.clip(pos, 0, len(ions) - 1)] - mz)
        hit = open_ & (np.minimum(left, right) <= tol)
        best[hit] = w
        open_ &= ~hit
    total = it.sum()
    return float((it * best).sum() / total) if total > 0 else 0.0
