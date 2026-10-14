"""Read access to the mass-sorted PubChem SMILES index (``prvsiyan/pubchem-npformula-massindex``, CC0).

Layout (inferred and checked against RDKit masses): ``smiles.txt`` holds the SMILES back
to back; ``off.npy`` (int64, n+1) and ``len.npy`` (uint16, n) give each row's byte
offset and length; ``mass_sorted.npy`` (float64) is the neutral monoisotopic mass in
ascending order and ``order.npy`` (int64) maps a sorted position to its row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


class PubChemIndex:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        load = lambda name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)  # noqa: E731
        self.off, self.length, self.mass, self.order = load("off"), load("len"), load("mass_sorted"), load("order")
        self._fh = None

    def __len__(self) -> int:
        return len(self.mass)

    def _file(self):
        if self._fh is None:
            self._fh = open(self.root / "smiles.txt", "rb")  # noqa: SIM115 — one handle per process
        return self._fh

    def smiles(self, pos: int) -> str:
        """SMILES at sorted position ``pos``."""
        row = int(self.order[pos])
        f = self._file()
        f.seek(int(self.off[row]))
        return f.read(int(self.length[row])).decode("ascii", errors="replace")

    def window(self, mass: float, tol_da: float) -> range:
        """Sorted positions with |mass − m| ≤ tol_da."""
        lo = int(np.searchsorted(self.mass, mass - tol_da, side="left"))
        hi = int(np.searchsorted(self.mass, mass + tol_da, side="right"))
        return range(lo, hi)

    def mass_range(self, lo: float, hi: float) -> range:
        return range(int(np.searchsorted(self.mass, lo, side="left")), int(np.searchsorted(self.mass, hi, side="right")))


def murcko(mol: Chem.Mol) -> str:
    """Murcko scaffold SMILES ('' for acyclic molecules or on failure)."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except (ValueError, RuntimeError):
        return ""


def np_and_scaffold(smiles_list: list[str], np_model, scaffolds: set[str]) -> tuple[NDArray, NDArray]:
    """RDKit NP-likeness score and COCONUT-scaffold hit per SMILES (NaN / False if unparsable)."""
    from rdkit.Contrib.NP_Score import npscorer

    scores = np.full(len(smiles_list), np.nan)
    hits = np.zeros(len(smiles_list), dtype=bool)
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        scores[i] = npscorer.scoreMol(mol, np_model)
        sc = murcko(mol)
        hits[i] = bool(sc) and sc in scaffolds
    return scores, hits
