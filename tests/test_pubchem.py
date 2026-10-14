import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from casmi.pubchem import PubChemIndex, murcko

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_pool import select_pubchem  # noqa: E402


@pytest.fixture
def index(tmp_path) -> PubChemIndex:
    smiles = ["CCO", "c1ccccc1O", "CC(=O)O", "OCC1OC(O)C(O)C(O)C1O"]
    masses = np.array([46.04, 94.04, 60.02, 180.06])
    blob = "".join(smiles).encode()
    off = np.concatenate([[0], np.cumsum([len(s) for s in smiles])]).astype(np.int64)
    (tmp_path / "smiles.txt").write_bytes(blob)
    np.save(tmp_path / "off.npy", off)
    np.save(tmp_path / "len.npy", np.array([len(s) for s in smiles], dtype=np.uint16))
    order = np.argsort(masses)
    np.save(tmp_path / "mass_sorted.npy", masses[order])
    np.save(tmp_path / "order.npy", order.astype(np.int64))
    return PubChemIndex(tmp_path)


def test_index_reads_rows_in_mass_order(index: PubChemIndex) -> None:
    assert [index.smiles(i) for i in range(len(index))] == ["CCO", "CC(=O)O", "c1ccccc1O", "OCC1OC(O)C(O)C(O)C1O"]
    assert list(index.window(60.0, 0.05)) == [1]
    assert list(index.mass_range(50, 100)) == [1, 2]


def test_murcko() -> None:
    assert murcko(Chem.MolFromSmiles("c1ccccc1CCO")) == "c1ccccc1"
    assert murcko(Chem.MolFromSmiles("CCO")) == ""


def test_select_pubchem_budget_and_mixtures() -> None:
    df = pd.DataFrame({
        "smiles": ["A", "B", "C", "D", "E.Cl", "F"],
        "np_score": [0.2, 1.5, 0.9, 0.3, 3.0, 2.0],
        "scaffold_hit": [True, False, False, False, False, True],
    })
    kept, t = select_pubchem(df, 4)
    assert set(kept["smiles"]) == {"A", "F", "B", "C"}  # scaffold hits first, then highest NP; mixtures dropped
    assert t == pytest.approx(0.9)
