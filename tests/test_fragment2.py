import numpy as np

from casmi.channels.fragment2 import (
    NEUTRAL_LOSSES,
    Fragment2Config,
    _formula_mass,
    explained_weighted,
    explanations,
    fragment_tree,
    ion_levels,
)
from casmi.chem import PROTON


def test_tree_depths_and_the_molecule() -> None:
    tree = fragment_tree("CCOC(=O)c1ccccc1")  # ethyl benzoate
    depths = {d for _, _, d in tree}
    assert depths == {0, 1, 2, 3}
    mol = [t for t in tree if t[2] == 0][0]
    assert abs(mol[0] - 150.0681) < 1e-3 and mol[1] == (9, 10, 0, 2)
    assert fragment_tree("CCOC(=O)c1ccccc1", Fragment2Config(max_depth=1)) and all(
        d <= 1 for _, _, d in fragment_tree("CCOC(=O)c1ccccc1", Fragment2Config(max_depth=1)))


def test_losses_need_the_atoms() -> None:
    hexose = _formula_mass(NEUTRAL_LOSSES["hexose"])
    assert abs(hexose - 162.0528) < 1e-3
    glucoside = "OCC1OC(Oc2ccccc2)C(O)C(O)C1O"  # phenyl glucoside, 256.095
    masses = np.concatenate([m for _, m in explanations(fragment_tree(glucoside))])
    assert np.isclose(masses, 256.0947 - hexose, atol=1e-3).any()  # phenol after the hexose loss
    no_n = np.concatenate([m for _, m in explanations(fragment_tree("CCCCCCO"))])
    assert not np.isclose(no_n, 102.1045 - 17.0265, atol=1e-3).any()  # no NH3 loss without nitrogen


def test_best_explanation_per_peak() -> None:
    expl = [(1.0, np.array([100.0])), (0.5, np.array([150.0]))]
    levels = ion_levels(expl, "positive", Fragment2Config(h_shifts=(0,)))
    peaks = np.array([[100.0 + PROTON, 3.0], [150.0 + PROTON, 1.0], [123.0, 4.0]], dtype=np.float32)
    assert abs(explained_weighted(levels, peaks, Fragment2Config(h_shifts=(0,))) - (3.0 + 0.5) / 8.0) < 1e-6
    tight = Fragment2Config(h_shifts=(0,), tol_da=0.0001, tol_ppm=1.0)
    off = peaks.copy()
    off[0, 0] += 0.005
    assert explained_weighted(levels, off, tight) < explained_weighted(levels, off, Fragment2Config(h_shifts=(0,)))
