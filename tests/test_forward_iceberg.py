import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from casmi.config import EXTERNAL_DIR
from casmi.forward import iceberg

WEIGHTS = EXTERNAL_DIR / "ms_pred" / "derived_state_dicts"
GEN, INTEN = WEIGHTS / "iceberg_msg_all_gen_sd.pt", WEIGHTS / "iceberg_msg_all_inten_contr_sd.pt"
ETHYL_BENZOATE = "CCOC(=O)c1ccccc1"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

pytestmark = pytest.mark.skipif(not (GEN.exists() and INTEN.exists()), reason="ICEBERG weights not present")


@pytest.fixture(scope="module")
def model() -> iceberg.Iceberg:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    return iceberg.Iceberg.load(GEN, INTEN, device="cpu")


def _formula(m: iceberg.Molecule, atoms: tuple[int, ...]) -> tuple[int, ...]:
    """Element counts of a fragment, hydrogens from its atoms."""
    v = m.feat[list(atoms), :iceberg.N_ELEM].sum(0).astype(int)
    v[iceberg._H] = int(m.hs[list(atoms)].sum())
    return tuple(v)


def test_load_is_strict_and_checks_computed_buffers(model: iceberg.Iceberg) -> None:
    sd = torch.load(GEN, map_location="cpu", weights_only=True)["state_dict"]
    assert set(model.gen.state_dict()) == set(sd)
    assert set(model.inten.state_dict()) == set(torch.load(INTEN, map_location="cpu", weights_only=True)["state_dict"])
    sd["adduct_embedder"] = sd["adduct_embedder"].clone()
    sd["adduct_embedder"][0, 0] = 0.5
    with pytest.raises(ValueError, match="adduct_embedder"):
        iceberg._fill(iceberg._FragGNN(512, 6, 14), sd, Path(GEN))


def test_predict_shapes_and_order(model: iceberg.Iceberg) -> None:
    out = model.predict([ETHYL_BENZOATE], "[M+H]+", [10.0, 30.0, 50.0])
    assert len(out) == 1 and len(out[0]) == 3
    for mz, it in out[0]:
        assert mz.dtype == np.float64 and it.dtype == np.float32 and len(mz) == len(it) == 100
        assert np.all(np.diff(it) <= 0) and it[0] > 0
    (mz, it), = model.predict([ETHYL_BENZOATE], "[M+H]+", [30.0], top=10)[0]
    assert len(mz) == 10 and np.allclose(it, out[0][1][1][:10])
    assert model.predict([ETHYL_BENZOATE], "[M+H]+", []) == [[]]


def test_none_for_inputs_upstream_cannot_handle(model: iceberg.Iceberg, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = ["", "not a smiles", "CC[Pt]CC", "C[N+](C)(C)C", "CCO.O", "CC(=O)[O-].[Na+]"]
    out = model.predict(bad + [ETHYL_BENZOATE], "[M+H]+", [20.0])
    assert out[:-1] == [None] * len(bad) and out[-1] is not None
    assert model.predict([ETHYL_BENZOATE, CAFFEINE], "[M+K]+", [20.0]) == [None, None]
    assert iceberg.featurise("CC(C)(C)C") is not None
    monkeypatch.setattr(iceberg, "MAX_ATOM_BONDS", 3)  # neopentane's centre has 4 bonds
    assert iceberg.featurise("CC(C)(C)C") is None


def test_canonicalisation_drops_stereo_and_uses_upstream_tautomer_rules(model: iceberg.Iceberg) -> None:
    assert iceberg.canonical_smiles("C/C=C/C(=O)OCC") == iceberg.canonical_smiles("CC=CC(=O)OCC")
    assert iceberg.canonical_smiles("CC(O)=CC(C)=O") == "CC(=O)CC(C)=O"  # keto/enol transform
    assert iceberg.canonical_smiles("Oc1ccccn1") == "Oc1ccccn1"  # not in ms-pred's two-transform set
    a, b = model.predict(["C/C=C/C(=O)OCC", "CC=CC(=O)OCC"], "[M+H]+", [20.0])
    assert np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[0][1], b[0][1])


def test_deterministic_and_energy_dependent(model: iceberg.Iceberg) -> None:
    a = model.predict([CAFFEINE], "[M+H]+", [10.0, 60.0])[0]
    b = model.predict([CAFFEINE], "[M+H]+", [10.0, 60.0])[0]
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    (mz10, it10), (mz60, it60) = a
    assert not (np.array_equal(mz10, mz60) and np.allclose(it10, it60))
    assert model.dag(CAFFEINE, "[M+H]+", 10.0) != model.dag(CAFFEINE, "[M+H]+", 60.0)  # the DAG depends on CE


@pytest.mark.parametrize("smiles, adduct", [(ETHYL_BENZOATE, "[M+H]+"), (CAFFEINE, "[M+Na]+")])
def test_mz_are_sub_formula_masses_below_the_precursor(model: iceberg.Iceberg, smiles: str, adduct: str) -> None:
    m = iceberg.featurise(smiles)
    (mz, it), = model.predict([smiles], adduct, [30.0])[0]
    mz = mz[it > 0]
    heavy = [i for i, n in enumerate(m.form) if n and i != iceberg._H]
    grid = np.array([c for c in itertools.product(*[range(int(m.form[i]) + 1) for i in heavy])])
    masses = grid @ iceberg.MONO_MASSES[heavy]
    h = np.arange(int(m.form[iceberg._H]) + iceberg.MAX_BROKEN + 1) * iceberg.MONO_MASSES[iceberg._H]
    sub = np.sort((masses[:, None] + h[None, :]).ravel())

    def gap(x: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(sub, x), 1, len(sub) - 1)
        return np.minimum(np.abs(sub[i] - x), np.abs(sub[i - 1] - x))

    # each row is a fragment ± k H plus the adduct, or minus an electron (the unshifted m/z group)
    assert np.minimum(gap(mz - iceberg.ION_MASS[adduct]), gap(mz + iceberg.ELECTRON_MASS)).max() < 2e-4
    assert mz.max() <= float(m.form @ iceberg.MONO_MASSES) + iceberg.ION_MASS[adduct] + 1e-3


def test_dag_holds_the_single_bond_breaks_of_ethyl_benzoate(model: iceberg.Iceberg) -> None:
    m = iceberg.featurise(ETHYL_BENZOATE)
    dag = model.dag(ETHYL_BENZOATE, "[M+H]+", 30.0)
    assert dag[0]["atoms"] == tuple(range(m.natoms)) and dag[0]["prob"] == 1.0 and dag[0]["broken"] == 0.0
    assert len(dag) == iceberg.MAX_NODES and all(x["prob"] >= y["prob"] for x, y in zip(dag, dag[1:]))
    got = {_formula(m, d["atoms"]) for d in dag}
    mol = Chem.MolFromSmiles(m.smiles)
    expected = set()
    for a in range(m.natoms):  # every component of ≥ 2 atoms left by removing one atom
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(a)
        keep = [i for i in range(m.natoms) if i != a]
        expected |= {_formula(m, tuple(keep[i] for i in c)) for c in Chem.GetMolFrags(rw) if len(c) >= 2}
    assert expected <= got
    named = {"benzoyl": "C7H5O", "phenyl": "C6H5", "ethyl": "C2H5", "ethoxy": "C2H5O", "benzoate": "C7H5O2"}
    for formula in named.values():
        assert tuple(iceberg.formula_vector(formula).astype(int)) in got
    masses = {round(d["mass"], 3) for d in dag}
    assert {105.034, 77.039, 29.039} <= masses  # neutral C7H5O, C6H5, C2H5


def test_one_atom_molecule_has_only_the_root(model: iceberg.Iceberg) -> None:
    assert [d["atoms"] for d in model.dag("C", "[M+H]+", 20.0)] == [(0,)]
    (mz, it), = model.predict(["C"], "[M+H]+", [20.0])[0]
    ch4 = iceberg.MONO_MASSES[0] + 4 * iceberg.MONO_MASSES[iceberg._H]
    assert len(mz) == 2 * iceberg.N_SHIFTS and (it > 0).sum() == 2  # the root, shift 0, both m/z groups
    assert np.allclose(np.sort(mz[it > 0]), [ch4 - iceberg.ELECTRON_MASS, ch4 + iceberg.ION_MASS["[M+H]+"]], atol=1e-4)
