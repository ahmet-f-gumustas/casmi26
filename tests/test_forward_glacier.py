from itertools import combinations

import numpy as np
import pytest
import torch
from rdkit import Chem

from casmi.config import EXTERNAL_DIR
from casmi.forward import glacier

WEIGHTS = EXTERNAL_DIR / "ms_pred" / "derived_state_dicts" / "glacier_msg_sd.pt"
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason="GLACIER weights not present")
ETHYL_BENZOATE = "CCOC(=O)c1ccccc1"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


@pytest.fixture(scope="module")
def model() -> glacier.Glacier:
    return glacier.Glacier.load(WEIGHTS, device="cpu")


@needs_weights
def test_state_dict_loads_strictly_and_computed_tables_match() -> None:
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    net = glacier.GlacierNet()
    assert set(net.state_dict()) == set(ckpt["state_dict"])
    for name in glacier.COMPUTED_TABLES:
        assert torch.equal(net.state_dict()[name], ckpt["state_dict"][name]), name
    net.load_state_dict(ckpt["state_dict"], strict=True)


def test_canonical_smiles_follows_upstream_normalisation() -> None:
    assert glacier.canonical_smiles("C[C@H](N)C(=O)O") == "CC(N)C(=O)O"  # stereo removed
    assert glacier.canonical_smiles("OC(C)=C") == "CC(C)=O"  # keto/enol transform
    assert glacier.canonical_smiles("Oc1ccccn1") == "Oc1ccccn1"  # not RDKit's default pyridone
    assert glacier.canonical_smiles("C[Hg]C") is None and glacier.canonical_smiles("nonsense") is None


@needs_weights
def test_shapes_order_and_dtypes(model: glacier.Glacier) -> None:
    out = model.predict([ETHYL_BENZOATE, CAFFEINE], "[M+H]+", [20.0, 40.0, 60.0], top=50)
    assert len(out) == 2 and all(len(r) == 3 for r in out)
    for spectra in out:
        for mz, inten in spectra:
            assert mz.dtype == np.float64 and inten.dtype == np.float32 and len(mz) == len(inten) == 50
            assert np.all(np.diff(inten) <= 0) and inten.max() <= 1 and inten.min() >= 0


@needs_weights
def test_bad_inputs_give_none(model: glacier.Glacier) -> None:
    bad = ["not a smiles", "", "C[Hg]C", "CCO.CC", "C[N+](C)(C)C", "C" * 110]
    out = model.predict([ETHYL_BENZOATE, *bad], "[M+H]+", [30.0])
    assert out[0] is not None and out[1:] == [None] * len(bad)
    assert model.predict([ETHYL_BENZOATE], "[M-H]-", [30.0]) == [None]


@needs_weights
def test_deterministic_and_energy_dependent(model: glacier.Glacier) -> None:
    a = model.predict([CAFFEINE], "[M+H]+", [10.0, 60.0])[0]
    b = model.predict([CAFFEINE], "[M+H]+", [10.0, 60.0])[0]
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b, strict=True))
    (mz_lo, it_lo), (mz_hi, it_hi) = a
    assert not (np.array_equal(mz_lo, mz_hi) and np.allclose(it_lo, it_hi, atol=1e-3))
    mh = 194.080376 + 1.007276  # caffeine [M+H]+

    def intact(mz: np.ndarray, it: np.ndarray) -> float:
        hit = np.abs(mz - mh) < 1e-3
        return float(it[hit].max()) if hit.any() else 0.0

    assert intact(mz_lo, it_lo) > intact(mz_hi, it_hi)  # the intact ion fades with energy


def _subgraph_masses(smiles: str) -> np.ndarray:
    """Masses (heavy atoms + their Hs) of every connected atom subset, computed independently of the port."""
    mol = Chem.MolFromSmiles(smiles)
    pt = Chem.GetPeriodicTable()
    atom = [pt.GetMostCommonIsotopeMass(a.GetSymbol()) + a.GetTotalNumHs() * pt.GetMostCommonIsotopeMass("H")
            for a in mol.GetAtoms()]
    adj = Chem.GetAdjacencyMatrix(mol)
    masses = []
    for size in range(1, mol.GetNumAtoms() + 1):
        for sub in combinations(range(mol.GetNumAtoms()), size):
            seen, stack = {sub[0]}, [sub[0]]
            while stack:
                u = stack.pop()
                for v in sub:
                    if adj[u, v] and v not in seen:
                        seen.add(v)
                        stack.append(v)
            if len(seen) == size:
                masses.append(sum(atom[i] for i in sub))
    return np.unique(np.round(masses, 6))


@needs_weights
@pytest.mark.parametrize("adduct", glacier.SUPPORTED_ADDUCTS)
def test_fragment_mz_are_subgraph_masses_with_h_shifts(model: glacier.Glacier, adduct: str) -> None:
    pt = Chem.GetPeriodicTable()
    h, e = pt.GetMostCommonIsotopeMass("H"), 0.00054858
    ion = {"[M+H]+": h - e, "[M+Na]+": pt.GetMostCommonIsotopeMass("Na") - e}[adduct]
    subs = _subgraph_masses(ETHYL_BENZOATE)
    allowed = np.sort((subs[:, None, None] + np.arange(-6, 7)[None, :, None] * h
                       + np.array([ion, -e])[None, None, :]).ravel())
    precursor = subs.max() + ion
    for mz, inten in model.predict([ETHYL_BENZOATE], adduct, [20.0, 50.0], top=200)[0]:
        mz = mz[inten > 0]  # zero rows are masked H shifts
        assert len(mz) > 20 and np.all(mz <= precursor + 1e-6)
        nearest = np.abs(mz[:, None] - allowed[np.clip(np.searchsorted(allowed, mz), 1, len(allowed) - 1)
                                               [:, None] + np.array([-1, 0])]).min(1)
        assert nearest.max() < 1e-6


def _reference_fragments(mask: torch.Tensor, adj: torch.Tensor, n_atoms: torch.Tensor) -> list[torch.Tensor]:
    """``breakpoints_to_patterns`` as upstream writes it: Floyd–Warshall closure, ``torch.unique`` over rows."""
    b, m, n = mask.shape
    a = (adj.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, n, n) > 0)
    flat = mask.reshape(b * m, n)
    a = a & ~flat.unsqueeze(-1) & ~flat.unsqueeze(-2)
    for k in range(n):
        a = a | (a[:, :, k:k + 1] & a[:, k:k + 1, :])
    valid = torch.arange(n)[None, :] < n_atoms[:, None]
    a = a & (valid.unsqueeze(-1) & valid.unsqueeze(-2)).repeat_interleave(m, 0)
    rows = torch.cat((torch.arange(b).repeat_interleave(m)[:, None, None].expand(-1, n, 1), a.long()), -1)
    uniq = torch.unique(rows.reshape(-1, n + 1), dim=0)
    uniq = uniq[uniq[:, 1:].sum(-1) > 0]
    return [uniq[uniq[:, 0] == i, 1:].bool() if (uniq[:, 0] == i).any() else valid[i:i + 1] for i in range(b)]


def test_fragments_match_upstream_closure_and_order() -> None:
    gen = torch.Generator().manual_seed(0)
    for _ in range(30):
        b, m, n = 3, 10, int(torch.randint(3, 13, (1,), generator=gen))
        n_atoms = torch.randint(1, n + 1, (b,), generator=gen)
        adj = torch.zeros(b, n, n)
        for i in range(b):
            for v in range(1, int(n_atoms[i])):
                u = int(torch.randint(0, v, (1,), generator=gen))
                adj[i, u, v] = adj[i, v, u] = 1.5 if v % 3 else 1.0
        mask = torch.rand(b, m, n, generator=gen) < 0.25
        frags, counts = glacier._fragments(mask, adj, n_atoms)
        for i, ref in enumerate(_reference_fragments(mask, adj, n_atoms)):
            assert int(counts[i]) == len(ref) and torch.equal(frags[i, :len(ref)], ref)


def test_linsat_keeps_the_order_of_the_logits() -> None:
    x = torch.randn(64, 20, generator=torch.Generator().manual_seed(1)) * 20
    for k in (1, 2, 3):
        y = glacier._linsat(x, k)
        assert torch.equal(torch.argsort(y, dim=1), torch.argsort(x, dim=1))
