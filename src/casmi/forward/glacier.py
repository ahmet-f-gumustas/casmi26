"""GLACIER forward model (structure → predicted MS/MS spectrum) as plain-PyTorch inference.

Ported from coleygroup/ms-pred @ 708148c2, MIT License, Copyright (c) 2023 Samuel Goldman:
``glacier/joint_model.py`` (``JointModel``: molecular embedding, slot breakpoint decoder, fragments, intensity
decoder, ``predict_inten_frag_batch_sparse``), ``glacier/dataset.py`` (``TreeProcessor.featurize_tree`` and
``create_graphormer_input``), ``glacier/predict_smis_joint.py`` (``prepare_entry``), ``common/chem_utils.py``
(element, adduct and instrument tables; ``rm_stereo`` and the InChI round trip with its tautomer canonicaliser),
``magma/fragmentation.py`` (atom H counts and masses) and ``nn_utils`` (``SlotDecoder``, ``random_walk_pe``, the
abs-sines count embedder). The Graphormer encoder follows ``graphormer/*.py``, MIT License, Copyright (c)
Microsoft Corporation, parts Copyright (c) Facebook, Inc. and its affiliates.

The maths is upstream's. What changes is the machinery:

- no DGL, torch_scatter, Lightning or LinSATNet. ``node_ranking`` runs LinSAT with one equality constraint
  (Σ = k) for one iteration; ``_linsat`` writes out that kernel's own float ops;
- LinSAT and the breakpoint top-k run over each molecule's own atoms, not the batch's padded width
  (``_breakpoints``); the two agree whenever a batch's molecules have the same atom count;
- the multi-hop edge features are gathered from a per-molecule bond table instead of a dense one-hot tensor.
  The attention bias does not depend on the collision energy, so it is built once per molecule;
- fragments come from a reachability closure by repeated squaring over the distinct breakpoint sets, and are
  de-duplicated as packed bit rows. This gives the same fragments in the same (lexicographic) order as the
  Floyd–Warshall loop plus ``torch.unique``;
- fragment m/z are computed in float64 (``predict_smis_joint`` uses float32).

Weights: ``glacier_msg_sd.pt``, the MassSpecGym checkpoint saved again as ``{state_dict, hyper_parameters}``.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from torch import Tensor, nn

RDLogger.DisableLog("rdApp.*")

SUPPORTED_ADDUCTS = ("[M+H]+", "[M+Na]+")  # the only adducts in MassSpecGym training data
INSTRUMENTS = {"Orbitrap": 0, "QTOF": 1, "IT-FT": 2, "Unknown": 3}  # other strings → Orbitrap, as upstream
MAX_HEAVY_ATOMS = 100
MAX_ATOM_BONDS = 6

ELEMENTS = ("C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B", "Se", "Fe", "Co", "As", "Na", "K")
_GROUP = {"C": 4, "N": 3, "P": 3, "O": 5, "S": 5, "Si": 4, "I": 6, "H": 0, "Cl": 6, "F": 6, "Br": 6, "B": 2,
          "Se": 5, "Fe": 7, "Co": 7, "As": 3, "Na": 1, "K": 1}
_PT = Chem.GetPeriodicTable()
_ATOM_NUM = [_PT.GetAtomicNumber(e) for e in ELEMENTS]
_EL = {e: i for i, e in enumerate(ELEMENTS)}
_MASS = {e: _PT.GetMostCommonIsotopeMass(e) for e in ELEMENTS}
_H = _MASS["H"]
ELECTRON_MASS = 0.00054858
ADDUCT_MASS = {"[M+H]+": _H - ELECTRON_MASS, "[M+Na]+": _MASS["Na"] - ELECTRON_MASS}

# chem_utils.ion2onehot_pos: the 14 adduct classes of the MSG checkpoints
_ADDUCT_CLASS = {"[M+H]+": 0, "[M+Na]+": 1, "[M+K]+": 2, "[M-H2O+H]+": 3, "[M+H-H2O]+": 3, "[M+H3N+H]+": 4,
                 "[M+NH4]+": 4, "[M]+": 5, "[M-H4O2+H]+": 6, "[M+H-2H2O]+": 6, "[M+H-H3N]+": 7, "[M-H3N+H]+": 7,
                 "[M-NH3+H]+": 7, "[M+H-NH3]+": 7, "[M+H-CH2O2]+": 8, "[M+CHO2]-": 9, "[M-H]-": 10, "[M+Cl]-": 11,
                 "[M-H2O-H]-": 12, "[M-H-H2O]-": 12, "[M-H-CO2]-": 13}

HIDDEN = 256
HEADS = 8
SLOTS = 200
HOPS = 10
PE_STEPS = 14
SHIFTS = 13  # H shifts −6…+6
LINSAT_TAU = 15.0
NODE_FEATS = 88
HPARAMS = {"hidden_size": HIDDEN, "graphormer_layers": 5, "node_feats": NODE_FEATS, "edge_feats": 28,
           "multi_hop_max_dist": HOPS, "num_edge_dis": HOPS, "max_breakpoints": SLOTS, "linsat_tau": LINSAT_TAU,
           "max_broken_bonds": 6, "frag_decoder_layers": 4, "frag_encoder_layers": 0, "inten_decoder_layers": 4,
           "inten_encoder_layers": 3, "pe_embed_k": PE_STEPS, "embed_adduct": True, "embed_collision": True,
           "embed_instrument": True, "embed_elem_group": True, "encode_forms": True, "enable_decoder_norm": False}
COMPUTED_TABLES = ("adduct_embedder", "collision_embedder_denominators", "collision_embed_merged",
                   "instrument_embedder", "embedder.int_to_feat_matrix")

Spectrum = tuple[NDArray[np.float64], NDArray[np.float32]]

# --- canonicalisation and featurisation (CPU, once per molecule) ---------------------------------------------


@lru_cache(maxsize=1)
def _tautomers() -> rdMolStandardize.TautomerEnumerator:
    """``chem_utils._get_tautomer_enumerator``: RDKit's enumerator restricted to upstream's two transforms."""
    params = rdMolStandardize.CleanupParameters()
    rdMolStandardize.UpdateParamsFromJSON(params, json.dumps({"tautomerTransformData": [
        {"name": "1,3 hetero atom H shift", "smarts": "[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]"},
        {"name": "1,3 (thio)keto/enol r", "smarts": "[O,S,Se,Te;X2!H0]-[C]=[C]"}]}))
    return rdMolStandardize.TautomerEnumerator(params)


def canonical_smiles(smiles: str) -> str | None:
    """The SMILES upstream feeds to GLACIER, or None where its entry filter drops the molecule.

    ``rm_stereo``, then ``smi_inchi_round_smi`` (InChI round trip plus tautomer canonicalisation, as ms-pred builds
    its retrieval candidates), then ``prepare_entry`` (valid elements, RemoveHs, ≤ 100 heavy atoms, canonical
    SMILES). Its atom order is the order of every per-atom tensor.
    """
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        inchi = Chem.MolToInchi(mol) if mol is not None else ""
        mol = Chem.MolFromInchi(inchi) if inchi else None
        if mol is None:
            return None
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(_tautomers().Canonicalize(mol)))
        if mol is None or any(a.GetAtomicNum() not in _ATOM_NUM for a in mol.GetAtoms()):
            return None
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol) if 0 < mol.GetNumAtoms() <= MAX_HEAVY_ATOMS else None
    except Exception:  # noqa: BLE001 - prepare_entry drops any row RDKit raises on
        return None


@dataclass(frozen=True)
class MolFeatures:
    """One molecule's CPU features (``featurize_tree`` + ``create_graphormer_input``), shared by all energies."""

    smiles: str
    x: NDArray[np.float32]  # [n, 88] atom features, then the 14-step random-walk PE
    degree: NDArray[np.int64]  # [n] heavy-atom degree
    spatial: NDArray[np.int64]  # [n, n] BFS distance, clipped to 1023
    hops: NDArray[np.int16]  # [n, n, HOPS] bond (1-based, 0 = none) on hop h of the BFS path from row to column
    bonds: NDArray[np.float32]  # [n_bonds, 28] bond features
    adj: NDArray[np.float32]  # [n, n] adjacency with bond orders (aromatic 1.5)
    hs: NDArray[np.int64]  # [n] H count per atom (Kekulé form)
    mass: NDArray[np.float64]  # [n] atom mass with its Hs
    form: NDArray[np.float32]  # [n, 18] atom formula vectors
    root: NDArray[np.float32]  # [18] molecular formula

    @property
    def n(self) -> int:
        return len(self.degree)


def _onehot(value: object, choices: Sequence[object], other: bool = True) -> list[float]:
    out = [float(value == c) for c in choices]
    return out + [float(value not in choices)] if other else out


_BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
_HYBRID = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
           Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2)
_CHIRAL = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
           Chem.ChiralType.CHI_OTHER)
_STEREO = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)
_RINGS = range(3, 9)
_FORMULA = re.compile(r"([A-Z][a-z]*)([0-9]*)")


def _atom_features(mol: Chem.Mol, atom_rings: NDArray[np.int64]) -> NDArray[np.float32]:
    feats = []
    for atom in mol.GetAtoms():
        degree = atom.GetTotalDegree() - atom.GetTotalNumHs()
        charge = atom.GetFormalCharge()
        n_h = atom.GetTotalNumHs()
        ring = atom.IsInRing()
        f = (_onehot(atom.GetAtomicNum(), _ATOM_NUM) + _onehot(degree, range(6), other=False) + [float(degree > 5)]
             + _onehot(charge, range(-2, 3), other=False) + [float(abs(charge) > 2)]
             + _onehot(atom.GetHybridization(), _HYBRID) + [float(atom.GetIsAromatic())]
             + _onehot(n_h, range(5), other=False) + [float(n_h > 4), float(not ring), float(ring)])
        for size in _RINGS:
            in_size = atom.IsInRingSize(size)
            f += [float(in_size), float(not in_size)]
        fused = atom_rings[atom.GetIdx()] >= 2
        f += [float(not fused), float(fused)] + _onehot(atom.GetChiralTag(), _CHIRAL, other=False)
        f += np.eye(8)[_GROUP[atom.GetSymbol()]].tolist()
        try:
            q = float(atom.GetProp("_GasteigerCharge"))
            f.append(q if math.isfinite(q) else 0.0)
        except KeyError:  # no charge computed
            f.append(0.0)
        feats.append(f)
    return np.asarray(feats, np.float32)


def _bond_features(mol: Chem.Mol, bond_rings: dict[tuple[int, int], int]) -> NDArray[np.float32]:
    feats = []
    for bond in mol.GetBonds():
        conj, ring = bond.GetIsConjugated(), bond.IsInRing()
        f = _onehot(bond.GetBondType(), _BOND_TYPES) + [float(not conj), float(conj), float(not ring), float(ring)]
        for size in _RINGS:
            in_size = bond.IsInRingSize(size)
            f += [float(in_size), float(not in_size)]
        fused = bond_rings[tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))] >= 2
        f += [float(not fused), float(fused)] + _onehot(bond.GetStereo(), _STEREO)
        feats.append(f)
    return np.asarray(feats, np.float32).reshape(-1, 28)


def _random_walk_pe(adj_bo: NDArray[np.float64]) -> Tensor:
    """``nn_utils.random_walk_pe`` (dense path) on bond-order edge weights: diag of P¹…P¹⁴."""
    n = len(adj_bo)
    src, dst = np.nonzero(adj_bo)
    value = torch.tensor(adj_bo[src, dst], dtype=torch.float32)
    rows = torch.from_numpy(src)
    norm = torch.zeros(n).index_add_(0, rows, value)[rows] + 1e-30
    p = torch.zeros(n, n)
    p[rows, torch.from_numpy(dst)] = value / norm
    out, diag = p, [p.diagonal()]
    for _ in range(PE_STEPS - 1):
        out = out @ p
        diag.append(out.diagonal())
    return torch.stack(diag, dim=-1)


def _paths(adj: NDArray[np.bool_], bond_id: NDArray[np.int16]) -> tuple[NDArray[np.int64], NDArray[np.int16]]:
    """BFS distances and the bond on each of the first ``HOPS`` hops of upstream's BFS path.

    Upstream's parent of ``v`` seen from ``s`` is the lowest-index neighbour of ``v`` one step closer to ``s``
    (``argmax`` over the frontier); the path from ``s`` to ``t`` follows parents back from ``t``.
    """
    n = len(adj)
    unreachable = 32767
    dist = np.full((n, n), unreachable, np.int64)
    np.fill_diagonal(dist, 0)
    seen = np.eye(n, dtype=bool)
    frontier = seen.copy()
    a = adj.astype(np.float32)
    for depth in range(1, n):
        nxt = ((frontier.astype(np.float32) @ a) > 0) & ~seen
        if not nxt.any():
            break
        dist[nxt] = depth
        seen |= nxt
        frontier = nxt
    cand = adj[None, :, :] & (dist[:, :, None] == dist[:, None, :] - 1)  # [s, u, v]
    parent = np.where(cand.any(axis=1), cand.argmax(axis=1), np.arange(n)[:, None])  # s itself where none
    reach = dist < unreachable
    far = int(dist[reach].max())
    up = np.empty((far + 1, n, n), np.int64)  # up[m][s, t]: m parent steps from t towards s
    up[0] = np.arange(n)[None, :]
    for m in range(far):
        up[m + 1] = np.take_along_axis(parent, up[m], axis=1)
    s_idx, t_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    hops = np.zeros((n, n, HOPS), np.int16)
    for h in range(HOPS):
        valid = reach & (dist > h)
        m = np.where(valid, dist - h - 1, 0)
        lo, hi = up[m, s_idx, t_idx], up[np.minimum(m + 1, far), s_idx, t_idx]
        hops[:, :, h] = np.where(valid, bond_id[hi, lo], 0)
    return np.clip(dist, 0, 1023), hops


def featurise(smiles: str) -> MolFeatures | None:
    """Features of a canonical SMILES (``canonical_smiles``), or None for a molecule the port does not handle.

    None for a charged or multi-component molecule, an atom with more than ``MAX_ATOM_BONDS`` bonds
    (``FragmentEngine`` would overflow), explicit H atoms (its atom order would not match the heavy-atom graph) or a
    Kekulisation failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or "." in smiles or Chem.GetFormalCharge(mol) != 0:
        return None
    heavy = Chem.RemoveHs(mol)
    n = heavy.GetNumAtoms()
    if n == 0 or n != mol.GetNumAtoms() or any(a.GetDegree() > MAX_ATOM_BONDS for a in mol.GetAtoms()):
        return None
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except ValueError:  # KekulizeException; FragmentEngine would raise
        return None
    atoms = list(kekule.GetAtoms())
    symbols = [a.GetSymbol() for a in atoms]
    hs = np.array([a.GetNumImplicitHs() + a.GetNumExplicitHs() for a in atoms], np.int64)
    iso = [a.GetIsotope() for a in atoms]
    weights = np.array([_PT.GetMassForIsotope(s, i) if i else _MASS[s] for s, i in zip(symbols, iso, strict=True)])
    form = np.zeros((n, len(ELEMENTS)), np.float32)
    form[np.arange(n), [_EL[s] for s in symbols]] = 1
    form[:, _EL["H"]] += hs
    root = np.zeros(len(ELEMENTS), np.float32)
    for symbol, count in _FORMULA.findall(re.findall(r"^([^\+,^\-]*)", CalcMolFormula(mol))[0]):
        if symbol not in _EL:
            return None
        root[_EL[symbol]] += int(count) if count else 1

    with contextlib.suppress(RuntimeError, ValueError):  # upstream ignores a failed charge model
        rdPartialCharges.ComputeGasteigerCharges(heavy)
    bond_rings = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): 0 for b in heavy.GetBonds()}
    atom_rings = np.zeros(n, np.int64)
    for ring in Chem.GetSymmSSSR(heavy):
        ring = list(ring)
        atom_rings[ring] += 1
        for k in range(len(ring)):
            key = tuple(sorted((ring[k], ring[(k + 1) % len(ring)])))
            if key in bond_rings:
                bond_rings[key] += 1
    adj_bo = Chem.GetAdjacencyMatrix(heavy, useBO=True)
    x = torch.cat((torch.from_numpy(_atom_features(heavy, atom_rings)), _random_walk_pe(adj_bo)), -1).numpy()
    bond_id = np.zeros((n, n), np.int16)
    for b in heavy.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_id[i, j] = bond_id[j, i] = b.GetIdx() + 1
    spatial, hops = _paths(Chem.GetAdjacencyMatrix(heavy).astype(bool), bond_id)
    degree = np.array([a.GetTotalDegree() - a.GetTotalNumHs() for a in heavy.GetAtoms()], np.int64)
    return MolFeatures(smiles=smiles, x=x, degree=degree, spatial=spatial, hops=hops,
                       bonds=_bond_features(heavy, bond_rings), adj=adj_bo.astype(np.float32), hs=hs,
                       mass=hs * _H + weights, form=form, root=root)


# --- network ---------------------------------------------------------------------------------------------------


def _adduct_modes(ion: str) -> set[int]:
    """``chem_utils.ion_pos2extra_multihot`` for one adduct string (its parser, verbatim)."""
    mode = ion[-1]
    body = ion.strip(mode).strip("[M").strip("]")
    parts = []
    for i, part in enumerate(body.split("+")):
        if part:
            part = "+" + part if i else part
            parts += [p if j == 0 else "-" + p for j, p in enumerate(part.split("-")) if p]
    bits = {0 if mode == "+" else 1}
    for bit, names in ((2, ("+Na", "+K")), (3, ("+H",)), (4, ("-H",)), (5, ("+Cl",)), (6, ("-H2O", "-H4O2", "-2H2O")),
                       (7, ("-CO2",)), (8, ("+NH3",))):
        if any(name in parts for name in names):
            bits.add(bit)
    return bits


def _adduct_table() -> Tensor:
    """One-hot class plus the multi-hot mode bits (pos, neg, alkali, +H, −H, +Cl, −H2O, −CO2): [14, 22]."""
    classes = len(set(_ADDUCT_CLASS.values()))
    modes: dict[int, set[int]] = {c: set() for c in range(classes)}
    for ion, c in _ADDUCT_CLASS.items():
        modes[c] |= _adduct_modes(ion)
    n_modes = len(set().union(*modes.values()))
    multihot = torch.zeros(classes, n_modes)
    for c, bits in modes.items():
        multihot[c, sorted(bits)] = 1
    return torch.cat((torch.eye(classes), multihot), dim=-1)


def _count_table() -> Tensor:
    """``FourierFeaturizerAbsoluteSines.int_to_feat_matrix``: |sin(2π c / 2^j)|, j = 2…9, for c = 0…254."""
    freqs = (0.5 ** torch.arange(10, dtype=torch.float32))[2:]
    return torch.abs(torch.sin(torch.arange(255, dtype=torch.float32)[:, None] * (2 * np.pi * freqs)[None, :]))


class _CountEmbedder(nn.Module):
    """Abs-sines embedding of element counts; counts ≥ 255 share the learned extra row."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("int_to_feat_matrix", _count_table())
        self._extra_embeddings = nn.Parameter(torch.zeros(1, 8), requires_grad=False)

    def forward(self, counts: Tensor) -> Tensor:
        table = torch.cat((self.int_to_feat_matrix, self._extra_embeddings))
        return table[counts.long().clamp(max=len(self.int_to_feat_matrix))].reshape(*counts.shape[:-1], -1)


class _GraphAttention(nn.Module):
    """``graphormer.MultiheadAttention``: fused qkv projection, additive bias, T × B × C layout."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        t, b, c = x.shape
        q, k, v = (y.view(t, b, self.heads, -1).permute(1, 2, 0, 3) for y in self.qkv_proj(x).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.view(b, self.heads, t, -1))
        return self.out_proj(a.permute(2, 0, 1, 3).reshape(t, b, c))


class _GraphormerLayer(nn.Module):
    """``GraphormerGraphEncoderLayer``: post-LN, GELU feed-forward."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.self_attn = _GraphAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Identity(), nn.Linear(4 * dim, dim))
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        x = self.self_attn_layer_norm(x + self.self_attn(x, bias))
        return self.final_layer_norm(x + self.mlp(x))


class _NodeFeature(nn.Module):
    def __init__(self, n_in: int, dim: int) -> None:
        super().__init__()
        self.atom_encoder = nn.Linear(n_in, dim)
        self.degree_encoder = nn.Embedding(8, dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, dim)

    def forward(self, x: Tensor, degree: Tensor) -> Tensor:
        nodes = self.atom_encoder(x) + self.degree_encoder(degree)
        return torch.cat((self.graph_token.weight.unsqueeze(0).repeat(len(x), 1, 1), nodes), dim=1)


class _AttnBias(nn.Module):
    """``GraphAttnBias`` with multi-hop edges: spatial, virtual-token and path-averaged edge biases per head."""

    def __init__(self, heads: int, n_edge: int) -> None:
        super().__init__()
        self.heads = heads
        self.edge_encoder = nn.Linear(n_edge, heads)
        self.edge_dis_encoder = nn.Embedding(HOPS * heads * heads, 1)
        self.spatial_pos_encoder = nn.Embedding(1025, heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, heads)

    def forward(self, base: Tensor, spatial: Tensor, hops: Tensor, bonds: Tensor) -> Tensor:
        """``base`` [B, N+1, N+1]: 0 on real tokens, −99999 on padding; ``hops`` [B, N, N, HOPS] index rows of
        ``bonds`` (row 0 is the all-zero feature of a missing hop, which still contributes the encoder bias)."""
        b, n = spatial.shape[:2]
        h = self.heads
        bias = base.unsqueeze(1).repeat(1, h, 1, 1)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + self.spatial_pos_encoder(spatial).permute(0, 3, 1, 2)
        t = self.graph_token_virtual_distance.weight.view(1, h, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + t
        bias[:, :, 0, :] = bias[:, :, 0, :] + t
        steps = spatial.clone()
        steps[steps == 0] = 1
        steps = torch.where(steps > 1, steps - 1, steps).clamp(0, HOPS)
        edge = self.edge_encoder(bonds)[hops].permute(3, 0, 1, 2, 4).reshape(HOPS, -1, h)
        edge = torch.bmm(edge, self.edge_dis_encoder.weight.reshape(-1, h, h)[:HOPS])
        edge = edge.reshape(HOPS, b, n, n, h).permute(1, 2, 3, 0, 4)
        edge = (edge.sum(-2) / steps.float().unsqueeze(-1)).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + edge
        return bias + base.unsqueeze(1)


class _Graphormer(nn.Module):
    def __init__(self, n_in: int, dim: int, heads: int, layers: int, n_edge: int) -> None:
        super().__init__()
        self.graph_node_feature = _NodeFeature(n_in, dim)
        self.graph_attn_bias = _AttnBias(heads, n_edge)
        self.layers = nn.ModuleList(_GraphormerLayer(dim, heads) for _ in range(layers))

    def forward(self, x: Tensor, degree: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
        """Node embeddings [B, N, C] and the graph token [B, C]."""
        h = self.graph_node_feature(x, degree).transpose(0, 1)
        for layer in self.layers:
            h = layer(h, bias)
        return h[1:].transpose(0, 1), h[0]


class _SlotDecoder(nn.Module):
    """``nn_utils.SlotDecoder``: learned slots cross-attending to [root token, atoms]; every layer's output."""

    def __init__(self, dim: int, slots: int, heads: int, layers: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.zeros(slots, dim))
        self.decoder_layers = nn.ModuleList(
            nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True) for _ in range(layers))

    def forward(self, root: Tensor, nodes: Tensor, padding: Tensor) -> Tensor:
        slots = self.slots.unsqueeze(0).expand(len(nodes), *self.slots.shape)
        memory = torch.cat((root, nodes), dim=1)
        outs = []
        for layer in self.decoder_layers:
            slots = layer(tgt=slots, memory=memory, memory_key_padding_mask=padding)
            outs.append(slots)
        return torch.stack(outs)


def _linsat(x: Tensor, k: int) -> Tensor:
    """LinSATNet ``linsat_layer(x, E=1ᵀ, f=k, max_iter=1, tau=LINSAT_TAU)`` (grouped dense v2 kernel), op for op.

    A per-row monotone map of ``x``, so top-k on it is top-k on ``x`` except where float32 saturates.
    """
    n = x.shape[1]
    b = torch.tensor([[float(k), float(n - k)]], device=x.device)
    if torch.any(b == 0):
        b = b + 1e-7
    log_a = torch.log(torch.ones(1, n, device=x.device) / torch.full((1, 1), float(n), device=x.device))
    log_b = torch.log(b / b.sum(dim=-1, keepdim=True))
    log_x = torch.stack((x / LINSAT_TAU, torch.zeros_like(x)), dim=1)
    log_x = log_x - torch.logsumexp(log_x, 1, keepdim=True)
    total = torch.logsumexp(log_x[:, None, :, :] + log_a[None, :, None, :], 3, keepdim=True)
    if torch.all(torch.abs(total - log_b[None, :, :, None]) < 1e-3):
        return torch.exp(log_x[:, 0, :])
    t = (log_x[:, None, :, :] - total + log_b[None, :, :, None])[:, 0]
    return torch.exp((t - torch.logsumexp(t, 1, keepdim=True))[:, 0, :])


def _breakpoints(logits: Tensor, card: Tensor, n_atoms: Tensor) -> Tensor:
    """``breakpoint_inference``: per slot the top-k atoms, k = argmax of its cardinality logits. [B, M, N] bool.

    LinSAT and top-k run over each molecule's own atoms. Upstream runs them over the batch's padded width, with
    the padding at −1e9. That shifts every LinSAT output of a row by the same constant, so the order is the same,
    but it changes which near-1 values round to the same float32 and so how ties break. The two agree when all
    molecules in a batch have the same atom count, as in ``predict_mol`` and in the size-sorted batches of
    ``predict_smis_joint`` over isomer candidates. Symmetric atoms have logits equal up to float rounding, so
    which of them a slot takes still depends on the batch shape, here as upstream.
    """
    b, m, n = logits.shape
    k = card.argmax(dim=-1)
    out = torch.zeros(b, m, n, dtype=torch.bool, device=logits.device)
    for size in torch.unique(n_atoms).tolist():
        rows = torch.nonzero(n_atoms == size).squeeze(1)
        flat = logits[rows, :, :size].reshape(-1, size)
        ranked = torch.stack([torch.zeros_like(flat)] + [_linsat(flat, j) for j in (1, 2, 3)], dim=-1)
        flat_k = k[rows].reshape(-1)
        preds = torch.gather(ranked, -1, flat_k[:, None, None].expand(-1, size, 1)).squeeze(-1)
        top = min(3, size)
        idx = torch.topk(preds, k=top, dim=-1).indices
        keep = torch.arange(top, device=preds.device)[None, :] < flat_k[:, None]
        sel = torch.zeros_like(preds, dtype=torch.bool)
        sel[torch.arange(len(preds), device=preds.device)[:, None].expand(-1, top)[keep], idx[keep]] = True
        out[rows, :, :size] = sel.view(len(rows), m, size)
    return out


_BITS = 63


def _pack(bits: Tensor) -> Tensor:
    """[K, N] bool → [K, W] int64, first atom most significant, so word order is lexicographic bit order."""
    k, n = bits.shape
    w = -(-n // _BITS)
    place = 2 ** torch.arange(_BITS - 1, -1, -1, device=bits.device, dtype=torch.int64)
    return (F.pad(bits, (0, w * _BITS - n)).view(k, w, _BITS).long() * place).sum(-1)


def _unpack(words: Tensor, n: int) -> Tensor:
    shift = torch.arange(_BITS - 1, -1, -1, device=words.device, dtype=torch.int64)
    return ((words[:, :, None] >> shift) & 1).reshape(len(words), words.shape[1] * _BITS)[:, :n].bool()


def _fragments(breaks: Tensor, adj: Tensor, n_atoms: Tensor, chunk: int = 1 << 25) -> tuple[Tensor, Tensor]:
    """``breakpoints_to_patterns``: the distinct atom sets reachable after removing each slot's breakpoints.

    An atom left without a neighbour reaches nothing and yields no fragment, as in upstream's closure; a row with
    no fragment at all gets the whole molecule. Returns [B, F, N] bool (lexicographic order) and counts [B].
    """
    b, m, n = breaks.shape
    dev = breaks.device
    row = torch.arange(b, device=dev).repeat_interleave(m)
    sets = torch.unique(torch.cat((row[:, None], _pack(breaks.reshape(b * m, n))), 1), dim=0)
    row, removed = sets[:, 0], _unpack(sets[:, 1:], n)
    valid = torch.arange(n, device=dev)[None, :] < n_atoms[row][:, None]
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    steps = max(1, math.ceil(math.log2(max(n - 1, 2))))
    words, step = [], max(1, chunk // (n * n))
    for s in range(0, len(row), step):
        r, keep, ok = row[s:s + step], ~removed[s:s + step], valid[s:s + step]
        p = ((adj[r] > 0) & keep[:, :, None] & keep[:, None, :]).to(dtype)
        for _ in range(steps):
            p = ((p @ p + p) > 0).to(dtype)
        reach = (p > 0) & ok[:, :, None] & ok[:, None, :]
        words.append(torch.cat((r.repeat_interleave(n)[:, None], _pack(reach.reshape(-1, n))), 1))
    uniq = torch.unique(torch.cat(words), dim=0)
    uniq = uniq[(uniq[:, 1:] != 0).any(-1)]
    counts = torch.bincount(uniq[:, 0], minlength=b)
    frags = torch.zeros(b, max(1, int(counts.max())), n, dtype=torch.bool, device=dev)
    start = torch.cumsum(counts, 0) - counts
    frags[uniq[:, 0], torch.arange(len(uniq), device=dev) - start[uniq[:, 0]]] = _unpack(uniq[:, 1:], n)
    empty = counts == 0
    frags[empty, 0] = torch.arange(n, device=dev)[None, :] < n_atoms[empty][:, None]
    return frags, counts.clamp(min=1)


class GlacierNet(nn.Module):
    """``glacier.joint_model.JointModel`` at inference, under the checkpoint's state-dict names."""

    def __init__(self) -> None:
        super().__init__()
        dim = HIDDEN
        self.register_buffer("adduct_embedder", _adduct_table())
        self.register_buffer("collision_embedder_denominators", torch.pow(10000, 2 * torch.arange(32) / 64))
        self.register_buffer("collision_embed_merged", torch.zeros(64))
        self.register_buffer("instrument_embedder", torch.eye(len(INSTRUMENTS)))
        n_in = NODE_FEATS + self.adduct_embedder.shape[1] + 64 + len(INSTRUMENTS)
        self.root_module = _Graphormer(n_in, dim, HEADS, HPARAMS["graphormer_layers"], HPARAMS["edge_feats"])
        self.embedder = _CountEmbedder()
        form_dim = len(ELEMENTS) * 8
        self.formula_mapper = nn.Linear(form_dim + dim, dim)
        self.token_mapper = nn.Linear(dim + 2 * form_dim + 7, dim)
        self.fragment_decoder = _SlotDecoder(dim, SLOTS, HEADS, HPARAMS["frag_decoder_layers"])
        self.frag_card_mapper = nn.Linear(dim, 4)
        self.inten_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, HEADS, 4 * dim, dropout=0.0, batch_first=True), 4)
        self.inten_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, HEADS, 4 * dim, dropout=0.0, batch_first=True), 3)
        self.output_map = nn.Linear(dim, 2 * SHIFTS)

    def forward(self, mol: dict[str, Tensor], energies: Tensor, adduct: int, instrument: int
                ) -> tuple[Tensor, Tensor, Tensor]:
        """Intensities [R, F·26], fragment masses [R, F] (float64) and fragment counts [R] for R = B·E rows,
        molecule-major. Row f·26 + 13·g + s is fragment f, ion group g (0: + adduct, 1: − electron), H shift s − 6.
        """
        n_mol, n = mol["degree"].shape
        rows, n_e = n_mol * len(energies), len(energies)
        of = torch.arange(n_mol, device=energies.device).repeat_interleave(n_e)
        bias = self.root_module.graph_attn_bias(mol["base"], mol["spatial"], mol["hops"], mol["bonds"])
        ce = energies.repeat(n_mol)[:, None] / self.collision_embedder_denominators[None, :]
        ce = torch.cat((torch.sin(ce), torch.cos(ce)), dim=1)
        ce = torch.where(torch.isnan(ce), self.collision_embed_merged[None, :], ce)
        x = torch.cat((mol["x"][of], self.adduct_embedder[adduct].expand(rows, n, -1),
                       ce[:, None, :].expand(rows, n, -1), self.instrument_embedder[instrument].expand(rows, n, -1)),
                      dim=-1)
        nodes, graph = self.root_module(x, mol["degree"][of], bias.repeat_interleave(n_e, 0))
        root = graph.unsqueeze(1)
        n_atoms = mol["n_atoms"][of]
        root_form = mol["root"][of]

        padding = F.pad(torch.arange(n, device=x.device)[None, :] >= n_atoms[:, None], (1, 0), value=False)
        slots_root = self.formula_mapper(torch.cat((root, self.embedder(root_form)[:, None, :]), dim=-1))
        slots = self.fragment_decoder(slots_root, nodes, padding)
        card = self.frag_card_mapper(slots)[-1]
        logits = torch.einsum("nbij,bkj->nbik", slots, nodes)[-1]
        frags, counts = _fragments(_breakpoints(logits, card, n_atoms), mol["adj"][of], n_atoms)

        n_frag = frags.shape[1]
        real = torch.arange(n_frag, device=x.device)[None, :] < counts[:, None]
        fm = frags.float()
        adj = mol["adj"][of]
        frag_form = fm @ mol["form"][of]
        frag_hs = (fm * mol["hs"][of][:, None, :]).sum(-1)
        broken = (fm * adj.sum(-1)[:, None, :]).sum(-1) - ((fm @ adj) * fm).sum(-1)
        tokens = self.token_mapper(torch.cat((root.expand(rows, n_frag, HIDDEN), self.embedder(frag_form),
                                              self.embedder(root_form[:, None, :] - frag_form),
                                              F.one_hot(broken.clamp(max=6).long(), 7).float()), dim=-1))
        attend = frags | ~real[:, :, None]  # a padded fragment may see every atom, as upstream pads with True
        hidden = self.inten_decoder(tgt=tokens, memory=nodes, memory_mask=(~attend).repeat_interleave(HEADS, 0),
                                    tgt_key_padding_mask=~real)
        with warnings.catch_warnings():  # the fast path's "nested tensors are in prototype stage" notice
            warnings.filterwarnings("ignore", message=".*nested tensors.*")
            hidden = self.inten_encoder(hidden, src_key_padding_mask=~real)
        shift = torch.arange(SHIFTS, device=x.device)
        max_remove = frag_hs.clamp(max=6)
        max_add = (mol["total_hs"][of][:, None] - frag_hs).clamp(max=6)
        ok = (shift <= max_add[..., None] + 6) & (shift >= 6 - max_remove[..., None]) & real[..., None]
        ok = ok.unsqueeze(-2).expand(rows, n_frag, 2, SHIFTS).reshape(rows, n_frag, 2 * SHIFTS)
        inten = torch.sigmoid(self.output_map(hidden)).masked_fill(~ok, 0).reshape(rows, -1)
        frag_mass = (frags.double() @ mol["mass"][of][:, :, None]).squeeze(-1)
        return inten, frag_mass, counts


def collate(mols: Sequence[MolFeatures], device: torch.device) -> dict[str, Tensor]:
    """Pad a batch of molecules (``IntenDataset.collate_fn``, graphormer part) onto ``device``."""
    b, n = len(mols), max(m.n for m in mols)
    x = np.zeros((b, n, NODE_FEATS), np.float32)
    degree = np.zeros((b, n), np.int64)
    spatial = np.zeros((b, n, n), np.int64)
    hops = np.zeros((b, n, n, HOPS), np.int64)
    base = np.full((b, n + 1, n + 1), -99999, np.float32)
    adj = np.zeros((b, n, n), np.float32)
    hs = np.zeros((b, n), np.float32)
    mass = np.zeros((b, n), np.float64)
    form = np.zeros((b, n, len(ELEMENTS)), np.float32)
    bonds = [np.zeros((1, 28), np.float32)]
    offset = 0
    for i, m in enumerate(mols):
        k = m.n
        x[i, :k], degree[i, :k], spatial[i, :k, :k] = m.x, m.degree, m.spatial
        hops[i, :k, :k] = np.where(m.hops > 0, m.hops + offset, 0)
        base[i, :k + 1, :k + 1] = 0
        adj[i, :k, :k], hs[i, :k], mass[i, :k], form[i, :k] = m.adj, m.hs, m.mass, m.form
        bonds.append(m.bonds)
        offset += len(m.bonds)
    arrays = {"x": x, "degree": degree, "spatial": spatial, "hops": hops, "base": base, "adj": adj, "hs": hs,
              "mass": mass, "form": form, "bonds": np.concatenate(bonds), "root": np.stack([m.root for m in mols]),
              "total_hs": np.array([m.hs.sum() for m in mols], np.float32),
              "n_atoms": np.array([m.n for m in mols], np.int64)}
    return {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}


def check_tables(net: GlacierNet, state_dict: dict[str, Tensor]) -> None:
    """Raise unless the tables upstream computes at construction equal the stored ones, up to float rounding
    (libm differs across platforms: 6e-8 on aarch64); the stored values are the ones loaded."""
    own = net.state_dict()
    for name in COMPUTED_TABLES:
        if not torch.allclose(own[name], state_dict[name], rtol=0.0, atol=1e-5):
            diff = (own[name] - state_dict[name]).abs().max().item()
            raise ValueError(f"GLACIER table {name} differs from the checkpoint (max |Δ| = {diff:.3g})")


class Glacier:
    """GLACIER (MassSpecGym weights): sparse predicted spectra for SMILES at given collision energies (eV)."""

    def __init__(self, net: GlacierNet, device: torch.device) -> None:
        self.net = net
        self.device = device

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> Glacier:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        hp = ckpt["hyper_parameters"]
        bad = {k: hp.get(k) for k, v in HPARAMS.items() if hp.get(k) != v}
        if bad:
            raise ValueError(f"{path}: not the GLACIER configuration this port implements: {bad}")
        net = GlacierNet()
        check_tables(net, ckpt["state_dict"])
        net.load_state_dict(ckpt["state_dict"], strict=True)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return cls(net.to(dev).eval(), dev)

    def predict(self, smiles: Sequence[str], adduct: str, energies: Sequence[float], instrument: str = "QTOF",
                top: int = 100, batch: int = 64) -> list[list[Spectrum] | None]:
        """Per SMILES one spectrum per energy, or None where upstream cannot predict (see ``canonical_smiles``,
        ``featurise``) and for every SMILES when ``adduct`` is not in ``SUPPORTED_ADDUCTS``.

        A spectrum is the top-``top`` rows by intensity of ``predict_inten_frag_batch_sparse``: m/z (float64) and
        sigmoid intensity (float32, not normalised), most intense first. When a molecule has fewer than ``top``
        rows in total its masked H-shift rows (intensity 0) are included, as upstream writes them. ``batch``
        molecules go through the GPU together, each with all its energies. A non-finite energy raises ValueError
        (upstream skips such rows).
        """
        ces = [float(e) for e in energies]
        if not all(math.isfinite(e) for e in ces):
            raise ValueError(f"collision energies must be finite (eV), got {list(energies)}")
        if adduct not in SUPPORTED_ADDUCTS:
            return [None] * len(smiles)
        canon = {s: canonical_smiles(s) for s in dict.fromkeys(smiles)}
        feats = {c: featurise(c) for c in dict.fromkeys(canon.values()) if c is not None}
        order = sorted((c for c, f in feats.items() if f is not None), key=lambda c: (feats[c].n, c))
        spectra: dict[str, list[Spectrum]] = {}
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            preds = self._run([feats[c] for c in chunk], ces, adduct, instrument, top) if ces else [[]] * len(chunk)
            spectra.update(zip(chunk, preds, strict=True))
        return [list(spectra[canon[s]]) if canon[s] in spectra else None for s in smiles]

    @torch.inference_mode()
    def _run(self, mols: Sequence[MolFeatures], energies: Sequence[float], adduct: str, instrument: str,
             top: int) -> list[list[Spectrum]]:
        batch = collate(mols, self.device)
        inten, frag_mass, counts = self.net(batch, torch.tensor(energies, dtype=torch.float32, device=self.device),
                                            _ADDUCT_CLASS[adduct], INSTRUMENTS.get(instrument, INSTRUMENTS["Orbitrap"]))
        width = inten.shape[1]
        n_rows = counts * 2 * SHIFTS
        k = min(top, width)
        val, idx = torch.topk(inten.masked_fill(torch.arange(width, device=inten.device)[None, :] >= n_rows[:, None],
                                                float("-inf")), k=k, dim=1)
        val = torch.where(torch.isfinite(val), val, torch.zeros_like(val))
        frag, ion = idx // (2 * SHIFTS), idx % (2 * SHIFTS)
        h_shift = (torch.arange(-6, 7, device=idx.device, dtype=torch.float64) * _H)[ion % SHIFTS]
        ion_shift = torch.tensor([ADDUCT_MASS[adduct], -ELECTRON_MASS], dtype=torch.float64,
                                 device=idx.device)[ion // SHIFTS]
        mz = torch.gather(frag_mass, 1, frag) + h_shift + ion_shift
        mz = torch.where(mz > 0, mz, torch.zeros_like(mz)).cpu().numpy()
        val, keep = val.cpu().numpy(), torch.clamp(n_rows, max=k).cpu().numpy()
        out = [(mz[r, :keep[r]], val[r, :keep[r]].astype(np.float32)) for r in range(len(mz))]
        return [out[i * len(energies):(i + 1) * len(energies)] for i in range(len(mols))]
