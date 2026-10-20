"""ICEBERG 2.1 forward model (MassSpecGym ``msg_all`` weights) as plain PyTorch inference.

Ported from coleygroup/ms-pred @ 708148c2 (MIT License, Copyright (c) 2023 Samuel Goldman): the fragment
generator ``FragGNN`` (``iceberg/gen_model.py``), the intensity model ``IntenGNN`` (``iceberg/inten_model.py``),
their inference glue (``iceberg/joint_model.py``, ``iceberg/predict_smis.py``) and what they use from
``common/chem_utils.py`` (element, adduct, instrument tables; canonicalisation), ``magma/fragmentation.py``
(``FragmentEngine``), ``iceberg/dag_data.py`` (featurisation) and ``nn_utils`` (``MLPBlocks``, ``random_walk_pe``,
the abs-sines formula embedder, the DGL graph ops of ``dgl_graph_ops.py``). ``GatedGraphConv`` is DGL's
(Apache License 2.0), as copied into ms-pred's ``nn_utils/dgl_modules.py``.

No DGL or torch_scatter: a fragment is a boolean mask over its molecule's atoms and its graph the induced subgraph
(upstream's fragments are induced subgraphs of the root with nodes in atom order, so this is the same graph).
Messages run over per-atom neighbour slots and pooling over contiguous segments, every sum in a fixed order, so
GPU output is deterministic for a given batch (another batch composition changes GEMM shapes and so the last float
bits, as in upstream). Settings are upstream's MassSpecGym retrieval run
(``run_scripts/iceberg/msg_all/10_run_retrieval_msg_all_iceberg.py``): threshold 0, 100 DAG nodes, 3 expansion
steps, top-100 sparse rows, no binning. ``ms-pred file:line`` comments mark where the DAG expansion has to follow
upstream's DGL / torch semantics exactly (ordering, ties, de-duplication, stale rows).
"""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from torch import Tensor, nn

from casmi import degraded

RDLogger.DisableLog("rdApp.*")

SUPPORTED_ADDUCTS = ("[M+H]+", "[M+Na]+")  # the only adducts in MassSpecGym training data
INSTRUMENTS = {"Orbitrap": 0, "QTOF": 1, "IT-FT": 2, "Unknown": 3}  # anything else → Orbitrap, as predict_smis
VALID_ELEMENTS = ("C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B", "Se", "Fe", "Co", "As", "Na", "K")
ELEMENT_GROUP = {"C": 4, "N": 3, "P": 3, "O": 5, "S": 5, "Si": 4, "I": 6, "H": 0, "Cl": 6, "F": 6, "Br": 6, "B": 2,
                 "Se": 5, "Fe": 7, "Co": 7, "As": 3, "Na": 1, "K": 1}
BOND_ORDER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 2}
N_ELEM, N_GROUP, MAX_H = len(VALID_ELEMENTS), 8, 6
N_FEAT = N_ELEM + N_GROUP + MAX_H  # 32 node features before the random-walk encoding
MAX_ATOM_BONDS = 6
MAX_BROKEN = 6
N_SHIFTS = 2 * MAX_BROKEN + 1  # hydrogen shifts −6…+6
MAX_DEPTH = 3
MAX_NODES = 100
THRESHOLD = 0.0
CE_DIM, CE_SCALE = 64, 10000
N_BINS, MZ_MAX = 15000, 1500
ATOM_BUDGET = 48  # a batch of `batch` rows is sized for molecules of this many heavy atoms

ELECTRON_MASS = 0.00054858
MONO_MASSES = np.array([Chem.GetPeriodicTable().GetMostCommonIsotopeMass(e) for e in VALID_ELEMENTS])
_ELEM = {e: i for i, e in enumerate(VALID_ELEMENTS)}
_H = _ELEM["H"]
ION_MASS = {"[M+H]+": MONO_MASSES[_H] - ELECTRON_MASS, "[M+Na]+": MONO_MASSES[_ELEM["Na"]] - ELECTRON_MASS}

# ion → adduct class before the equivalent-spelling additions (ms-pred common/chem_utils.py:187-216)
_ADDUCT_CLASS = {"[M+H]+": 0, "[M+Na]+": 1, "[M+K]+": 2, "[M-H2O+H]+": 3, "[M+H-H2O]+": 3, "[M+H3N+H]+": 4,
                 "[M+NH4]+": 4, "[M]+": 5, "[M-H4O2+H]+": 6, "[M+H-2H2O]+": 6, "[M+H-H3N]+": 7, "[M-H3N+H]+": 7,
                 "[M-NH3+H]+": 7, "[M+H-NH3]+": 7, "[M+H-CH2O2]+": 8, "[M+CHO2]-": 9, "[M-H]-": 10, "[M+Cl]-": 11,
                 "[M-H2O-H]-": 12, "[M-H-H2O]-": 12, "[M-H-CO2]-": 13}
_TAUTOMER_TRANSFORMS = ("// Name SMARTS Bonds Charges\n"
                        "1,3 hetero atom H shift\t[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]\n"
                        "1,3 (thio)keto/enol r\t[O,S,Se,Te;X2!H0]-[C]=[C]\n")
_HPARAMS = {
    "gen": {"mpnn_type": "GGNN", "pool_op": "avg", "root_encode": "gnn", "inject_early": False, "set_layers": 0,
            "node_feats": N_FEAT + 14, "pe_embed_k": 14, "max_broken": MAX_BROKEN, "embed_adduct": True,
            "embed_collision": True, "embed_instrument": True, "embed_elem_group": True, "encode_forms": True,
            "add_hs": True},
    "inten": {"mpnn_type": "GGNN", "pool_op": "avg", "root_encode": "gnn", "inject_early": False, "set_layers": 0,
              "node_feats": N_FEAT, "pe_embed_k": 0, "max_broken": MAX_BROKEN, "mlp_layers": 1, "embed_adduct": True,
              "embed_collision": True, "embed_instrument": True, "embed_elem_group": True, "encode_forms": True,
              "add_hs": True, "include_unshifted_mz": True, "loss_fn": "cosine"},
}
Spectrum = tuple[NDArray[np.float64], NDArray[np.float32]]

# --- chemistry (common/chem_utils.py, magma/fragmentation.py, iceberg/dag_data.py) ---------------------------


@lru_cache(maxsize=1)
def _tautomer_enumerator() -> rdMolStandardize.TautomerEnumerator:
    """ms-pred's tautomer canonicaliser on RDKit without the Python MolStandardize (chem_utils.py:305-321): two
    transforms, read from a transforms file when the enumerator is built."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tautomer_transforms.txt"
        path.write_text(_TAUTOMER_TRANSFORMS)
        params = rdMolStandardize.CleanupParameters()
        params.tautomerTransformsFile = str(path)
        return rdMolStandardize.TautomerEnumerator(params)


def canonical_smiles(smiles: str) -> str | None:
    """The root SMILES upstream predicts for: stereo removed, then the InChI round trip with tautomer
    canonicalisation (joint_model.py:108-111, chem_utils.py:546-558,596-685; the training DAGs' roots went through
    the same round trip, fragmentation.py:54-59). ``predict_smis.py:182,242`` skips it (canonical_root_smi=True)
    and relies on pre-canonical input. None when RDKit fails."""
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
        return Chem.MolToSmiles(_tautomer_enumerator().Canonicalize(mol)) or None
    except Exception:  # RDKit raises on exotic input; one bad molecule must not stop a batch
        return None


def formula_vector(formula: str) -> NDArray[np.float32] | None:
    """Element counts in ``VALID_ELEMENTS`` order (chem_utils.py:329-352); None for another element."""
    v = np.zeros(N_ELEM, np.float32)
    for sym, num in re.findall(r"([A-Z][a-z]*)([0-9]*)", formula):
        if sym not in _ELEM:
            return None
        v[_ELEM[sym]] += int(num) if num else 1
    return v


@dataclass(frozen=True)
class Molecule:
    """A root molecule as ``FragmentEngine`` + ``TreeProcessor.dgl_featurize`` see it (fragmentation.py:47-146,
    dag_data.py:404-455): kekulised, atoms in the canonical SMILES order."""

    smiles: str
    feat: NDArray[np.float32]  # [n, 32] element one-hot | element group one-hot | H-count one-hot
    hs: NDArray[np.int64]  # [n] hydrogens per atom
    nbr: NDArray[np.int64]  # [n, MAX_ATOM_BONDS] bonded atoms, −1 padded
    order: NDArray[np.int64]  # [n, MAX_ATOM_BONDS] bond orders (TYPEW: aromatic and double 2), 0 padded
    form: NDArray[np.float32]  # [18] formula of the root SMILES

    @property
    def natoms(self) -> int:
        return len(self.hs)


def featurise(smiles: str) -> Molecule | None:
    """The molecule upstream would predict for ``smiles``, or None where upstream fails or the input is out of
    scope: invalid SMILES, an element outside ``VALID_ELEMENTS``, more than ``MAX_ATOM_BONDS`` bonds or ``MAX_H``
    or more hydrogens on an atom, another bond type, a net charge or several components."""
    root = canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(root) if root else None
    if mol is None or mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return None
    form = formula_vector(re.findall(r"^([^\+,^\-]*)", CalcMolFormula(mol))[0])  # chem_utils.py:526-570
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    n = mol.GetNumAtoms()
    feat = np.zeros((n, N_FEAT), np.float32)
    hs = np.zeros(n, np.int64)
    for a in mol.GetAtoms():
        sym, i = a.GetSymbol(), a.GetIdx()
        hs[i] = a.GetNumImplicitHs() + a.GetNumExplicitHs()
        if sym not in _ELEM or hs[i] >= MAX_H:
            return None
        feat[i, _ELEM[sym]] = feat[i, N_ELEM + ELEMENT_GROUP[sym]] = feat[i, N_ELEM + N_GROUP + hs[i]] = 1
    nbr = np.full((n, MAX_ATOM_BONDS), -1, np.int64)
    order = np.zeros((n, MAX_ATOM_BONDS), np.int64)
    deg = np.zeros(n, np.int64)
    for b in mol.GetBonds():
        t = BOND_ORDER.get(b.GetBondType())
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if t is None or deg[i] == MAX_ATOM_BONDS or deg[j] == MAX_ATOM_BONDS:  # fragmentation.py:106-138
            return None
        nbr[i, deg[i]], order[i, deg[i]], nbr[j, deg[j]], order[j, deg[j]] = j, t, i, t
        deg[i] += 1
        deg[j] += 1
    if form is None:
        return None
    return Molecule(root, feat, hs, nbr, order, form)


def _adduct_table() -> Tensor:
    """14 × 22: adduct class one-hot | 8 "mode" bits (pos, neg, alkali, +H, −H, +Cl, −H2O, −CO2); ms-pred
    chem_utils.py:220-255 and gen_model.py:121-133."""
    modes: dict[int, set[int]] = {v: set() for v in _ADDUCT_CLASS.values()}
    for ion, v in _ADDUCT_CLASS.items():
        sign = ion[-1]
        parts = []
        for i, p in enumerate(ion.strip(sign).strip("[M").strip("]").split("+")):
            for j, q in enumerate(("+" + p if i else p).split("-") if p else []):
                if q:
                    parts.append(q if j == 0 else "-" + q)
        modes[v].add(0 if sign == "+" else 1)
        for bit, tags in ((2, ("+Na", "+K")), (3, ("+H",)), (4, ("-H",)), (5, ("+Cl",)),
                          (6, ("-H2O", "-H4O2", "-2H2O")), (7, ("-CO2",)), (8, ("+NH3",))):
            if any(t in parts for t in tags):
                modes[v].add(bit)
    n, m = len(modes), len({b for s in modes.values() for b in s})
    multi = torch.zeros(n, m)
    for i, bits in modes.items():
        multi[i, sorted(bits)] = 1
    return torch.cat([torch.eye(n), multi], 1)


# --- packed fragment graphs (replaces DGL) -----------------------------------------------------------------


@dataclass
class _Graphs:
    """Induced subgraphs of fragment masks; nodes packed fragment by fragment, atoms ascending within one."""

    frag: Tensor  # [N] fragment of each node
    atom: Tensor  # [N] its atom index in the root
    sizes: Tensor  # [G] nodes per fragment
    slots: Tensor  # [N, d] neighbour node per slot, −1 for none (valid slots first)
    order: Tensor  # [N, d] bond order per slot, 0 for none
    _msg: tuple[list[tuple[int, Tensor]], Tensor] | None = None

    def messages(self) -> tuple[list[tuple[int, Tensor]], Tensor]:
        """Per bond order the nodes that send it, and per slot the row of the message table (last row zero)."""
        if self._msg is None:
            n = self.frag.numel()
            senders, pos, off = [], torch.full((n, 4), -1, dtype=torch.long, device=self.frag.device), 0
            for t in (1, 2, 3):
                s = (self.order == t).any(1).nonzero().squeeze(1)
                if s.numel():
                    pos[s, t] = torch.arange(off, off + s.numel(), device=s.device)
                    senders.append((t, s))
                    off += s.numel()
            index = torch.where(self.slots >= 0, pos[self.slots.clamp(min=0), self.order], off)
            self._msg = senders, index
        return self._msg


def _graphs(masks: Tensor, mol: Tensor, nbr: Tensor, order: Tensor) -> _Graphs:
    frag, atom = masks.nonzero(as_tuple=True)
    flat = torch.full(masks.shape, -1, dtype=torch.long, device=masks.device)
    flat[frag, atom] = torch.arange(frag.numel(), device=masks.device)
    nb = nbr[mol[frag], atom]
    slots = torch.where(nb >= 0, flat[frag[:, None], nb.clamp(min=0)], -1)
    srt = (slots < 0).to(torch.int8).argsort(dim=1, stable=True)
    slots = slots.gather(1, srt)
    deg = int((slots >= 0).sum(1).max()) if frag.numel() else 0
    slots = slots[:, :deg]
    bond = torch.where(slots >= 0, order[mol[frag], atom].gather(1, srt)[:, :deg], 0)
    return _Graphs(frag, atom, masks.sum(1), slots, bond)


def _mean(x: Tensor, sizes: Tensor) -> Tensor:
    """Per-fragment mean, DGL ``AvgPooling`` (deterministic segment sum)."""
    return torch.segment_reduce(x, "mean", lengths=sizes, axis=0)


def _random_walk_pe(g: _Graphs, k: int) -> Tensor:
    """Landing probabilities of 1…k-step random walks, edge-weighted by bond order (nn_utils.py:1104-1180)."""
    n, sizes = g.frag.numel(), g.sizes
    start = torch.cumsum(sizes, 0) - sizes
    local = torch.arange(n, device=sizes.device) - start[g.frag]
    w = torch.zeros(len(sizes), int(sizes.max()), int(sizes.max()), device=sizes.device)
    i, s = (g.slots >= 0).nonzero(as_tuple=True)
    w[g.frag[i], local[i], local[g.slots[i, s]]] = g.order[i, s].float()
    p = w / (w.sum(-1, keepdim=True) + 1e-30)
    out, pe = p, [p.diagonal(dim1=1, dim2=2)]
    for _ in range(k - 1):
        out = torch.bmm(out, p)
        pe.append(out.diagonal(dim1=1, dim2=2))
    return torch.stack(pe, -1)[g.frag, local]


def _pad(x: Tensor, rows: Tensor, counts: Tensor) -> Tensor:
    """``pad_packed_tensor`` (nn_utils.py:1056-1083): ``x`` grouped by ``rows`` → [R, max(counts), ...], 0-padded."""
    out = x.new_zeros((len(counts), int(counts.max()), *x.shape[1:]))
    out[rows, torch.arange(len(rows), device=x.device) - (torch.cumsum(counts, 0) - counts)[rows]] = x
    return out


def _first_occurrence(key: Tensor) -> Tensor:
    """[R, T] True where row ``key[r, t]`` is the first of its value within batch row r: ``np_like_unique``
    keeps np.unique's first occurrence (dgl_graph_ops.py:439-448, used at gen_model.py:650-656)."""
    r, t, _ = key.shape
    rows = torch.arange(r, device=key.device, dtype=key.dtype).repeat_interleave(t)[:, None]
    _, inv = torch.unique(torch.cat([rows, key.reshape(r * t, -1)], 1), dim=0, return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), r * t, device=key.device).scatter_reduce_(
        0, inv, torch.arange(r * t, device=key.device), "amin")
    out = torch.zeros(r * t, dtype=torch.bool, device=key.device)
    out[first] = True
    return out.view(r, t)


# --- network modules (state_dict names as upstream) ----------------------------------------------------------


class _GatedGraphConv(nn.Module):
    """DGL ``GatedGraphConv`` (dgl_modules.py:23-182): a = Σ_in-edges W_type h_src + b_type, h = GRU(a, h)."""

    def __init__(self, dim: int, steps: int, n_types: int = 4):
        super().__init__()
        self.steps = steps
        self.linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_types))
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, feat: Tensor, g: _Graphs) -> Tensor:
        senders, index = g.messages()
        for _ in range(self.steps):
            table = torch.cat([self.linears[t](feat[s]) for t, s in senders] + [feat.new_zeros(1, feat.shape[1])])
            a = (F.embedding_bag(index, table, mode="sum", padding_idx=len(table) - 1) if index.shape[1]
                 else feat.new_zeros(feat.shape))  # per-node sum over its slots, fixed order (no atomics)
            feat = self.gru(a, feat)
        return feat


class _GGNN(nn.Module):
    def __init__(self, dim: int, steps: int):
        super().__init__()
        self.model = _GatedGraphConv(dim, steps)


class _MoleculeGNN(nn.Module):
    """``MoleculeGNN`` with GGNN and no set-transformer layers (nn_utils.py:65-156)."""

    def __init__(self, n_in: int, dim: int, steps: int):
        super().__init__()
        self.input_project = nn.Linear(n_in, dim)
        self.gnn = _GGNN(dim, steps)

    def forward(self, x: Tensor, g: _Graphs) -> Tensor:
        return self.gnn.model(self.input_project(x), g)


class _MLP(nn.Module):
    """``MLPBlocks`` with ``num_layers=1`` (nn_utils.py:315-383): Linear, ReLU, output Linear."""

    def __init__(self, n_in: int, dim: int, n_out: int):
        super().__init__()
        self.input_layer = nn.Linear(n_in, dim)
        self.output_layer = nn.Linear(dim, n_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(F.relu(self.input_layer(x)))


class _AbsSines(nn.Module):
    """Formula-count embedder "abs-sines" (form_embedder.py:8-59,137-169): |sin(2π n f)| at 8 frequencies;
    counts ≥ 255 take the learnt extra row."""

    MAX_COUNT = 255

    def __init__(self) -> None:
        super().__init__()
        freqs = (0.5 ** torch.arange(int(np.ceil(np.log2(self.MAX_COUNT))) + 2, dtype=torch.float32))[2:]
        args = torch.arange(self.MAX_COUNT, dtype=torch.float32)[:, None] * (2 * np.pi * freqs)[None, :]
        self.register_buffer("int_to_feat_matrix", torch.abs(torch.sin(args)).float())
        self._extra_embeddings = nn.Parameter(torch.zeros(1, len(freqs)), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        big = x >= self.MAX_COUNT
        e = self.int_to_feat_matrix[torch.where(big, 0, x.long())]
        return torch.where(big[..., None], self._extra_embeddings[0], e).reshape(*x.shape[:-1], -1)


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        b, n, dim = x.shape
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2)
                   for t in F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, -1))
        att = (q / math.sqrt(dim // self.heads)) @ k.transpose(-2, -1)
        att = att.masked_fill(pad[:, None, None, :], float("-inf")).softmax(-1)
        return self.out_proj((att @ v).transpose(1, 2).reshape(b, n, dim))


class _TransformerLayer(nn.Module):
    """Post-norm encoder layer, ReLU feed-forward (nn_utils/transformer_layer.py:19-153, scaled-dot path)."""

    def __init__(self, dim: int, heads: int, ff: int):
        super().__init__()
        self.self_attn = _SelfAttention(dim, heads)
        self.linear1, self.linear2 = nn.Linear(dim, ff), nn.Linear(ff, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x: Tensor, pad: Tensor) -> Tensor:
        x = self.norm1(x + self.self_attn(x, pad))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


class _Conditioned(nn.Module):
    """Buffers both ICEBERG models share: broken-bond one-hot, adduct, collision-energy and instrument tables."""

    COMPUTED = ("broken_onehot", "adduct_embedder", "collision_embedder_denominators", "collision_embed_merged",
                "instrument_embedder", "embedder.int_to_feat_matrix")

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("broken_onehot", torch.eye(MAX_BROKEN + 1))
        self.register_buffer("adduct_embedder", _adduct_table())
        self.register_buffer("collision_embedder_denominators",
                             torch.pow(CE_SCALE, 2 * torch.arange(CE_DIM // 2) / CE_DIM))
        self.register_buffer("collision_embed_merged", torch.zeros(CE_DIM))
        self.register_buffer("instrument_embedder", torch.eye(len(INSTRUMENTS)))
        self.embedder = _AbsSines()

    def context(self, adduct: Tensor, energy: Tensor, instrument: Tensor) -> Tensor:
        """[R, 90] adduct | sinusoidal CE (NaN → zeros) | instrument (gen_model.py:263-318)."""
        arg = energy[:, None] / self.collision_embedder_denominators[None, :]
        ce = torch.cat([torch.sin(arg), torch.cos(arg)], 1)
        ce = torch.where(torch.isnan(ce), self.collision_embed_merged[None, :], ce)
        return torch.cat([self.adduct_embedder[adduct], ce, self.instrument_embedder[instrument]], 1)

    def onehot(self, broken: Tensor) -> Tensor:
        return self.broken_onehot[broken.clamp(max=MAX_BROKEN).long()]


class _FragGNN(_Conditioned):
    """``FragGNN`` (gen_model.py:17-403): per atom of a fragment, the probability that removing it is the next step."""

    def __init__(self, hidden: int, layers: int, pe_k: int):
        super().__init__()
        self.hidden, self.pe_k = hidden, pe_k
        self.gnn = _MoleculeGNN(N_FEAT + pe_k + 22 + CE_DIM + len(INSTRUMENTS), hidden, layers)
        self.root_module = self.gnn
        self.output_map = _MLP(3 * hidden + MAX_BROKEN + 1 + 2 * N_ELEM * 8, hidden, 1)
        self._split: tuple[Tensor, Tensor] | None = None

    def encode(self, b: _Batch, masks: Tensor, rows: Tensor, ctx: Tensor) -> tuple[_Graphs, Tensor, Tensor]:
        """Node embeddings and their per-fragment mean for the fragments ``masks`` of batch rows ``rows``."""
        g = _graphs(masks, b.mol[rows], b.nbr, b.order)
        x = torch.cat([b.feat[b.mol[rows][g.frag], g.atom], _random_walk_pe(g, self.pe_k), ctx[rows][g.frag]], 1)
        node = self.gnn(x, g)
        return g, node, _mean(node, g.sizes)

    def leaving(self, g: _Graphs, node: Tensor, avg: Tensor, root: Tensor, broken: Tensor, forms: Tensor,
                root_forms: Tensor, atoms: int) -> Tensor:
        """[G, atoms] removal probabilities (0 off the fragment). The output MLP's input layer is split into its
        per-fragment columns and its per-node columns; same sum as upstream's concatenation (gen_model.py:378-402)."""
        if self._split is None:
            w, h = self.output_map.input_layer.weight, self.hidden
            self._split = torch.cat([w[:, :2 * h], w[:, 3 * h:]], 1).contiguous(), w[:, 2 * h:3 * h].contiguous()
        w_frag, w_node = self._split
        per = F.linear(torch.cat([root, root - avg, self.onehot(broken), self.embedder(forms),
                                  self.embedder(root_forms - forms)], 1), w_frag, self.output_map.input_layer.bias)
        hid = F.relu(per[g.frag] + F.linear(node, w_node))
        p = torch.sigmoid(self.output_map.output_layer(hid)).squeeze(-1)
        out = p.new_zeros(len(g.sizes), atoms)
        out[g.frag, g.atom] = p
        return out

    def generate(self, b: _Batch) -> dict[str, Tensor]:
        """The fragment DAG per batch row: ``predict_mol`` (gen_model.py:472-731) with max_nodes 100,
        threshold 0. Ties in every argsort are broken by position, as upstream's CUDA sort does for rows of more
        than 32 entries (radix / merge sort, stable); hence ``stable=True`` throughout."""
        dev, r, a, k = b.feat.device, len(b.mol), b.feat.shape[1], MAX_NODES
        ar = torch.arange(r, device=dev)
        ctx = self.context(b.adduct, b.energy, b.instrument)
        root_mask = torch.arange(a, device=dev)[None, :] < b.natoms[b.mol][:, None]
        root_forms = b.form[b.mol]
        g, node, root = self.encode(b, root_mask, ar, ctx)  # the depth-0 frontier is the root itself

        prob = torch.zeros(r, k, device=dev)
        prob[:, 0] = 1
        brokens = torch.zeros(r, k, device=dev)
        frags = torch.zeros(r, k, a, dtype=torch.bool, device=dev)
        frags[:, 0] = root_mask
        masses = torch.zeros(r, k, device=dev)
        masses[:, 0] = _mass(root_forms)
        forms = torch.zeros(r, k, N_ELEM, device=dev)
        forms[:, 0] = root_forms
        hashes = torch.zeros(r, k, N_FEAT, device=dev)
        hashes[:, 0] = _hash(b, root_mask, b.mol)

        f_mask, f_row, f_broken, f_accu, f_forms, avg = root_mask, ar, torch.zeros(r, device=dev), \
            torch.ones(r, device=dev), root_forms, root
        for depth in range(MAX_DEPTH):  # gen_model.py:570 (no expansion of the last level)
            if depth:
                g, node, avg = self.encode(b, f_mask, f_row, ctx)
            leave = self.leaving(g, node, avg, root[f_row], f_broken, f_forms, root_forms[f_row], a)
            counts = torch.bincount(f_row, minlength=r)
            # gen_model.py:587-589: accumulated probabilities, rows laid out (fragment, atom); upstream lays them
            # out (fragment, node position), and node positions ascend with atom index, so the order is the same
            accu = _pad(f_accu[:, None] * leave, f_row, counts).view(r, -1)
            temp = torch.cat([prob, accu], 1)
            kept = temp.argsort(dim=1, descending=True, stable=True)[:, :2 * k]  # gen_model.py:594
            sel = (kept >= k) & (temp.gather(1, kept) > THRESHOLD)  # gen_model.py:597-598
            sb, sp = sel.nonzero(as_tuple=True)  # jobs ordered by row, then by rank
            if not sb.numel():  # gen_model.py:606
                break
            si = kept[sb, sp] - k
            parent = (torch.cumsum(counts, 0) - counts)[sb] + torch.div(si, a, rounding_mode="floor")
            split = _remove_atoms(b, f_mask[parent], si % a, b.mol[f_row[parent]])
            if split is None:  # gen_model.py:615
                break
            n_mask, job, added = split
            n_row, n_prob = sb[job], accu[sb, si][job]
            n_broken = f_broken[parent][job] + added  # gen_model.py:617
            n_count = torch.bincount(n_row, minlength=r)
            n_hash = _hash(b, n_mask, b.mol[n_row])
            t_frags = torch.cat([frags, _pad(n_mask, n_row, n_count)], 1)  # gen_model.py:633-635
            t_prob = torch.cat([prob, _pad(n_prob, n_row, n_count)], 1)  # gen_model.py:639-640
            t_hash = torch.cat([hashes, _pad(n_hash, n_row, n_count)], 1)  # gen_model.py:645-647
            # gen_model.py:650-661: a fragment whose hash was seen earlier in its row (kept nodes first, then new
            # ones in job order) is dropped; zero-probability rows are never marked
            t_prob[~_first_occurrence(t_hash) & (t_prob > 0)] = -1
            nk = t_prob.argsort(dim=1, descending=True, stable=True)[:, :k]  # gen_model.py:663
            prob = t_prob.gather(1, nk)
            frags = t_frags[ar[:, None], nk]
            brokens = torch.cat([brokens, _pad(n_broken, n_row, n_count)], 1).gather(1, nk)  # gen_model.py:672-683
            ti, tp = ((nk >= k) & (prob > THRESHOLD)).nonzero(as_tuple=True)  # gen_model.py:686-690
            flat = nk[ti, tp] - k + (torch.cumsum(n_count, 0) - n_count)[ti]
            if not flat.numel():  # gen_model.py:692
                break
            f_mask, f_row, f_broken, f_accu = n_mask[flat], n_row[flat], n_broken[flat], n_prob[flat]
            f_forms = _forms(b, f_mask, b.mol[f_row])
            # gen_model.py:703-710: kept old rows move to their new slot, kept new rows get their values; new
            # slots whose probability is not above the threshold keep their previous (stale) mass, form and hash
            rr = nk < k
            ri = rr.nonzero(as_tuple=True)[0]
            masses[rr] = masses[ri, nk[rr]]
            masses[ti, tp] = _mass(f_forms)
            forms[rr] = forms[ri, nk[rr]]
            forms[ti, tp] = f_forms
            hashes[rr] = hashes[ri, nk[rr]]
            hashes[ti, tp] = n_hash[flat]
        return {"frags": frags, "probs": prob, "brokens": brokens, "masses": masses, "forms": forms,
                "root_forms": root_forms, "nfrags": (prob > THRESHOLD).sum(1)}


class _IntenGNN(_Conditioned):
    """``IntenGNN`` (inten_model.py:20-716): per DAG node, intensities at 13 H shifts × 2 m/z groups."""

    COMPUTED = _Conditioned.COMPUTED + ("inten_buckets",)

    def __init__(self, hidden: int, layers: int, set_layers: int):
        super().__init__()
        self.register_buffer("inten_buckets", torch.from_numpy(np.linspace(0, MZ_MAX, N_BINS)))
        self.gnn = _MoleculeGNN(N_FEAT + 22 + CE_DIM + len(INSTRUMENTS), hidden, layers)
        self.root_module = self.gnn
        self.intermediate_out = _MLP(3 * hidden + MAX_BROKEN + 1 + 2 * N_ELEM * 8, hidden, hidden)
        self.trans_layers = nn.ModuleList(_TransformerLayer(hidden, 8, 4 * hidden) for _ in range(set_layers))
        self.output_map = nn.Linear(hidden, 2 * N_SHIFTS)
        self.isomer_attn_out = nn.Linear(hidden, 2 * N_SHIFTS)

    def _embed(self, b: _Batch, masks: Tensor, rows: Tensor, ctx: Tensor) -> Tensor:
        g = _graphs(masks, b.mol[rows], b.nbr, b.order)
        x = torch.cat([b.feat[b.mol[rows][g.frag], g.atom], ctx[rows][g.frag]], 1)
        return _mean(self.gnn(x, g), g.sizes)

    def forward(self, b: _Batch, dag: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Raw outputs and attention logits [R, F, 26] per DAG node (inten_model.py:474-644), F = max nodes."""
        dev, r, a = b.feat.device, len(b.mol), b.feat.shape[1]
        nf = dag["nfrags"]
        nmax = int(nf.max())
        ctx = self.context(b.adduct, b.energy, b.instrument)
        ar = torch.arange(r, device=dev)
        root = self._embed(b, torch.arange(a, device=dev)[None, :] < b.natoms[b.mol][:, None], ar, ctx)
        fb, ff = (torch.arange(nmax, device=dev)[None, :] < nf[:, None]).nonzero(as_tuple=True)
        avg = self._embed(b, dag["frags"][fb, ff], fb, ctx)
        ext = root[fb]
        hid = torch.zeros(r, nmax, 3 * avg.shape[1] + MAX_BROKEN + 1, device=dev)
        hid[fb, ff] = torch.cat([ext, ext - avg, avg, self.onehot(dag["brokens"][fb, ff])], 1)
        forms = dag["forms"][:, :nmax]
        hid = torch.cat([hid, self.embedder(forms), self.embedder(dag["root_forms"][:, None] - forms)], -1)
        h = self.intermediate_out(hid)
        pad = torch.arange(nmax, device=dev)[None, :] >= nf[:, None]
        for layer in self.trans_layers:
            h = layer(h, pad)
        return self.output_map(h), self.isomer_attn_out(h)


# --- DAG bookkeeping ---------------------------------------------------------------------------------------


@dataclass
class _Batch:
    """Rows (molecule, energy) on the device; per-molecule tables indexed by ``mol``."""

    mol: Tensor  # [R] table index per row
    adduct: Tensor  # [R]
    energy: Tensor  # [R] eV
    instrument: Tensor  # [R]
    natoms: Tensor  # [M]
    feat: Tensor  # [M, A, 32]
    hs: Tensor  # [M, A]
    nbr: Tensor  # [M, A, 6]
    order: Tensor  # [M, A, 6]
    form: Tensor  # [M, 18]


def _batch(mols: Sequence[Molecule], mol: Sequence[int], adduct: str, energy: Sequence[float], instrument: str,
           device: torch.device) -> _Batch:
    a = max(m.natoms for m in mols)
    feat = np.zeros((len(mols), a, N_FEAT), np.float32)
    hs = np.zeros((len(mols), a), np.int64)
    nbr = np.full((len(mols), a, MAX_ATOM_BONDS), -1, np.int64)
    order = np.zeros((len(mols), a, MAX_ATOM_BONDS), np.int64)
    for i, m in enumerate(mols):
        feat[i, :m.natoms], hs[i, :m.natoms], nbr[i, :m.natoms], order[i, :m.natoms] = m.feat, m.hs, m.nbr, m.order

    def t(x: object, dtype: torch.dtype | None = None) -> Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype).to(device)

    r = len(mol)
    return _Batch(t(mol), t([_ADDUCT_CLASS[adduct]] * r), t(energy, torch.float32), t([INSTRUMENTS[instrument]] * r),
                  t([m.natoms for m in mols]), t(feat), t(hs), t(nbr), t(order), t(np.stack([m.form for m in mols])))


def _hash(b: _Batch, masks: Tensor, mol: Tensor) -> Tensor:
    """``msg_passing_frag_graph_hash`` (dgl_graph_ops.py:451-472): Σ_edges u→v h_u·bond order, i.e. per node its
    32 features times its in-fragment bond-order valence, summed. Small integers, exact in float32."""
    nb = b.nbr[mol]
    inside = (nb >= 0) & masks.gather(1, nb.clamp(min=0).flatten(1)).view(nb.shape) & masks[:, :, None]
    val = (b.order[mol] * inside).sum(-1).float()
    return (val[:, :, None] * b.feat[mol]).sum(1)


def _forms(b: _Batch, masks: Tensor, mol: Tensor) -> Tensor:
    """``frag_to_form_vec`` (dgl_graph_ops.py:421-436): element counts, the H column replaced by the atoms' H."""
    m = masks.float()
    forms = (m[:, :, None] * b.feat[mol][:, :, :N_ELEM]).sum(1)
    forms[:, _H] = (m * b.hs[mol]).sum(1)
    return forms


def _mass(forms: Tensor) -> Tensor:
    """Neutral monoisotopic mass, float32 like ``form_vec_to_mass`` (dgl_graph_ops.py:411-418), summed in float64."""
    return (forms.double() @ torch.as_tensor(MONO_MASSES, device=forms.device)).float()


def _remove_atoms(b: _Batch, parents: Tensor, removed: Tensor, mol: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
    """``batch_remove_single_atoms`` (dgl_graph_ops.py:71-218): delete one atom per job, keep each connected
    component of ≥ 2 atoms. Returns the component masks ordered by job, then by smallest atom (upstream's
    component ids are the smallest node index, ``torch.unique`` sorted, then ``argsort``), their job, and the bond
    orders cut between the removed atom and the component (upstream: ⌊Σ over both edge directions / 2⌋)."""
    k, a = parents.shape
    keep = parents.clone()
    keep[torch.arange(k, device=keep.device), removed] = False
    nb = b.nbr[mol]
    nbc = nb.clamp(min=0).flatten(1)
    alive = (nb >= 0) & keep.gather(1, nbc).view(nb.shape) & keep[:, :, None]
    ids = torch.arange(a, device=keep.device).expand(k, a)
    lab = torch.where(keep, ids, a)  # label = smallest atom of the component (connected_components, :12-68)
    while True:
        nxt = torch.where(alive, lab.gather(1, nbc).view(nb.shape), a).amin(-1)
        new = torch.minimum(lab, nxt)
        new = torch.where(keep, torch.minimum(new, new.gather(1, new.clamp(max=a - 1))), a)
        if torch.equal(new, lab):
            break
        lab = new
    size = torch.zeros(k, a + 1, dtype=torch.long, device=keep.device).scatter_add_(1, lab, keep.long())
    head = keep & (lab == ids) & (size[:, :a] >= 2)  # isolated atoms are dropped (:55-66, :170)
    job, rep = head.nonzero(as_tuple=True)
    if not job.numel():
        return None
    masks = (lab[job] == rep[:, None]) & keep[job]
    rn = b.nbr[mol[job], removed[job]]
    cut = (rn >= 0) & masks.gather(1, rn.clamp(min=0))
    return masks, job, (b.order[mol[job], removed[job]] * cut).sum(1).float()


def _spectra(out: Tensor, att: Tensor, dag: dict[str, Tensor], adduct: str, buckets: Tensor,
             top: int) -> list[Spectrum]:
    """Masks, bin-wise softmax pooling and unbinning (inten_model.py:615-716, joint_model.py:137-191) on the CPU,
    then the top-``top`` rows by intensity (predict_smis.py:256-259). An invalid shift has weight −99999, so its
    share of a bin with a valid entry is exactly 0 and it gets intensity 0; only valid entries are pooled."""
    r, nmax, _ = out.shape
    nf = dag["nfrags"]
    out, att = out.view(r, nmax, 2, N_SHIFTS), att.view(r, nmax, 2, N_SHIFTS)
    broken, nh = dag["brokens"][:, :nmax], dag["forms"][:, :nmax, _H]
    s = torch.arange(N_SHIFTS)
    ok = ((s <= broken[..., None] + MAX_BROKEN) & (s >= MAX_BROKEN - torch.minimum(broken, nh)[..., None])
          & (torch.arange(nmax)[None, :] < nf[:, None])[..., None])
    ok = ok[:, :, None, :].expand(r, nmax, 2, N_SHIFTS)
    offset = (torch.tensor([ION_MASS[adduct], -ELECTRON_MASS], dtype=torch.float64)[:, None]
              + ((torch.arange(N_SHIFTS) - MAX_BROKEN) * float(MONO_MASSES[_H])).double())  # joint_model.py:142-147
    mz = dag["masses"][:, :nmax, None, None].double() + offset
    bins = torch.bucketize(mz, buckets, right=False).clamp(max=N_BINS - 1)
    key = (torch.arange(r)[:, None, None, None] * N_BINS + bins)[ok]
    w, o = att[ok], out[ok]
    _, inv = torch.unique(key, return_inverse=True)
    m = torch.full((int(inv.max()) + 1,), -torch.inf).scatter_reduce_(0, inv, w, "amax")
    e = (w - m[inv]).exp()
    pw = e / torch.zeros_like(m).index_add_(0, inv, e)[inv]
    inten = torch.zeros(r, nmax, 2, N_SHIFTS)
    inten[ok] = torch.sigmoid(torch.zeros_like(m).index_add_(0, inv, pw * o))[inv] * pw
    spectra = []
    for i in range(r):
        n = int(nf[i])
        mzs, its = mz[i, :n].reshape(-1).numpy(), inten[i, :n].reshape(-1).numpy()
        best = np.argsort(its, kind="stable")[::-1][:top]
        spectra.append((mzs[best].astype(np.float64), its[best].astype(np.float32)))
    return spectra


# --- public model ------------------------------------------------------------------------------------------


def _read(path: Path, name: str) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    hp = blob["hyper_parameters"]
    bad = {k: (hp.get(k), v) for k, v in _HPARAMS[name].items() if hp.get(k) != v}
    if bad:
        raise ValueError(f"{path}: unsupported ICEBERG {name} hyper-parameters (file, port): {bad}")
    return blob


def _fill(module: _Conditioned, sd: dict[str, Tensor], path: Path) -> None:
    """``load_state_dict(strict=True)``; the buffers upstream computes must equal the stored ones."""
    if any(not torch.equal(sd[k], sd["root_module" + k[3:]]) for k in sd if k.startswith("gnn.")):
        raise ValueError(f"{path}: root_module and gnn differ (upstream shares them)")
    computed = {k: module.state_dict()[k].clone() for k in module.COMPUTED}
    module.load_state_dict(sd, strict=True)
    stored = module.state_dict()
    diff = [k for k, v in computed.items() if v.dtype != stored[k].dtype or not torch.equal(v, stored[k])]
    if diff:
        raise ValueError(f"{path}: stored buffers differ from the ones upstream computes: {diff}")


class Iceberg:
    """ICEBERG generator + intensity model. ``predict`` gives upstream's sparse spectra (m/z, intensity)."""

    def __init__(self, gen: _FragGNN, inten: _IntenGNN, device: torch.device):
        self.gen, self.inten, self.device = gen.eval(), inten.eval(), device

    @classmethod
    def load(cls, gen_path: Path, inten_path: Path, device: str | None = None) -> Iceberg:
        """Both weights-only state_dicts (``{state_dict, hyper_parameters}``), loaded strictly; the buffers upstream
        computes (embedding tables, sinusoids, bins) must equal the stored ones. ``device`` None: CUDA if any."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                degraded.note("ICEBERG on the CPU (no GPU)")
        bg, bi = _read(Path(gen_path), "gen"), _read(Path(inten_path), "inten")
        hg, hi = bg["hyper_parameters"], bi["hyper_parameters"]
        gen = _FragGNN(hg["hidden_size"], hg["layers"], hg["pe_embed_k"])
        inten = _IntenGNN(hi["hidden_size"], hi["gnn_layers"], hi["frag_set_layers"])
        _fill(gen, bg["state_dict"], Path(gen_path))
        _fill(inten, bi["state_dict"], Path(inten_path))
        dev = torch.device(device)
        return cls(gen.to(dev), inten.to(dev), dev)

    @torch.inference_mode()
    def predict(self, smiles: Sequence[str], adduct: str, energies: Sequence[float], instrument: str = "QTOF",
                top: int = 100, batch: int = 32) -> list[list[Spectrum] | None]:
        """Per SMILES, one spectrum per energy (eV): the ``top`` rows by intensity, m/z float64 and intensity
        float32, highest first; None for a molecule upstream cannot handle (``featurise``) or an adduct outside
        ``SUPPORTED_ADDUCTS``. ``instrument`` outside ``INSTRUMENTS`` counts as Orbitrap (predict_smis.py:78-81).
        A GPU batch holds up to ``batch`` (molecule, energy) rows, fewer for molecules above ``ATOM_BUDGET``
        atoms; the DAG depends on the energy, so only the CPU featurisation is shared between energies. Inputs
        with the same canonical SMILES (e.g. stereoisomers) are predicted once. A NaN energy gets the model's
        no-energy embedding (zeros, gen_model.py:289-291); predict_smis.py:186-190 skips such rows."""
        energies = [float(e) for e in energies]
        feats = [featurise(s) for s in smiles] if adduct in SUPPORTED_ADDUCTS else [None] * len(smiles)
        uniq = list({m.smiles: m for m in feats if m is not None}.values())
        res: list[list[Spectrum] | None] = [[None] * len(energies) for _ in uniq]
        inst = instrument if instrument in INSTRUMENTS else "Orbitrap"
        rows = sorted(((i, j) for i in range(len(uniq)) for j in range(len(energies))),
                      key=lambda ij: -uniq[ij[0]].natoms)
        start = 0
        while start < len(rows):
            cap = max(1, batch * ATOM_BUDGET // max(ATOM_BUDGET, uniq[rows[start][0]].natoms))
            chunk = rows[start:start + cap]
            for (i, j), spec in zip(chunk, self._run(chunk, uniq, adduct, energies, inst, top)):
                if spec is None:
                    res[i] = None
                elif res[i] is not None:
                    res[i][j] = spec
            start += len(chunk)
        at = {m.smiles: i for i, m in enumerate(uniq)}
        return [None if m is None or res[at[m.smiles]] is None else list(res[at[m.smiles]]) for m in feats]

    def _run(self, rows: list[tuple[int, int]], mols: Sequence[Molecule], adduct: str,
             energies: Sequence[float], instrument: str, top: int) -> list[Spectrum | None]:
        """One GPU batch; on CUDA out-of-memory it is split in halves, a single row that does not fit gives None."""
        try:
            uniq = sorted({i for i, _ in rows})
            pos = {i: p for p, i in enumerate(uniq)}
            b = _batch([mols[i] for i in uniq], [pos[i] for i, _ in rows], adduct, [energies[j] for _, j in rows],
                       instrument, self.device)
            dag = self.gen.generate(b)
            out, att = self.inten(b, dag)
            cpu = {k: v.cpu() for k, v in dag.items()}
            return _spectra(out.float().cpu(), att.float().cpu(), cpu, adduct, self.inten.inten_buckets.cpu(), top)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(rows) == 1:
                degraded.note("ICEBERG molecule too large for the GPU")
                return [None]
            h = len(rows) // 2
            return (self._run(rows[:h], mols, adduct, energies, instrument, top)
                    + self._run(rows[h:], mols, adduct, energies, instrument, top))

    @torch.inference_mode()
    def dag(self, smiles: str, adduct: str, energy: float, instrument: str = "QTOF") -> list[dict] | None:
        """The generated fragment DAG of one molecule (atom indices in ``featurise(smiles).smiles`` order,
        probability, broken bond orders, neutral mass), in upstream's node order."""
        m = featurise(smiles)
        if m is None or adduct not in SUPPORTED_ADDUCTS:
            return None
        d = self.gen.generate(_batch([m], [0], adduct, [energy], instrument if instrument in INSTRUMENTS
                                     else "Orbitrap", self.device))
        return [{"atoms": tuple(np.flatnonzero(d["frags"][0, f].cpu().numpy()).tolist()),
                 "prob": float(d["probs"][0, f]), "broken": float(d["brokens"][0, f]),
                 "mass": float(d["masses"][0, f])} for f in range(int(d["nfrags"][0]))]
