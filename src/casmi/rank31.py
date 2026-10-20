"""Ranker on the 31 candidate features of prvsiyan's V33 notebook, fed by our channels.

Ported from "Analog Propagation CASMI 2026 baseline" (kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline,
Apache License 2.0; see NOTICE): ``rank_features`` (the feature definitions), the MetFrag-lite
fragmentation score (``fragment_masses`` / ``explain_score``, sqrt intensities, max over spectra) and the
ranker recipe (HistGradientBoostingClassifier, Class 1 simulation weighted ``w1`` and Class 2 ``1 − w1``,
averaged over ``PRIORS`` × ``SEEDS``). Training rows come either from their ``rank_train.npz`` (X, Y, M, G:
features, answer flag, 0 = Class 1 / 1 = Class 2 simulation, query group) or from our simulated splits in
the same format.

What feeds the features here (theirs in brackets):

- ``lv``: our library channel, best entropy similarity of the candidate's library spectra over the
  molecule's spectra [their entropy search within ±10 ppm];
- analogs: our flash-entropy analog hits (±200 Da, hybrid search), best similarity per library
  structure over the molecule's spectra, the top ``N_ANALOG`` structures [one representative spectrum per
  structure, same instrument as the test];
- Tanimoto between candidates and analogs on the full ``ap6930`` descriptor (Morgan r2 4096 + r3 4096 +
  RDKit path 2048 + MACCS, no column selection) [the same descriptor through their ``fp_bits.npy`` columns];
- model features: our fingerprint model's ``f·z`` (unshifted) with its own layout [their FPNet];
- ``frag``: the ported MetFrag-lite score.

``m_minus_max`` (f·z minus its window maximum) is on the logit scale of the model that made the rows,
so it is dropped (``DROP``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem, RDLogger

from casmi import degraded
from casmi.channels.analog import tanimoto
from casmi.channels.fingerprint import LAYOUTS, descriptors

RDLogger.DisableLog("rdApp.*")

FEATURES = ["lv", "lv_rank", "lv_max", "lv_minus_max", "lv_pos", "ap", "ap_rank", "ap_max", "ap_minus_max",
            "a1", "best_tan", "top_tan", "mean_tan", "top_sim", "log_nc",
            "m_z", "m_rank", "m_minus_max", "mn_z", "mn_rank", "m_is_max",
            "f", "f_rank", "f_minus_max", "f_z",
            "x_lv_mr", "x_ap_mr", "x_agree", "x_agree_a", "x_agree_lvmax", "x_corr"]
DROP = ("m_minus_max",)
SOURCE = "src_pubchem"  # 1 for a PubChem-only candidate joined by casmi.pipeline.pubchem_join
USED = [f for f in FEATURES if f not in DROP]
N_ANALOG = 80
P_SIM = 3.0
ANALOG_TOP_K = 400  # analog hits per spectrum (library rows), enough for N_ANALOG structures
PRIORS = (0.30, 0.60)
SEEDS = (0, 1, 2, 3)
GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)
TAN_LAYOUT = "ap6930"

# --- feature definitions (ported) ---------------------------------------------------------------


def _rank_norm(x: NDArray) -> NDArray[np.float64]:
    o = np.argsort(-x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def _z(x: NDArray) -> NDArray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def rank_features(lv: NDArray, tan: NDArray | None, analog_sim: NDArray, raw: NDArray | None,
                  cand_bits: NDArray | None, frag: NDArray | None) -> NDArray[np.float32]:
    """The 31 features (``FEATURES`` order) for one molecule's ``nc`` candidates.

    ``lv`` (nc,) library similarity (0 if none); ``tan`` (nc, na) Tanimoto to the analogs, whose
    similarities ``analog_sim`` (na,) are descending; ``raw`` (nc,) f·z and ``cand_bits`` (nc,) the
    candidates' bit counts in the model's layout; ``frag`` (nc,) the MetFrag-lite score.
    """
    lv = np.asarray(lv, np.float32)
    nc = len(lv)
    lvmax = float(lv.max()) if nc else 0.0
    if tan is not None and len(analog_sim):
        tan = np.asarray(tan, np.float32)
        w = np.clip(np.asarray(analog_sim, np.float32), 0, None)
        wp = w ** P_SIM
        ap = (tan * wp[None, :]).max(1)
        a1 = (tan * w[None, :]).max(1)
        best_tan, top_tan, top_sim = tan.max(1), tan[:, 0], float(w[0])
        mean_tan = (tan * wp[None, :]).sum(1) / (wp.sum() + 1e-9)
    else:
        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32)
        top_sim = 0.0
    apmax = float(ap.max()) if nc else 0.0
    if raw is not None:
        raw = np.asarray(raw, np.float32)
        nrm = raw / np.sqrt(np.maximum(np.asarray(cand_bits, np.float32), 1.0))
        mfeat = [_z(raw), _rank_norm(raw), raw - raw.max(), _z(nrm), _rank_norm(nrm),
                 (raw == raw.max()).astype(np.float32)]
        mr = _rank_norm(raw)
        lbest = int(np.argmax(lv)) if lvmax > 0 else -1
        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0
        abest = int(np.argmax(ap)) if apmax > 0 else -1
        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0
        corr = float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0
        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),
                 np.full(nc, agree * lvmax), np.full(nc, corr)]
    else:
        mfeat = [np.zeros(nc, np.float32)] * 6
        xfeat = [np.zeros(nc, np.float32)] * 6
    if frag is not None:
        fr = np.asarray(frag, np.float32)
        ffeat = [fr, _rank_norm(fr), fr - fr.max() if nc else fr, _z(fr)]
    else:
        ffeat = [np.zeros(nc, np.float32)] * 4
    return np.column_stack([
        lv, _rank_norm(lv), np.full(nc, lvmax), lv - lvmax, (lv > 0).astype(np.float32),
        ap, _rank_norm(ap), np.full(nc, apmax), ap - apmax,
        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim),
        np.full(nc, np.log(max(nc, 1))),
        *mfeat, *ffeat, *xfeat,
    ]).astype(np.float32)


# --- MetFrag-lite (ported) ----------------------------------------------------------------------

_AMU = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.9949146196, "P": 30.97376163,
        "S": 31.97207100, "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473,
        "Na": 22.9897692809, "K": 38.96370668, "Si": 27.9769265325, "B": 11.0093054, "Se": 79.9165213}
_H = _AMU["H"]
_PROTON = _H - 0.00054857990


def _components(n: int, bonds: Sequence[tuple[int, int]], drop: set[int]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i not in drop:
            adj[a].append(b)
            adj[b].append(a)
    seen = np.zeros(n, bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, cur = [s], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    cur.append(v)
        comps.append(cur)
    return comps


def fragment_masses(smiles: str, max_breaks: int = 2, max_bonds: int = 34) -> NDArray[np.float64]:
    """Neutral fragment masses from breaking 1 or 2 bonds (only the molecule above ``max_bonds``)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.zeros(0)
    n = mol.GetNumAtoms()
    w = np.array([_AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * _H for a in mol.GetAtoms()])
    if not n or (w == 0).any():
        return np.zeros(0)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    nb = len(bonds)
    if nb == 0 or nb > max_bonds:
        return np.array([w.sum()])
    out = {float(w.sum())}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))


def explain_score(frag_mass: NDArray, peak_mz: NDArray, peak_int: NDArray, mode: float = 1.0, tol: float = 0.01,
                  h_shifts: Sequence[int] = (-2, -1, 0, 1, 2)) -> float:
    """Fraction of total sqrt intensity explained by some fragment ion."""
    if len(frag_mass) == 0 or len(peak_mz) == 0:
        return 0.0
    ion = np.sort(np.concatenate([frag_mass + dh * _H + (_PROTON if mode > 0 else -_PROTON) for dh in h_shifts]))
    w = np.sqrt(np.asarray(peak_int, float))
    tot = w.sum()
    if tot <= 0:
        return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)


def clean_peaks(mz: NDArray, it: NDArray, floor: float = 0.002, topk: int = 256) -> tuple[NDArray, NDArray]:
    """Peaks at or above ``floor`` × base peak, the ``topk`` most intense, intensities summing to 1."""
    mz, it = np.asarray(mz, np.float32), np.asarray(it, np.float32)
    if not len(mz) or it.max() <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = np.flatnonzero(it >= floor * it.max())
    if len(keep) > topk:
        keep = np.sort(keep[np.argsort(it[keep], kind="stable")[len(keep) - topk:]])
    m, i = mz[keep], it[keep]
    s = i.sum()
    return m, (i / s if s > 0 else i)


# --- our channels → features ----------------------------------------------------------------------


def analog_list(spectrum_hits: Sequence[tuple[NDArray, NDArray]], row_key: Mapping[int, str],
                n: int = N_ANALOG) -> tuple[list[str], NDArray[np.float32]]:
    """Library structures among a molecule's analog hits, best similarity each, top ``n`` (descending)."""
    best: dict[str, float] = {}
    for rows, sims in spectrum_hits:
        for r, s in zip(rows, sims):
            k = row_key.get(int(r))
            if k is not None and s > best.get(k, -1.0):
                best[k] = float(s)
    top = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [k for k, _ in top], np.array([s for _, s in top], np.float32)


def _pad8(packed: NDArray[np.uint8]) -> NDArray[np.uint8]:
    extra = (-packed.shape[1]) % 8
    return np.pad(packed, ((0, 0), (0, extra))) if extra else packed


_FSTATE: dict = {}


def _frag(smiles: str) -> NDArray[np.float64]:
    try:
        return fragment_masses(smiles)
    except Exception:  # noqa: BLE001 - the port scores an unparsable candidate as unexplained
        return np.zeros(0)


def frame(queries: pd.DataFrame, windows: Mapping[str, list[str]], smiles_of: Mapping[str, str],
          hits: Sequence[tuple[NDArray, NDArray]], row_key: Mapping[int, str],
          library_max: Mapping[str, Mapping[str, float]], logits: Mapping[str, NDArray[np.float32]] | None,
          layout: str | None, bits: NDArray[np.int64] | None = None, workers: int = 1,
          caches: dict | None = None) -> pd.DataFrame:
    """One row per (molecule, candidate) with the 31 ``FEATURES``.

    ``queries`` must be in the row order of ``hits`` (``casmi.pipeline.analog_hits``); ``library_max``
    is the library channel's best match; ``logits`` per molecule with ``layout`` / ``bits`` the model's
    descriptor (None: no model features). ``caches`` keeps descriptors / fragments across calls.
    """
    caches = {} if caches is None else caches
    tan_fp, model_fp, frags = (caches.setdefault(c, {}) for c in ("tan_fp", "model_fp", "frag"))
    idx_of = queries.groupby("molecule_id").indices
    analogs = {m: analog_list([hits[i] for i in idx], row_key) for m, idx in idx_of.items()}
    cand = sorted({k for m, ks in windows.items() if m in idx_of for k in ks})
    need_tan = sorted((set(cand) | {k for ks, _ in analogs.values() for k in ks}) - set(tan_fp))
    tan_fp.update(zip(need_tan, descriptors([smiles_of.get(k, "") for k in need_tan], TAN_LAYOUT, None, workers)))
    if logits is not None:
        need_m = sorted(set(cand) - set(model_fp))
        model_fp.update(zip(need_m, descriptors([smiles_of.get(k, "") for k in need_m], layout, bits, workers)))
    need_f = sorted(set(cand) - set(frags))
    if need_f:
        smi = [smiles_of.get(k, "") for k in need_f]
        if workers > 1:
            with Pool(workers) as p:
                out = p.map(_frag, smi, chunksize=16)
        else:
            out = [_frag(s) for s in smi]
        frags.update(zip(need_f, out))
    degraded.note("rank31 candidate without a descriptor (zero Tanimoto)", sum(tan_fp.get(k) is None for k in cand))
    if logits is not None:
        degraded.note("rank31 molecule without model features", sum(1 for m in windows if m in idx_of and windows[m]
                                                                      and m not in logits))
    n_tan = LAYOUTS[TAN_LAYOUT]
    empty_tan = np.zeros(-(-n_tan // 8), np.uint8)
    parts = []
    for m, keys in windows.items():
        if m not in idx_of or not keys:
            continue
        grp = queries.iloc[idx_of[m]]
        lib = library_max.get(m, {})
        lv = np.array([lib.get(k, 0.0) for k in keys], np.float32)
        a_keys, a_sim = analogs[m]
        a_keys_ok = [(k, s) for k, s in zip(a_keys, a_sim) if tan_fp.get(k) is not None]
        tan = None
        sims = np.zeros(0, np.float32)
        if a_keys_ok:
            cfp = _pad8(np.stack([tan_fp[k] if tan_fp.get(k) is not None else empty_tan for k in keys]))
            afp = _pad8(np.stack([tan_fp[k] for k, _ in a_keys_ok]))
            tan = tanimoto(cfp, afp)
            sims = np.array([s for _, s in a_keys_ok], np.float32)
        raw = bits = None
        if logits is not None and m in logits:
            nb = len(bits) if bits is not None else LAYOUTS[layout]
            fz = np.zeros((len(keys), nb), np.float32)
            for i, k in enumerate(keys):
                f = model_fp.get(k)
                if f is not None:
                    fz[i] = np.unpackbits(f, count=nb)
            raw, bits = fz @ np.asarray(logits[m], np.float32), fz.sum(1)
        mode = float(np.mean([1.0 if x == "positive" else -1.0 for x in grp["ionization_mode"]]))
        peaks = [clean_peaks(mz, it) for mz, it in zip(grp["ms2_mzs"], grp["ms2_normalized_intensities"])]
        frag = np.array([max((explain_score(frags[k], a, b, mode=mode) for a, b in peaks), default=0.0)
                         for k in keys], np.float32)
        X = rank_features(lv, tan, sims, raw, bits, frag)
        parts.append(pd.DataFrame(X, columns=FEATURES).assign(molecule_id=m, key=keys))
    if not parts:
        return pd.DataFrame(columns=["molecule_id", "key", *FEATURES])
    return pd.concat(parts, ignore_index=True)[["molecule_id", "key", *FEATURES]]


# --- ranker -------------------------------------------------------------------------------------


def fit(X: NDArray, Y: NDArray, M: NDArray, priors: Sequence[float] = PRIORS, seeds: Sequence[int] = SEEDS,
        params: Mapping | None = None) -> list:
    """One classifier per (Class 1 weight, seed); Class 1 rows (M = 0) weigh ``w1``, Class 2 rows ``1 − w1``.

    The public rows hold every query once per class, so ``w1`` is the Class 1 share of the weight. Rows
    with another class mix are rescaled to the same meaning (per-row weight × N / (2 · rows of its
    class)); with equal halves, as in ``rank_train.npz``, the weights are exactly the original ones.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    M = np.asarray(M)
    n, n1 = len(M), int((M == 0).sum())
    scale0, scale2 = n / (2 * max(n1, 1)), n / (2 * max(n - n1, 1))
    models = []
    for w1 in priors:
        w = np.where(M == 0, w1 * scale0, (1.0 - w1) * scale2)
        for sd in seeds:
            models.append(HistGradientBoostingClassifier(random_state=sd, **(params or GBM)).fit(X, Y, sample_weight=w))
    return models


def proba(models: Sequence, X: NDArray) -> NDArray[np.float64]:
    """Mean answer probability over the models."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def columns(names: Sequence[str] = USED) -> list[int]:
    """Column indices of ``names`` in the full ``FEATURES`` matrix (``rank_train.npz`` order)."""
    return [FEATURES.index(n) for n in names]


# --- popularity prior (V33 stage 2) ---------------------------------------------------------------

POP_TOP = 60
LIB_GATE = 0.90
POP_FEATURES = ["pop_sid", "pop_pmid"]  # log1p(PubChem substances), log1p(PubMed records)


def popularity_columns(keys: Sequence[str], table: pd.DataFrame) -> NDArray[np.float32]:
    """``POP_FEATURES`` per key from a popularity table (key, pc_sid, pc_pmid); 0 where PubChem lacks it."""
    t = table.set_index("key")
    sid = t["pc_sid"].reindex(keys).fillna(0.0).to_numpy(np.float64)
    pm = t["pc_pmid"].reindex(keys).fillna(0.0).to_numpy(np.float64)
    return np.column_stack([np.log1p(sid), np.log1p(pm)]).astype(np.float32)


def popularity_scores(molecule_id: Sequence[str], p: NDArray, pop: NDArray, lib_max: NDArray, mu: float,
                      top: int = POP_TOP, gate: float = LIB_GATE) -> NDArray[np.float64]:
    """Scores whose per-molecule order is V33's popularity re-order.

    For a molecule whose best library match is below ``gate``, its ``top`` candidates by ``p`` are
    re-sorted by z(p over them) + ``mu`` · pop (pop = log1p substances + log1p PubMed) and stay above
    the rest, which keep their order; other molecules are unchanged.
    """
    p = np.asarray(p, np.float64)
    out = p.copy()
    if mu == 0:
        return out
    pop, lib_max = np.asarray(pop, np.float64), np.asarray(lib_max, np.float64)
    for idx in pd.Series(range(len(p))).groupby(np.asarray(molecule_id)).indices.values():
        if lib_max[idx].max() >= gate:
            continue
        head = idx[np.argsort(-p[idx], kind="stable")[:top]]
        s = p[head]
        sd = s.std()
        z = (s - s.mean()) / sd if sd > 1e-12 else np.zeros(len(head))
        out[head] = 100.0 + z + mu * pop[head]
    return out
