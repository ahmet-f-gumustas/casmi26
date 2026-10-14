"""Local re-implementation of the competition metric: MRR@25 per molecule.

A prediction is correct when its tautomer-canonical InChIKey14 (first block of
the InChIKey, i.e. connectivity only — stereo and tautomers do not matter)
matches the ground truth. Keys must be computed with the pinned RDKit version
(see ``casmi.config.RDKIT_VERSION``).

NOTE: once the official metric notebook is downloaded, compare its
canonicalization call against ``scorer_key`` and log any difference as EXP-000.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from casmi.config import TOP_K

RDLogger.DisableLog("rdApp.*")

_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
HITS_AT = (1, 5, 10, 25)


@lru_cache(maxsize=1_000_000)
def scorer_key(smiles: str) -> str | None:
    """Tautomer-canonical InChIKey14 for ``smiles``, or ``None`` if unparseable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
    except Exception:  # noqa: BLE001 — scorer falls back to the raw molecule
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def reciprocal_rank(
    pred_keys: Sequence[str | None], true_key: str | None, k: int = TOP_K
) -> float:
    """1/rank of the first match within the top ``k`` raw slots, else 0.

    Slot positions are taken as-is: duplicates and invalid entries (``None``)
    still occupy a slot, exactly like in the scorer.
    """
    if true_key is None:
        return 0.0
    for i, key in enumerate(pred_keys[:k]):
        if key is not None and key == true_key:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    predictions: Mapping[Hashable, Sequence[str]],
    truth: Mapping[Hashable, str],
    k: int = TOP_K,
    classes: Mapping[Hashable, int] | None = None,
) -> dict[str, float]:
    """Score ranked SMILES lists against true SMILES, one entry per molecule.

    Molecules present in ``truth`` but missing from ``predictions`` count as
    misses. Returns ``mrr@k``, ``hits@n``, ``median_rank`` (over found
    molecules only, NaN if none), ``missing_frac`` (share not found in top
    ``k``) and, when ``classes`` is given, ``mrr_class{c}`` per novelty class.
    """
    ids = list(truth)
    ranks = np.full(len(ids), np.inf)
    for i, mol_id in enumerate(ids):
        pred_keys = [scorer_key(s) for s in predictions.get(mol_id, [])[:k]]
        rr = reciprocal_rank(pred_keys, scorer_key(truth[mol_id]), k=k)
        if rr > 0:
            ranks[i] = 1.0 / rr

    rr_all = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    found = np.isfinite(ranks)
    out: dict[str, float] = {
        "n": float(len(ids)),
        f"mrr@{k}": float(rr_all.mean()) if len(ids) else 0.0,
    }
    for n in HITS_AT:
        if n <= k:
            out[f"hits@{n}"] = float((ranks <= n).mean()) if len(ids) else 0.0
    out["median_rank"] = float(np.median(ranks[found])) if found.any() else float("nan")
    out["missing_frac"] = float((~found).mean()) if len(ids) else 0.0

    if classes is not None:
        cls = np.array([classes.get(mol_id) for mol_id in ids], dtype=object)
        for c in sorted({c for c in cls if c is not None}):
            mask = cls == c
            out[f"mrr_class{c}"] = float(rr_all[mask].mean())
    return out


def dedup_by_key(smiles: Sequence[str], k: int = TOP_K) -> list[str]:
    """Keep the first SMILES per scorer key, drop unparseable ones, cut at ``k``."""
    seen: set[str] = set()
    out: list[str] = []
    for s in smiles:
        key = scorer_key(s)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) == k:
            break
    return out
