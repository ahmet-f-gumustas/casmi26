"""Local mirror of the official competition metric: MRR@25 per molecule.

Mirrors Kaggle's ``metric/casmi-mean-reciprocal-rank`` notebook (compared in EXP-000):

- key: ``RemoveStereochemistry`` → ``TautomerEnumerator().Canonicalize`` →
  ``InChIKey[:14]`` under RDKit 2026.03.3; any RDKit failure is a non-match.
  Stereo is stripped *first* because a stereocentre can pin a double bond and
  change the canonical tautomer, i.e. the InChIKey14.
- guesses are split on ``;``, whitespace-stripped and empty ones dropped
  *before* ranking; invalid and duplicate SMILES keep their slot.
- an answer may list several acceptable SMILES separated by ``,``.
- more than ``k`` guesses for a molecule is an error, as on Kaggle.

The official scorer also skips canonicalization when a guess's heavy-atom
composition differs from the answer's. That is only a speed gate (a shared
InChIKey14 implies the same heavy atoms) and is not reproduced here.
"""

from __future__ import annotations

import importlib.util
import json
import types
from collections.abc import Hashable, Mapping, Sequence, Set
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from casmi.config import EXTERNAL_DIR, TOP_K

RDLogger.DisableLog("rdApp.*")

_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
HITS_AT = (1, 5, 10, 25)
OFFICIAL_NOTEBOOK = EXTERNAL_DIR / "official_metric" / "casmi-mean-reciprocal-rank.ipynb"


@lru_cache(maxsize=1_000_000)
def scorer_key(smiles: str) -> str | None:
    """Tautomer-canonical, stereo-free InChIKey14 for ``smiles``, or ``None``."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        key = Chem.MolToInchiKey(_TAUTOMER_ENUMERATOR.Canonicalize(mol))
    except (ValueError, RuntimeError, TypeError):
        return None
    return key[:14] or None


def split_guesses(raw: str | Sequence[str]) -> list[str]:
    """Guesses exactly as the scorer ranks them: ``;``-split, stripped, empties dropped."""
    joined = raw if isinstance(raw, str) else ";".join(map(str, raw))
    return [g.strip() for g in joined.split(";") if g.strip()]


def answer_keys(answer: str) -> set[str]:
    """Accepted keys for a ``,``-separated answer field; raises if none parse."""
    keys = {k for k in map(scorer_key, (s for s in str(answer).split(",") if s)) if k}
    if not keys:
        raise ValueError(f"answer SMILES could not be parsed: {answer!r}")
    return keys


def reciprocal_rank(pred_keys: Sequence[str | None], accept: str | Set[str] | None, k: int = TOP_K) -> float:
    """1/rank of the first key in ``accept`` within the top ``k`` slots, else 0.

    Slot positions are taken as-is: duplicates and invalid entries (``None``)
    still occupy a slot, exactly like in the scorer.
    """
    if accept is None:
        return 0.0
    accept = {accept} if isinstance(accept, str) else accept
    for i, key in enumerate(pred_keys[:k]):
        if key is not None and key in accept:
            return 1.0 / (i + 1)
    return 0.0


def _first_hit(guesses: Sequence[str], accept: Set[str]) -> int | None:
    for rank, guess in enumerate(guesses, 1):
        if scorer_key(guess) in accept:
            return rank
    return None


def evaluate(
    predictions: Mapping[Hashable, str | Sequence[str]],
    truth: Mapping[Hashable, str],
    k: int = TOP_K,
    classes: Mapping[Hashable, int] | None = None,
) -> dict[str, float]:
    """Score ranked guesses against answers, one entry per molecule.

    ``predictions`` values are a list of SMILES or a ``;``-joined string.
    Molecules in ``truth`` but missing from ``predictions`` score 0. Returns
    ``mrr@k``, ``hits@n``, ``median_rank`` (over found molecules only, NaN if
    none), ``missing_frac`` (share not found) and, when ``classes`` is given,
    ``mrr_class{c}`` per novelty class.
    """
    ids = list(truth)
    ranks = np.full(len(ids), np.inf)
    for i, mol_id in enumerate(ids):
        guesses = split_guesses(predictions.get(mol_id, []))
        if len(guesses) > k:
            raise ValueError(f"At most {k} guesses per molecule, got {len(guesses)} for {mol_id}")
        hit = _first_hit(guesses, answer_keys(truth[mol_id]))
        if hit is not None:
            ranks[i] = hit

    found = np.isfinite(ranks)
    rr = np.where(found, 1.0 / ranks, 0.0)
    n = len(ids)
    out: dict[str, float] = {"n": float(n), f"mrr@{k}": float(rr.mean()) if n else 0.0}
    for h in HITS_AT:
        if h <= k:
            out[f"hits@{h}"] = float((ranks <= h).mean()) if n else 0.0
    out["median_rank"] = float(np.median(ranks[found])) if found.any() else float("nan")
    out["missing_frac"] = float((~found).mean()) if n else 0.0

    if classes is not None:
        cls = np.array([classes.get(mol_id) for mol_id in ids], dtype=object)
        for c in sorted({c for c in cls if c is not None}):
            out[f"mrr_class{c}"] = float(rr[cls == c].mean())
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


def load_official_scorer(path: Path = OFFICIAL_NOTEBOOK) -> types.ModuleType | None:
    """The official metric notebook as a module, or ``None`` if not downloaded.

    Fetch it with ``kaggle kernels pull metric/casmi-mean-reciprocal-rank -p external/official_metric``.
    """
    if not path.exists():
        return None
    cells = json.loads(path.read_text())["cells"]
    source = "\n\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("casmi_official_metric", loader=None))
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 — trusted Kaggle metric code
    module.install_dependencies()
    return module
