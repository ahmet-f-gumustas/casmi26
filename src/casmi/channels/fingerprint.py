"""Fingerprint channel: a spectrum → fingerprint model ranks candidates by f·z.

The model turns a molecule's spectra into per-bit logits ``z``; a candidate with binary
fingerprint ``f`` scores ``f·z``, its log-likelihood under independent bits up to a
candidate-independent constant (CSI:FingerID-style). Unlike the analog channel it needs
no library neighbour, only the candidate's structure.

Descriptor layouts (the bit order a model was trained on):

- ``ap6930``: Morgan r2 4096 + Morgan r3 4096 + RDKit path 2048 (maxPath 6) + MACCS 167,
  columns selected by ``fp_bits.npy`` — the layout of prvsiyan's FPNet checkpoints;
- ``casmi6311``: Morgan r2 4096 + MACCS 167 + FCFP r2 2048 (feature invariants) — our model.

Scores are shifted per molecule to be positive (min → 1e-6) so the rank-normalising ranker
treats every candidate as having evidence; the order is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from multiprocessing import Pool

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

LAYOUTS = {"ap6930": 10407, "casmi6311": 6311}
_GEN: dict = {}


def _generators() -> dict:
    if not _GEN:
        _GEN["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        _GEN["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        _GEN["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
        _GEN["fc"] = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    return _GEN


def descriptor(smiles: str, layout: str, bits: NDArray[np.int64] | None = None) -> NDArray[np.uint8] | None:
    """0/1 fingerprint of ``smiles`` in ``layout`` (``bits`` selects ap6930 columns); None if unparsable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    g = _generators()
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)
    if layout == "ap6930":
        full = np.concatenate([g["m2"].GetFingerprintAsNumPy(mol).astype(np.uint8),
                               g["m3"].GetFingerprintAsNumPy(mol).astype(np.uint8),
                               g["rk"].GetFingerprintAsNumPy(mol).astype(np.uint8), maccs])
        return full if bits is None else full[bits]
    if layout == "casmi6311":
        return np.concatenate([g["m2"].GetFingerprintAsNumPy(mol).astype(np.uint8), maccs,
                               g["fc"].GetFingerprintAsNumPy(mol).astype(np.uint8)])
    raise ValueError(f"unknown fingerprint layout {layout!r}")


_STATE: dict = {}


def _packed(smiles: str) -> NDArray[np.uint8] | None:
    fp = descriptor(smiles, _STATE["layout"], _STATE["bits"])
    return None if fp is None else np.packbits(fp)


def descriptors(smiles: Sequence[str], layout: str, bits: NDArray[np.int64] | None = None,
                workers: int = 1) -> list[NDArray[np.uint8] | None]:
    """Packed fingerprints for many SMILES (None where RDKit fails)."""
    _STATE.update(layout=layout, bits=bits)
    if workers > 1 and len(smiles) > 1000:
        with Pool(workers) as p:
            out = p.map(_packed, smiles, chunksize=200)
    else:
        out = [_packed(s) for s in smiles]
    _STATE.clear()
    return out


def score(z: NDArray[np.float32], packed: Sequence[NDArray[np.uint8] | None], nbits: int,
          how: str = "dot") -> NDArray[np.float64]:
    """Candidate scores, positive; unparsable candidates score 0.

    ``dot``: f·z shifted so the lowest is 1e-6. ``soft_tanimoto``: Tanimoto between f and the
    predicted bit probabilities sigmoid(z), the similarity the ranking loss is trained on.
    """
    ok = [i for i, f in enumerate(packed) if f is not None]
    out = np.zeros(len(packed))
    if not ok:
        return out
    f = np.unpackbits(np.stack([packed[i] for i in ok]), axis=1, count=nbits).astype(np.float32)
    z = np.asarray(z, dtype=np.float32)
    if how == "dot":
        raw = f @ z
        out[ok] = raw - raw.min() + 1e-6
    elif how == "soft_tanimoto":
        p = 1.0 / (1.0 + np.exp(-z))
        inter = f @ p
        out[ok] = inter / np.maximum(f.sum(axis=1) + p.sum() - inter, 1e-6) + 1e-6
    else:
        raise ValueError(f"unknown fingerprint score {how!r}")
    return out


def fingerprint_channel(logits: Mapping[str, NDArray[np.float32]], windows: Mapping[str, list[str]],
                        smiles_of: Mapping[str, str], layout: str, bits: NDArray[np.int64] | None = None,
                        workers: int = 1, cache: dict | None = None, how: str = "dot") -> dict[str, dict[str, float]]:
    """Channel scores per molecule from its logits ``z``; candidate fingerprints are cached by key."""
    cache = {} if cache is None else cache
    nbits = len(bits) if bits is not None else LAYOUTS[layout]
    todo = sorted({k for m, ks in windows.items() if m in logits for k in ks} - set(cache))
    cache.update(zip(todo, descriptors([smiles_of[k] for k in todo], layout, bits, workers)))
    out = {}
    for m, z in logits.items():
        keys = windows.get(m, [])
        if keys:
            out[m] = dict(zip(keys, score(z, [cache[k] for k in keys], nbits, how).tolist()))
    return out
