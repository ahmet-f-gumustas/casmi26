"""Build the candidate structure pool: COCONUT ∪ train structures → external/pool/.

Per unique structure: scorer key (RemoveStereochemistry → TautomerEnumerator.Canonicalize
→ InChIKey14, the official metric's chain), neutral-atom exact mass, formula, formal
charge and a Morgan radius-2 2048-bit fingerprint of the canonical tautomer. Structures
are deduplicated by key, preferring a neutral representative, then a train SMILES, then
the shortest SMILES. Keys of a random sample are re-checked with the official notebook
function. Output files are described in ``casmi.candidates``; a report goes to notes/POOL.md.

    python scripts/build_pool.py --coconut external/coconut/raw/coconut_csv-10-2026.csv
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from casmi.candidates import POOL_DIR
from casmi.chem import ELECTRON
from casmi.config import REPO_ROOT
from casmi.data import read_table
from casmi.metric import load_official_scorer

RDLogger.DisableLog("rdApp.*")
FP_BITS = 2048
_TAUT = None
_FPGEN = None
_OFFICIAL = None


def featurize(smiles: str) -> tuple | None:
    """(key, exact_mass, formula, formal_charge, packed_fp) or None if RDKit fails."""
    global _TAUT, _FPGEN
    if _TAUT is None:
        _TAUT = rdMolStandardize.TautomerEnumerator()
        _FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        charge = Chem.GetFormalCharge(mol)
        mass = Descriptors.ExactMolWt(mol) + charge * ELECTRON
        formula = rdMolDescriptors.CalcMolFormula(mol)
        Chem.RemoveStereochemistry(mol)
        canon = _TAUT.Canonicalize(mol)
        key = Chem.MolToInchiKey(canon)[:14]
        if not key:
            return None
        fp = np.packbits(_FPGEN.GetFingerprintAsNumPy(canon).astype(np.uint8))
        return key, mass, formula, charge, fp.tobytes()
    except (ValueError, RuntimeError, TypeError):
        return None


def official_key(smiles: str) -> str | None:
    return _OFFICIAL.canonicalize_smiles_to_ik14(smiles)


def mass_prefilter(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return np.nan if mol is None else Descriptors.ExactMolWt(mol) + Chem.GetFormalCharge(mol) * ELECTRON


def load_sources(coconut_csv: Path) -> pd.DataFrame:
    coco = pd.read_csv(coconut_csv, usecols=["identifier", "canonical_smiles"]).dropna()
    coco = coco.rename(columns={"canonical_smiles": "smiles", "identifier": "coconut_id"})
    coco["source"] = "coconut"
    train = pd.DataFrame({"smiles": read_table("train", ["normalized_smiles"]).column("normalized_smiles").unique()
                          .to_pylist()})
    train["source"], train["coconut_id"] = "train", None
    return pd.concat([train, coco], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coconut", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=POOL_DIR)
    ap.add_argument("--mass-range", type=float, nargs=2, default=(100.0, 1300.0),
                    help="keep structures in this neutral mass range (hidden test: 157–1159 Da)")
    ap.add_argument("--check", type=int, default=20_000, help="keys re-checked with the official function")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    src = load_sources(args.coconut)
    uniq = pd.Series(src["smiles"].unique())
    with Pool(args.workers) as pool:
        pre = np.array(pool.map(mass_prefilter, uniq.tolist(), chunksize=1000))
    lo, hi = args.mass_range
    in_range = uniq[(pre >= lo) & (pre <= hi)].tolist()
    print(f"{len(uniq):,} unique SMILES, {len(in_range):,} in {lo:g}–{hi:g} Da; featurizing…", flush=True)
    with Pool(args.workers) as pool:
        feats = pool.map(featurize, in_range, chunksize=200)
    ok = [(s, f) for s, f in zip(in_range, feats) if f is not None]
    rows = pd.DataFrame([(s, *f[:4]) for s, f in ok], columns=["smiles", "key", "exact_mass", "formula", "formal_charge"])
    fps = np.frombuffer(b"".join(f[4] for _, f in ok), dtype=np.uint8).reshape(len(ok), FP_BITS // 8)
    print(f"featurized {len(rows):,} ({len(in_range) - len(rows):,} failed) in {time.time() - t0:.0f}s", flush=True)

    train_smiles = set(src.loc[src["source"] == "train", "smiles"])
    coco_ids = src[src["source"] == "coconut"].drop_duplicates("smiles").set_index("smiles")["coconut_id"]
    rows["in_train"] = rows["smiles"].isin(train_smiles)
    rows["coconut_id"] = rows["smiles"].map(coco_ids)
    rows["in_coconut"] = rows["coconut_id"].notna()
    rows["fp_row"] = np.arange(len(rows))
    by_key = rows.groupby("key").agg(in_train=("in_train", "any"), in_coconut=("in_coconut", "any"),
                                     coconut_id=("coconut_id", "first"), n_smiles=("smiles", "size"),
                                     mass_max=("exact_mass", "max"), mass_min=("exact_mass", "min"))
    by_key["mass_spread"] = by_key.pop("mass_max") - by_key.pop("mass_min")
    rep = (rows.assign(charged=rows["formal_charge"] != 0, len=rows["smiles"].str.len())
           .sort_values(["key", "charged", "in_train", "len"], ascending=[True, True, False, True])
           .drop_duplicates("key"))
    meta = rep[["key", "smiles", "exact_mass", "formula", "formal_charge", "fp_row"]].join(by_key, on="key")
    meta = meta.sort_values(["exact_mass", "key"]).reset_index(drop=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "masses.npy", meta["exact_mass"].to_numpy(np.float64))
    np.save(args.out / "fps.npy", np.ascontiguousarray(fps[meta["fp_row"].to_numpy()]))
    meta = meta.drop(columns=["fp_row"])
    meta.to_parquet(args.out / "meta.parquet", index=False, compression="zstd")

    # Re-check keys with the official notebook function on a random sample.
    global _OFFICIAL
    _OFFICIAL = load_official_scorer()
    check_note = "official notebook not downloaded: key re-check skipped"
    if _OFFICIAL is not None and args.check:
        sample = meta.sample(n=min(args.check, len(meta)), random_state=args.seed)
        with Pool(args.workers) as pool:
            off = pool.map(official_key, sample["smiles"].tolist(), chunksize=200)
        agree = float(np.mean(np.asarray(off, dtype=object) == sample["key"].to_numpy()))
        check_note = f"key agreement with the official function on {len(sample):,} random pool structures: {agree:.4%}"

    sizes = {p.name: p.stat().st_size for p in sorted(args.out.iterdir()) if p.is_file()}
    total = sum(sizes.values())
    report = "\n\n".join([
        "# Candidate pool",
        f"- sources: {int((src['source'] == 'train').sum()):,} train SMILES, "
        f"{int((src['source'] == 'coconut').sum()):,} COCONUT rows ({args.coconut.name}); "
        f"{len(uniq):,} unique SMILES, {len(in_range):,} within {lo:g}–{hi:g} Da, {len(rows):,} featurized",
        f"- **{len(meta):,} structures after dedup by scorer key**: train {int(meta['in_train'].sum()):,}, "
        f"COCONUT {int(meta['in_coconut'].sum()):,}, both {int((meta['in_train'] & meta['in_coconut']).sum()):,}",
        f"- keys with several SMILES: {int((meta['n_smiles'] > 1).sum()):,}; of those with a mass spread "
        f"> 0.5 Da (protomers / charged forms): {int((meta['mass_spread'] > 0.5).sum()):,} — representative is neutral",
        f"- {check_note}",
        "- files: " + ", ".join(f"`{k}` {v / 1e6:.1f} MB" for k, v in sizes.items())
        + f"; total {total / 1e6:.0f} MB (fits a Kaggle dataset comfortably)",
        f"- build time: {time.time() - t0:.0f}s",
    ]) + "\n"
    note = REPO_ROOT / "notes" / "POOL.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
