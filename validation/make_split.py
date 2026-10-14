"""Build the molecule-level validation split (see ``casmi.split`` for the class contract).

Writes to ``WORK_DIR/splits/seed{seed}/``:
- ``val_ids.parquet``: molecule_id, inchikey14 (scorer key), class, smiles, source, exact_mass, n_query
- ``library_mask.parquet``: one row per train spectrum (train order): row, in_library, query_molecule_id
- ``config.json``
and a summary to ``notes/SPLIT_seed{seed}.md``. Scorer keys and masses per unique
SMILES are cached in ``WORK_DIR/structure_keys.parquet``.

    python validation/make_split.py [--n-val 2000] [--seed 0] [--no-np-examples]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from casmi.chem import exact_mass, parse_adduct, ppm_error, precursor_mz
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.data import read_table
from casmi.metric import scorer_key
from casmi.report import md_table
from casmi.split import SplitConfig, candidate_pool_keys, ce_pattern, check_leakage, make_split


def _keys_and_mass(smiles: str) -> tuple[str | None, float | None]:
    return scorer_key(smiles), exact_mass(smiles)


def structure_keys(smiles: pd.Series, workers: int) -> pd.DataFrame:
    """Scorer key and neutral mass per unique SMILES, cached under WORK_DIR."""
    cache = WORK_DIR / "structure_keys.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    uniq = sorted(smiles.dropna().unique().tolist())
    with Pool(workers) as pool:
        res = pool.map(_keys_and_mass, uniq, chunksize=200)
    out = pd.DataFrame({"normalized_smiles": uniq, "key": [k for k, _ in res],
                        "exact_mass": np.array([m for _, m in res], dtype=np.float64)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def load_rows(workers: int) -> pd.DataFrame:
    cols = ["ingest_lib", "normalized_smiles", "adduct", "precursor_mz", "base_peak_intensity", "collision_energy_ev"]
    t = read_table("train", cols)
    rows = t.drop(["collision_energy_ev"]).to_pandas()
    rows["n_ce"] = pc.list_value_length(t.column("collision_energy_ev")).to_numpy(zero_copy_only=False)
    keys = structure_keys(rows["normalized_smiles"], workers).set_index("normalized_smiles")
    rows = rows.join(keys, on="normalized_smiles")
    rows["ppm"] = np.nan
    for adduct, idx in rows.groupby("adduct").groups.items():
        if parse_adduct(adduct) is None:
            continue
        sub = rows.loc[idx]
        rows.loc[idx, "ppm"] = ppm_error(sub["precursor_mz"].to_numpy(), precursor_mz(sub["exact_mass"].to_numpy(), adduct))
    return rows


def load_test() -> pd.DataFrame:
    t = read_table("test", ["molecule_id", "adduct", "collision_energy_ev"])
    test = t.drop(["collision_energy_ev"]).to_pandas()
    test["n_ce"] = pc.list_value_length(t.column("collision_energy_ev")).to_numpy(zero_copy_only=False)
    return test


def summary(rows: pd.DataFrame, test: pd.DataFrame, split, cfg: SplitConfig) -> str:
    val, mask = split.val, split.library_mask
    q = rows[mask["query_molecule_id"].notna().to_numpy()].assign(molecule_id=mask["query_molecule_id"].dropna().to_numpy())

    def shares(a: pd.Series, b: pd.Series, name: str) -> pd.DataFrame:
        df = pd.concat([a.value_counts(normalize=True).rename("val queries"),
                        b.value_counts(normalize=True).rename("visible test")], axis=1).fillna(0.0)
        return df.sort_values("visible test", ascending=False).reset_index(names=name)

    spm = pd.DataFrame({
        "val": q.groupby("molecule_id").size().describe(percentiles=[0.5, 0.9]),
        "visible test": test.groupby("molecule_id").size().describe(percentiles=[0.5, 0.9]),
    }).T.reset_index(names="spectra/mol")
    classes = val.groupby(["class", "source"]).size().unstack(fill_value=0).reset_index()
    removed = (~mask["in_library"]).sum()
    return "\n\n".join([
        f"# Validation split (seed {cfg.seed})",
        f"- config: `{json.dumps(dataclasses.asdict(cfg))}`",
        f"- {len(val):,} molecules, {len(q):,} query spectra; {removed:,} train spectra "
        f"({removed / len(mask):.2%}) leave the library; C3 removes {int((val['class'] == 3).sum())} structures "
        "from the candidate pool",
        "### Classes × source", md_table(classes),
        "### Adduct share", md_table(shares(q["adduct"], test["adduct"], "adduct")),
        "### CE pattern share", md_table(shares(ce_pattern(q["n_ce"]), ce_pattern(test["n_ce"]), "ce_pattern")),
        "### Spectra per molecule", md_table(spm),
        "### Neutral mass", md_table(val["exact_mass"].describe(percentiles=[0.05, 0.5, 0.95]).to_frame().T),
        "> Leakage checks passed: no query or C2/C3 spectrum in the library, C1 keeps only non-timsTOF "
        "references, C3 structures not in the candidate pool, molecules grouped by scorer key.",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shares", type=float, nargs=3, default=(0.16, 0.45, 0.39), metavar=("C1", "C2", "C3"))
    ap.add_argument("--no-np-examples", action="store_true", help="do not force enveda-np-examples into val")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = SplitConfig(n_val=args.n_val, seed=args.seed, class_shares=tuple(args.shares),
                      include_np_examples=not args.no_np_examples)
    out_dir = args.out_dir or WORK_DIR / "splits" / f"seed{cfg.seed}"

    rows = load_rows(args.workers)
    test = load_test()
    split = make_split(rows, test, cfg)
    check_leakage(rows, split, candidate_pool_keys(rows["key"].dropna(), split))

    out_dir.mkdir(parents=True, exist_ok=True)
    split.val.to_parquet(out_dir / "val_ids.parquet", index=False)
    split.library_mask.to_parquet(out_dir / "library_mask.parquet", index=False)
    (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
    report = summary(rows, test, split, cfg)
    note = REPO_ROOT / "notes" / f"SPLIT_seed{cfg.seed}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(report)
    print(report)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
