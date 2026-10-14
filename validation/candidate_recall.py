"""Candidate recall of the structure pool on a validation split.

For each validation molecule the neutral mass is the median over its query spectra
(precursor_mz + adduct); candidates are pool structures within ±ppm of that mass after
the timsTOF offset correction, with Class 3 answers removed from the pool. Recall
should be ~100 % for C1/C2 and exactly 0 % for C3 (leakage check).

    python validation/candidate_recall.py [--split outputs/splits/seed0] [--ppm 10] [--offset 2]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.candidates import CandidatePool, molecule_masses
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.data import read_table
from casmi.report import md_table


def load_queries(split_dir: Path) -> pd.DataFrame:
    mask = pd.read_parquet(split_dir / "library_mask.parquet")
    q = mask[mask["query_molecule_id"].notna()]
    rows = read_table("train", ["precursor_mz", "adduct"]).to_pandas().iloc[q["row"].to_numpy()]
    return rows.assign(molecule_id=q["query_molecule_id"].to_numpy())


def recall_table(val: pd.DataFrame, pool: CandidatePool, masses: pd.Series, ppm: float, offset: float) -> pd.DataFrame:
    keys = pool.meta["key"].to_numpy()
    out = []
    for mol_id, key in zip(val["molecule_id"], val["inchikey14"]):
        idx = pool.window(masses[mol_id], ppm=ppm, offset_ppm=offset)
        out.append({"molecule_id": mol_id, "n_candidates": len(idx), "hit": bool((keys[idx] == key).any())})
    return val[["molecule_id", "class", "source"]].merge(pd.DataFrame(out), on="molecule_id")


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (df.groupby(by).agg(n=("hit", "size"), recall=("hit", "mean"),
                               median_candidates=("n_candidates", "median"),
                               p90_candidates=("n_candidates", lambda s: s.quantile(0.9)))
            .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=WORK_DIR / "splits" / "seed0")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--offset", type=float, default=2.0)
    args = ap.parse_args()

    val = pd.read_parquet(args.split / "val_ids.parquet")
    pool = CandidatePool.load()
    in_pool = set(pool.meta["key"])
    c3 = val.loc[val["class"] == 3, "inchikey14"]
    c3_in_full_pool = c3.isin(in_pool).mean()
    c3_in_coconut = c3.isin(set(pool.meta.loc[pool.meta["in_coconut"], "key"])).mean()
    val_pool = pool.exclude(c3)
    masses = molecule_masses(load_queries(args.split))

    main_tab = recall_table(val, val_pool, masses, args.ppm, args.offset)
    rows = [summarise(main_tab, ["class"]).assign(window=f"±{args.ppm:g} ppm, +{args.offset:g} offset")]
    for ppm, off in [(args.ppm, 0.0), (5.0, args.offset), (20.0, args.offset)]:
        t = recall_table(val, val_pool, masses, ppm, off)
        rows.append(summarise(t, ["class"]).assign(window=f"±{ppm:g} ppm, +{off:g} offset"))
    by_window = pd.concat(rows, ignore_index=True)[["window", "class", "n", "recall", "median_candidates",
                                                     "p90_candidates"]]
    leak = main_tab.loc[main_tab["class"] == 3, "hit"].any()
    misses = main_tab[(main_tab["class"] != 3) & ~main_tab["hit"]]
    report = "\n\n".join([
        f"# Candidate recall — {args.split.name}",
        f"- pool: {len(pool):,} structures; validation pool drops {len(c3)} C3 answers "
        f"({c3_in_full_pool:.0%} of them were in the full pool, {c3_in_coconut:.0%} in COCONUT)",
        f"- **leakage check: C3 answer found in a window → {'FAIL' if leak else 'none (OK)'}**",
        "### By class and source", md_table(summarise(main_tab, ["class", "source"])),
        "### Window variants", md_table(by_window),
        f"### C1/C2 misses ({len(misses)})", md_table(misses.head(20)) if len(misses) else "none",
    ]) + "\n"
    note = REPO_ROOT / "notes" / f"RECALL_{args.split.name}.md"
    note.write_text(report)
    print(report)
    if leak:
        raise SystemExit("C3 leakage")


if __name__ == "__main__":
    main()
