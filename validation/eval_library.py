"""Evaluate the library channel on a validation split (EXP-001 baseline).

Queries are the split's query spectra from ``train_clean.parquet``; the library is
every cleaned spectrum still ``in_library`` under the split's mask; candidates are
pool structures in the molecule's mass window (C3 answers removed). Reports
MRR@25 per class and per source for max / mean aggregation, next to the library
coverage (answer has reference spectra in its window) and the expected MRR of a
random ranking of the window candidates.

    python validation/eval_library.py [--split outputs/splits/seed0] [--ppm 10] [--offset 2]
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from casmi.candidates import CandidatePool, molecule_masses
from casmi.channels.library import AGGREGATIONS, LibraryConfig, prepare, rank, score_molecule
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.metric import evaluate
from casmi.report import md_table

CLEAN = WORK_DIR / "train_clean.parquet"
COLS = ["row", "key", "ionization_mode", "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"]
CFG = LibraryConfig()
REFS: dict[str, list] = {}
QUERIES: dict[str, list] = {}
WINDOWS: dict[str, list[str]] = {}


def spectra(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    return [(mode, prepare(np.asarray(mz), np.asarray(it), prec, CFG))
            for mode, mz, it, prec in zip(df["ionization_mode"], df["ms2_mzs"], df["ms2_normalized_intensities"],
                                          df["precursor_mz"])]


def score_one(mol_id: str) -> tuple[str, pd.DataFrame]:
    return mol_id, score_molecule(QUERIES.get(mol_id, []), WINDOWS[mol_id], REFS, CFG)


def random_mrr(n: int, k: int = 25) -> float:
    return float(sum(1.0 / r for r in range(1, min(n, k) + 1)) / n) if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=WORK_DIR / "splits" / "seed0")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--offset", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()

    val = pd.read_parquet(args.split / "val_ids.parquet")
    mask = pd.read_parquet(args.split / "library_mask.parquet")
    qmask = mask[mask["query_molecule_id"].notna()]
    query_of = dict(zip(qmask["row"], qmask["query_molecule_id"]))
    dataset = ds.dataset(CLEAN)
    queries = dataset.to_table(columns=COLS, filter=pc.field("row").isin(list(query_of))).to_pandas()
    queries["molecule_id"] = queries["row"].map(query_of)
    lost_queries = len(qmask) - len(queries)

    pool = CandidatePool.load().exclude(val.loc[val["class"] == 3, "inchikey14"])
    keys = pool.meta["key"].to_numpy()
    masses = molecule_masses(queries)
    for mol_id in val["molecule_id"]:
        WINDOWS[mol_id] = keys[pool.window(masses[mol_id], args.ppm, args.offset)].tolist() if mol_id in masses else []
    cand = {k for ks in WINDOWS.values() for k in ks}

    lib = dataset.to_table(columns=COLS, filter=pc.field("key").isin(list(cand))).to_pandas()
    in_lib = mask.set_index("row")["in_library"]
    lib = lib[lib["row"].map(in_lib).to_numpy()]
    for key, grp in lib.groupby("key"):
        REFS[key] = spectra(grp)
    for mol_id, grp in queries.groupby("molecule_id"):
        QUERIES[mol_id] = spectra(grp)
    print(f"{len(val)} molecules, {len(queries)} queries ({lost_queries} lost in cleaning), {len(cand):,} candidate "
          f"structures, {len(lib):,} reference spectra; prep {time.time() - t0:.0f}s", flush=True)

    with Pool(args.workers) as workers:
        scores = dict(workers.map(score_one, val["molecule_id"].tolist(), chunksize=1))
    smiles_of = pool.meta.set_index("key")["smiles"]
    truth = dict(zip(val["molecule_id"], val["smiles"]))
    classes = dict(zip(val["molecule_id"], val["class"]))

    rows = []
    for by in AGGREGATIONS:
        preds = {m: [smiles_of[k] for k in rank(s, by)] for m, s in scores.items()}
        for source in ["all", *sorted(val["source"].unique())]:
            ids = val["molecule_id"] if source == "all" else val.loc[val["source"] == source, "molecule_id"]
            res = evaluate(preds, {m: truth[m] for m in ids}, classes=classes)
            rows.append({"aggregation": by, "source": source, "n": int(res["n"]), "mrr@25": res["mrr@25"],
                         **{f"C{c}": res.get(f"mrr_class{c}", np.nan) for c in (1, 2, 3)},
                         "hits@1": res["hits@1"], "hits@25": res["hits@25"]})
    results = pd.DataFrame(rows)

    key_of = dict(zip(val["molecule_id"], val["inchikey14"]))
    cov = pd.DataFrame([{
        "class": classes[m], "in_window": key_of[m] in WINDOWS[m], "has_refs": key_of[m] in REFS,
        "n_ranked": int((scores[m]["max"] > 0).sum()) if len(scores[m]) else 0,
        "random_mrr": random_mrr(len(WINDOWS[m])) if key_of[m] in WINDOWS[m] else 0.0,
    } for m in val["molecule_id"]])
    coverage = cov.groupby("class").agg(n=("in_window", "size"), answer_in_window=("in_window", "mean"),
                                        answer_has_refs=("has_refs", "mean"), median_ranked=("n_ranked", "median"),
                                        random_ranking_mrr=("random_mrr", "mean")).reset_index()
    report = "\n\n".join([
        f"# EXP-001 — library channel baseline ({args.split.name})",
        f"- window ±{args.ppm:g} ppm, +{args.offset:g} ppm offset; entropy similarity, ms2 tol {CFG.ms2_tol_da} Da, "
        f"noise {CFG.noise_threshold}, precursor removal {CFG.precursor_removal_da} Da; same ionization mode only",
        f"- {len(val)} molecules, {len(queries)} query spectra ({lost_queries} dropped by cleaning), "
        f"{len(cand):,} candidate structures, {len(lib):,} reference spectra; runtime {time.time() - t0:.0f}s",
        "### MRR@25", md_table(results),
        "### Coverage (upper bounds) and random-ranking reference", md_table(coverage),
    ]) + "\n"
    (REPO_ROOT / "notes" / "EXP001_library.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
