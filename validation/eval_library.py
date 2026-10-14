"""Evaluate the library channel on a validation split (EXP-001 baseline).

Runs ``casmi.pipeline.run_library_pipeline`` — the exact code path of the Kaggle
notebook — on the split's query spectra, with the library restricted by the
split's mask and Class 3 answers removed from the pool. Reports MRR@25 per class
for max / mean aggregation, plus coverage and a random-ranking reference.

    python validation/eval_library.py [--split outputs/splits/seed0] [--ppm 10] [--offset 2]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from casmi.channels.library import AGGREGATIONS
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import CLEAN_LIBRARY, class_mrr, load_val
from casmi.pipeline import PipelineConfig, candidate_windows, clean_queries, predictions, run_library_pipeline
from casmi.report import md_table


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

    data = load_val(args.split)
    cfg = PipelineConfig(ppm=args.ppm, offset_ppm=args.offset, workers=args.workers)
    _, scores = run_library_pipeline(data.queries, data.pool, CLEAN_LIBRARY, cfg, data.in_library)
    smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
    rows = []
    for by in AGGREGATIONS:
        preds = predictions(scores, smiles_of, by)
        rows.append(class_mrr(preds, data.val, f"library/{by}"))
        for src in sorted(data.val["source"].unique()):
            sub = data.val[data.val["source"] == src]
            rows.append(class_mrr(preds, sub, f"library/{by} [{src}]"))

    windows = candidate_windows(clean_queries(data.queries), data.pool, cfg)
    key_of = dict(zip(data.val["molecule_id"], data.val["inchikey14"]))
    cov = pd.DataFrame([{
        "class": c, "in_window": key_of[m] in windows.get(m, []),
        "has_refs": bool(len(scores[m])) and key_of[m] in set(scores[m]["key"]),
        "random_mrr": random_mrr(len(windows.get(m, []))) if key_of[m] in windows.get(m, []) else 0.0,
    } for m, c in zip(data.val["molecule_id"], data.val["class"])])
    coverage = cov.groupby("class").agg(n=("in_window", "size"), answer_in_window=("in_window", "mean"),
                                        answer_has_refs=("has_refs", "mean"),
                                        random_ranking_mrr=("random_mrr", "mean")).reset_index()
    report = "\n\n".join([
        f"# Library channel ({args.split.name})",
        f"- window ±{args.ppm:g} ppm, +{args.offset:g} ppm offset; {len(data.val)} molecules, "
        f"{len(data.queries)} query spectra; runtime {time.time() - t0:.0f}s",
        "### MRR@25", md_table(pd.DataFrame(rows)),
        "### Coverage and random-ranking reference", md_table(coverage),
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP001_library_{args.split.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
