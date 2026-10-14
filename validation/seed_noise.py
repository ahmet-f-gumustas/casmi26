"""Seed noise of the validation harness.

Runs the library and analog channels on several split seeds and reports MRR@25 per
class for library-only, analog-only (ranker v0's choice) and their equal-weight
rank-normalised sum, with the across-seed mean and standard deviation. The std of the
total is the decision threshold for later comparisons.

    python validation/seed_noise.py [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import class_mrr, compute_channels, load_val, predict
from casmi.report import md_table

RUNS = {"library only": {"library": 1.0}, "analog only (ranker v0)": {"analog": 1.0},
        "library + analog (equal)": {"library": 1.0, "analog": 1.0}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prefix", default="seed", help="split directory prefix: seed (v1) or v2_seed (split v2)")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for seed in args.seeds:
        data = load_val(WORK_DIR / "splits" / f"{args.prefix}{seed}")
        windows, channels = compute_channels(data, channels=("library", "analog"), workers=args.workers)
        smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
        for label, w in RUNS.items():
            rows.append({**class_mrr(predict(windows, channels, w, smiles_of, data.val["molecule_id"]), data.val, label),
                         "seed": seed})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    stats = df.groupby("run")[["C1", "C2", "C3", "total"]].agg(["mean", "std"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    report = "\n\n".join([
        f"# Seed noise ({args.prefix}, seeds {args.seeds})",
        "### Per seed", md_table(df[["run", "seed", "C1", "C2", "C3", "total", "hits@1", "n"]]),
        "### Mean and std across seeds", md_table(stats.reset_index()),
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    suffix = "" if args.prefix == "seed" else "_" + args.prefix.removesuffix("_seed")
    (REPO_ROOT / "notes" / f"EXP005_seed_noise{suffix}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
