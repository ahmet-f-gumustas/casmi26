"""Ranker v0: tune channel weights on half of the validation molecules, test on the other half.

All three channels are computed through ``casmi.pipeline`` (the code path the Kaggle
notebook runs). Molecules are split into halves A/B, stratified by class (seeded).
Weights are grid-searched on A by total MRR@25 and reported on B next to each single
channel. The chosen weights are written to ``outputs/ranker_v0.json``.

    python validation/eval_ranker.py [--split outputs/splits/seed0] [--half-seed 0]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import NOISE, class_mrr, compare, compute_channels, load_val, predict
from casmi.report import md_table

GRID = {"library": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0), "analog": (1.0,), "fragment": (0.0, 0.1, 0.25, 0.5, 1.0)}
SINGLES = {"library only": {"library": 1.0}, "analog only": {"analog": 1.0}, "fragment only": {"fragment": 1.0}}


def halves(val: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    a_ids = []
    for _, g in val.groupby("class"):
        ids = g["molecule_id"].to_numpy()
        a_ids += list(rng.permutation(ids)[: len(ids) // 2])
    a = val["molecule_id"].isin(a_ids)
    return val[a], val[~a]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=WORK_DIR / "splits" / "seed0")
    ap.add_argument("--index", type=Path, default=WORK_DIR / "analog_index")
    ap.add_argument("--half-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()

    data = load_val(args.split)
    windows, channels = compute_channels(data, args.index, workers=args.workers)
    smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
    half_a, half_b = halves(data.val, args.half_seed)

    grid_rows = []
    for w_lib, w_ana, w_frag in itertools.product(GRID["library"], GRID["analog"], GRID["fragment"]):
        w = {"library": w_lib, "analog": w_ana, "fragment": w_frag}
        res = class_mrr(predict(windows, channels, w, smiles_of, half_a["molecule_id"]), half_a, json.dumps(w))
        grid_rows.append({**res, "weights": w})
    grid = pd.DataFrame(grid_rows).sort_values("total", ascending=False)
    best = grid.iloc[0]["weights"]

    rows = []
    for label, w in {**SINGLES, "library + analog (equal)": {"library": 1.0, "analog": 1.0},
                     "ranker v0 (tuned on A)": best}.items():
        for name, part in (("A", half_a), ("B", half_b), ("all", data.val)):
            rows.append({**class_mrr(predict(windows, channels, w, smiles_of, part["molecule_id"]), part, label),
                         "half": name})
    table = pd.DataFrame(rows)
    b = table[table["half"] == "B"].set_index("run")["total"]
    verdicts = [f"- ranker v0 vs analog only (half B): {compare(b['analog only'], b['ranker v0 (tuned on A)'])}",
                f"- ranker v0 vs library + analog equal (half B): "
                f"{compare(b['library + analog (equal)'], b['ranker v0 (tuned on A)'])}"]
    (WORK_DIR / "ranker_v0.json").write_text(json.dumps({"weights": best, "split": args.split.name,
                                                         "half_seed": args.half_seed}, indent=2) + "\n")
    report = "\n\n".join([
        f"# Ranker v0 ({args.split.name}, halves seeded {args.half_seed}: A={len(half_a)}, B={len(half_b)})",
        f"- chosen weights (best total on A): `{json.dumps(best)}`; differences ≤ {NOISE} are not measurable",
        "### Results", md_table(table[["run", "half", "C1", "C2", "C3", "total", "hits@1", "n"]]),
        *verdicts,
        "### Grid on half A (top 10)", md_table(grid.drop(columns=["weights"]).head(10)),
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP004_ranker_{args.split.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
