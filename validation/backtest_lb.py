"""Backtest validation splits against leaderboard results we already have.

A split is useful for a decision only if it reproduces what the leaderboard said about similar
decisions. These public LB points can be recomputed from a feature table (the family's fingerprint
model stands in for the submission's):

- analog only: 0.285 (the level);
- rank average of analog and our fingerprint channel (FP ``rank_full``): 0.262 (Δ −0.023);
- combiner without FP, analog top-100 re-rank + 0.3·z(fragment) (kernel v12): 0.299 (Δ +0.014);
- the same combiner with FP v2 epoch 5 (kernel v10): 0.304 (Δ +0.019).

Per split family: MRR@25 per class (mean over seeds), the implied LB ``0.16·C1 + 0.45·C2``
(Class 3 scores 0) with C2 taken over the NP subset (the test is natural products), and every
implied Δ next to its LB Δ. A family passes when its implied analog-only LB is within ``--tol-level``
of 0.285 and its rank-average Δ within ``--tol-delta`` of −0.023 (the original rule); "passes all"
also needs the combiner Δs within ``--tol-delta``. The chosen family passes or, failing that, has the
smallest error in tolerance units over every point. Report: ``notes/<--name>.md``.

    python validation/backtest_lb.py --families v2=outputs/ranker_features:v2_seed v4=outputs/ranker_features_rankfull:v4_seed
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from casmi.combiners import combine_rows
from casmi.config import REPO_ROOT, TOP_K, WORK_DIR
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

LB = {"analog only": 0.285, "rank average": 0.262, "combiner, no FP (v12)": 0.299, "combiner + FP (v10)": 0.304}
SPECS = {"analog only": {"kind": "analog"},
         "combiner, no FP (v12)": {"kind": "topk", "k": 100, "alpha": 0.5, "fp": None, "extra": {"fragment": 0.3}},
         "combiner + FP (v10)": {"kind": "topk", "k": 100, "alpha": 0.5, "extra": {"fragment": 0.3}}}
SHARES = {"C1": 0.16, "C2": 0.45}


def class_mrr(df: pd.DataFrame, score: np.ndarray, val: pd.DataFrame) -> dict:
    d = df[["molecule_id", "key", "label", "analog"]].assign(s=score)
    d = d.sort_values(["molecule_id", "s", "analog", "key"], ascending=[True, False, False, True])
    d["pos"] = d.groupby("molecule_id").cumcount() + 1
    hit = d[d["label"] & (d["pos"] <= TOP_K)].set_index("molecule_id")["pos"]
    rr = val["molecule_id"].map(1.0 / hit).fillna(0.0)
    c2 = val["class"] == 2
    return {"C1": rr[val["class"] == 1].mean(), "C2": rr[c2].mean(),
            "NP C2": rr[c2 & (val["source"] == NP_EXAMPLES)].mean()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", required=True, help="name=feature folder:split prefix")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tol-level", type=float, default=0.02)
    ap.add_argument("--tol-delta", type=float, default=0.015)
    ap.add_argument("--name", default="BACKTEST_LB")
    args = ap.parse_args()
    rows = []
    for item in args.families:
        name, rest = item.split("=", 1)
        folder, prefix = rest.rsplit(":", 1)
        per = {k: [] for k in LB}
        for s in args.seeds:
            split = f"{prefix}{s}"
            df = pd.read_parquet(REPO_ROOT / folder / f"{split}.parquet")
            val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
            for point in LB:
                score = ((df["analog_pct"] + df["fingerprint_pct"]).to_numpy() if point == "rank average"
                         else combine_rows(df, SPECS[point]))
                per[point].append(class_mrr(df, score, val))
        m = {k: pd.DataFrame(v).mean() for k, v in per.items()}
        implied = {k: SHARES["C1"] * v["C1"] + SHARES["C2"] * v["NP C2"] for k, v in m.items()}
        row = {"split": name, "analog C1": m["analog only"]["C1"], "analog NP C2": m["analog only"]["NP C2"],
               "implied LB analog": implied["analog only"],
               "level error": implied["analog only"] - LB["analog only"]}
        for point in list(LB)[1:]:
            d = implied[point] - implied["analog only"]
            row[f"Δ {point}"] = d
            row[f"error {point}"] = d - (LB[point] - LB["analog only"])
        rows.append(row)
    tab = pd.DataFrame(rows)
    errors = [c for c in tab.columns if c.startswith("error ")]
    tab["passes"] = (tab["level error"].abs() <= args.tol_level) & (tab["error rank average"].abs() <= args.tol_delta)
    tab["passes all"] = tab["passes"] & (tab[errors].abs() <= args.tol_delta).all(axis=1)
    tab["error (tolerance units)"] = (tab["level error"].abs() / args.tol_level
                                      + tab[errors].abs().sum(axis=1) / args.tol_delta)
    best = tab.sort_values(["passes all", "passes", "error (tolerance units)"], ascending=[False, False, True]).iloc[0]
    state = "passes all" if best["passes all"] else "passes level + rank average" if best["passes"] else \
        "is the closest (passes neither fully)"
    verdict = (f"**{best['split']}** {state}: level error {best['level error']:+.3f}, "
               + ", ".join(f"{c.removeprefix('error ')} {best[c]:+.3f}" for c in errors))
    lb = ", ".join(f"{k} {v} (Δ {v - LB['analog only']:+.3f})" for k, v in LB.items())
    report = "\n\n".join([
        "# LB backtest of the validation splits",
        f"- LB: {lb}; implied LB = 0.16·C1 + 0.45·NP C2 (Class 3 = 0); targets: level ± {args.tol_level}, "
        f"Δ ± {args.tol_delta}; seeds {args.seeds}",
        md_table(tab.round(3)), f"- choice: {verdict}",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"{args.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
