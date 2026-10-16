"""Backtest validation splits against leaderboard results we already have.

A split is useful for a decision only if it reproduces what the leaderboard said about similar
decisions. Two public LB points can be recomputed from feature tables built with the fingerprint
model of that submission (``outputs/fp_models/rank_full``):

- analog only: 0.285;
- rank average of analog and our fingerprint channel: 0.262 (Δ −0.023 vs analog only).

Per split family: MRR@25 per class for both (mean over seeds), the implied LB
``0.16·C1 + 0.45·C2`` (Class 3 scores 0) with C2 taken over the NP subset (the test is natural
products; all Class 2 is shown too), and the implied Δ next to the LB Δ. A family passes when its
implied analog-only LB is within ``--tol-level`` of 0.285 and its implied Δ within ``--tol-delta`` of
−0.023; the chosen family passes both or, failing that, has the smallest error in tolerance units.
Report: ``notes/BACKTEST_LB.md``.

    python validation/backtest_lb.py --families v2=outputs/ranker_features:v2_seed v4=outputs/ranker_features_rankfull:v4_seed
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from casmi.config import REPO_ROOT, TOP_K, WORK_DIR
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

LB = {"analog only": 0.285, "rank average": 0.262}
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
        per = {"analog only": [], "rank average": []}
        for s in args.seeds:
            split = f"{prefix}{s}"
            df = pd.read_parquet(REPO_ROOT / folder / f"{split}.parquet")
            val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
            per["analog only"].append(class_mrr(df, df["analog"].to_numpy(), val))
            per["rank average"].append(class_mrr(df, (df["analog_pct"] + df["fingerprint_pct"]).to_numpy(), val))
        m = {k: pd.DataFrame(v).mean() for k, v in per.items()}
        implied = {k: {c2: SHARES["C1"] * v["C1"] + SHARES["C2"] * v[c2] for c2 in ("C2", "NP C2")} for k, v in m.items()}
        rows.append({"split": name, **{f"analog {c}": m["analog only"][c] for c in ("C1", "C2", "NP C2")},
                     **{f"rank avg {c}": m["rank average"][c] for c in ("C1", "C2", "NP C2")},
                     "implied LB analog (C2)": implied["analog only"]["C2"],
                     "implied LB analog (NP C2)": implied["analog only"]["NP C2"],
                     "implied Δ (C2)": implied["rank average"]["C2"] - implied["analog only"]["C2"],
                     "implied Δ (NP C2)": implied["rank average"]["NP C2"] - implied["analog only"]["NP C2"]})
    tab = pd.DataFrame(rows)
    lb_delta = LB["rank average"] - LB["analog only"]
    tab["level error"] = tab["implied LB analog (NP C2)"] - LB["analog only"]
    tab["Δ error"] = tab["implied Δ (NP C2)"] - lb_delta
    tab["passes"] = (tab["level error"].abs() <= args.tol_level) & (tab["Δ error"].abs() <= args.tol_delta)
    tab["error (tolerance units)"] = (tab["level error"].abs() / args.tol_level + tab["Δ error"].abs() / args.tol_delta)
    best = tab.sort_values(["passes", "error (tolerance units)"], ascending=[False, True]).iloc[0]
    verdict = (f"**{best['split']}** {'passes both' if best['passes'] else 'is the closest (passes neither fully)'}: "
               f"level error {best['level error']:+.3f}, Δ error {best['Δ error']:+.3f}")
    report = "\n\n".join([
        "# LB backtest of the validation splits",
        f"- LB: analog only {LB['analog only']}, rank average (analog + our FP, rank_full) {LB['rank average']} "
        f"(Δ {lb_delta:+.3f}); implied LB = 0.16·C1 + 0.45·C2 (Class 3 = 0); targets: level ± {args.tol_level}, "
        f"Δ ± {args.tol_delta} (NP C2); seeds {args.seeds}",
        md_table(tab.round(3)), f"- choice: {verdict}",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"{args.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
