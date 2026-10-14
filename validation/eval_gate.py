"""Ranker v1: library gate vs analog only, on several split seeds.

Configurations: analog only (ranker v0), library + analog equal (v0 failure case),
gate (library counted only where its best match ≥ τ; τ ∈ {0.6, 0.7, 0.8}, weight
w_lib ∈ {0.5, 1, 2}) and two-stage (library top-1 first if > τ, the rest from analog).
Reports MRR@25 per class as mean ± std over seeds, the difference to analog only with
the noise labels (total ±0.015, C1 ±0.04, C2 ±0.02), and how often the gate fires on
Class 1 (wanted) and Class 2 (false). All channel scores, fragment included, are saved
per split as ranker features. The best configuration goes to ``outputs/ranker_v1.json``.

    python validation/eval_gate.py [--prefix v2_seed] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import class_mrr, compute_channels, load_val, predict
from casmi.report import md_table

NOISE = {"total": 0.015, "C1": 0.04, "C2": 0.02}
CONFIGS = {"analog only": {"mode": "weighted", "weights": {"analog": 1.0}},
           "library + analog (equal, v0)": {"mode": "weighted", "weights": {"library": 1.0, "analog": 1.0}}}
for tau in (0.6, 0.7, 0.8):
    for w in (0.5, 1.0, 2.0):
        CONFIGS[f"gate τ={tau} w={w}"] = {"mode": "gate", "tau": tau, "w_lib": w}
    CONFIGS[f"two-stage τ={tau}"] = {"mode": "two_stage", "tau": tau}


def save_features(split: str, windows: dict, channels: dict) -> None:
    rows = []
    for m, keys in windows.items():
        lib, lmax = channels["library"].get(m, {}), channels["library_max"].get(m, {})
        ana, frag = channels["analog"].get(m, {}), channels["fragment"].get(m, {})
        rows += [(m, k, lib.get(k, 0.0), lmax.get(k, 0.0), ana.get(k, 0.0), frag.get(k, 0.0)) for k in keys]
    out = WORK_DIR / "channels" / split
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["molecule_id", "key", "library_mean", "library_max", "analog", "fragment"]).to_parquet(
        out / "features.parquet", index=False)


def gate_stats(val: pd.DataFrame, windows: dict, lmax: dict, tau: float) -> dict:
    out = {"tau": tau}
    for cls in (1, 2):
        ids = val.loc[val["class"] == cls, ["molecule_id", "inchikey14"]].itertuples(index=False)
        fires, truth_top = [], []
        for m, truth in ids:
            s = {k: v for k, v in lmax.get(m, {}).items() if k in set(windows.get(m, []))}
            best = min(s, key=lambda k: (-s[k], k)) if s else None
            fires.append(best is not None and s[best] > tau)
            truth_top.append(best == truth and s[best] > tau)
        out[f"C{cls} fires"] = float(np.mean(fires))
        out[f"C{cls} top-1 correct when fired"] = float(np.sum(truth_top) / max(np.sum(fires), 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="v2_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    rows, stats = [], []
    for seed in args.seeds:
        split = f"{args.prefix}{seed}"
        data = load_val(WORK_DIR / "splits" / split)
        windows, channels = compute_channels(data, workers=args.workers)
        save_features(split, windows, channels)
        smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
        ids = data.val["molecule_id"]
        for label, cfg in CONFIGS.items():
            rows.append({**class_mrr(predict(windows, channels, cfg, smiles_of, ids), data.val, label), "seed": seed})
        stats += [{**gate_stats(data.val, windows, channels["library_max"], tau), "seed": seed} for tau in (0.6, 0.7, 0.8)]
        print(f"{split} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("run", sort=False)[["C1", "C2", "C3", "total"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    base = agg.loc["analog only"]

    def verdict(row: pd.Series, col: str) -> str:
        d = row[f"{col}_mean"] - base[f"{col}_mean"]
        return f"{d:+.3f}" + ("" if abs(d) > NOISE[col] else " (n.m.)")

    agg["Δ total vs analog"] = agg.apply(verdict, axis=1, col="total")
    agg["Δ C1"] = agg.apply(verdict, axis=1, col="C1")
    agg["Δ C2"] = agg.apply(verdict, axis=1, col="C2")
    agg = agg.reset_index()
    best_label = agg.sort_values("total_mean", ascending=False)["run"].iloc[0]
    (WORK_DIR / "ranker_v1.json").write_text(json.dumps({"ranker": CONFIGS[best_label], "label": best_label,
                                                         "splits": f"{args.prefix}{args.seeds}"}, indent=2) + "\n")
    gs = pd.DataFrame(stats).groupby("tau").mean(numeric_only=True).drop(columns="seed").reset_index()
    cols = ["run", "C1_mean", "C1_std", "C2_mean", "C2_std", "total_mean", "total_std", "Δ total vs analog", "Δ C1", "Δ C2"]
    report = "\n\n".join([
        f"# Ranker v1: library gate ({args.prefix} {args.seeds})",
        "- n.m. = not measurable (|Δ| ≤ 0.015 total, 0.04 C1, 0.02 C2); C3 is 0 everywhere",
        "### MRR@25, mean ± std over seeds", md_table(agg[cols]),
        f"- best by mean total: **{best_label}** → `outputs/ranker_v1.json`",
        "### How often the library top-1 clears τ (gate fires), mean over seeds", md_table(gs),
        f"- channel features (library mean/max, analog, fragment) saved to outputs/channels/<split>/features.parquet; "
        f"runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP010_gate_{args.prefix}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
