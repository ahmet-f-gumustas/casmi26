"""One combiner for two validation splits: the training-free grid scored on both, the joint winner.

``eval_combiners.py`` adds the fragment channel only to each split's own best configuration, so the
two splits' fragment variants differ. Here every training-free configuration of its grid (top-K
re-rank, product, scale-free product) runs with no fragment, + fragment v1 and (``--fragment2``)
+ fragment v2 at the same weights, on every split of every set (feature tables from
``validation/ranker_features.py``, one fingerprint model per set).

Choice (the rule used for the Oct 18 combiner): a configuration qualifies when on every set its C1
is at most ``--c1-loss`` below analog only and its NP C2 is above analog only. Among them the best
worst-case NP C2 gain over the sets; within the NP C2 noise (0.02) of it, ``--incumbent`` (the
configuration already on the LB queue) when it is among them, else the best worst-case C1.
Report: ``notes/EXP_combiners_<name>.md``; the choice: ``outputs/combiner_<name>.json``.

    python validation/joint_combiners.py --sets v3=outputs/ranker_features_e5:v3_seed \\
        v5=outputs/ranker_features_sim3:v5_seed --fragment2 "ablation:depth 3, no losses" \
        --incumbent "(a) top-100 re-rank α=0.5 + 0.3 fragment"
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from casmi.combiners import combine_rows
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.report import md_table
from eval_combiners import NOISE, add_fragment2, class_mrr, grid, load

EXTRAS = {"topk": (0.1, 0.2, 0.3), "mixed": (0.25, 0.5, 1.0), "product": (0.05, 0.2)}


def full_grid(fragment2: bool) -> dict[str, dict]:
    out = {}
    for name, spec in grid().items():
        out[name] = spec
        if spec["kind"] not in EXTRAS:
            continue
        cols = {"fragment": "fragment", **({"fragment2": "fragment v2"} if fragment2 else {})}
        for col, shown in cols.items():
            for w in EXTRAS[spec["kind"]]:
                label = f"{shown} T3={w}" if spec["kind"] == "product" else f"{w} {shown}"
                out[f"{name} + {label}"] = {**spec, "extra": {col: w}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="name=feature folder:split prefix")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fragment2", default=None, metavar="GRID:SETTING")
    ap.add_argument("--c1-loss", type=float, default=0.01)
    ap.add_argument("--incumbent", default=None, help="configuration kept when within the noise of the best")
    ap.add_argument("--name", default="joint")
    args = ap.parse_args()
    t0 = time.time()
    specs = full_grid(bool(args.fragment2))
    rows = []
    for item in args.sets:
        set_name, rest = item.split("=", 1)
        folder, prefix = rest.rsplit(":", 1)
        for seed in args.seeds:
            split = f"{prefix}{seed}"
            df, val = load(REPO_ROOT / folder, split)
            if args.fragment2:
                df = add_fragment2(df, split, args.fragment2)
            for run, spec in specs.items():
                mrr = class_mrr(df, combine_rows(df, spec), val)
                rows.append({"set": set_name, "split": split, "run": run, **mrr})
        print(f"{set_name}: {len(specs)} configurations ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    m = res.groupby(["run", "set"], sort=False)[["C1", "C2", "NP C2"]].mean().unstack("set")
    sets = list(dict.fromkeys(res["set"]))
    gain = {c: m[c].sub(m.loc["analog only", c], axis=1) for c in ("C1", "NP C2")}
    worst_np, worst_c1 = gain["NP C2"].min(axis=1), gain["C1"].min(axis=1)
    ok = (worst_c1 >= -args.c1_loss) & (worst_np > 0)
    ok["analog only"] = False
    if not ok.any():
        raise SystemExit("no configuration keeps C1 and gains NP C2 on every set")
    top = worst_np[ok].max()
    close = worst_np[ok][worst_np[ok] >= top - NOISE["NP C2"]].index
    kept = args.incumbent in close
    pick = args.incumbent if kept else max(close, key=lambda r: (worst_c1[r], worst_np[r]))

    tab = pd.DataFrame(index=m.index)
    for s in sets:
        for c in ("C1", "C2", "NP C2"):
            tab[f"{s} {c}"] = m[(c, s)].round(3)
        tab[f"{s} Δ NP C2"] = gain["NP C2"][s].map(
            lambda x: f"{x:+.3f}" + ("" if abs(x) > NOISE["NP C2"] else " (n.m.)"))
    tab["worst Δ NP C2"] = worst_np.round(3)
    tab["worst Δ C1"] = worst_c1.round(3)
    tab["qualifies"] = ok
    tab = tab.sort_values(["qualifies", "worst Δ NP C2"], ascending=False)
    shown = pd.concat([tab.loc[["analog only", "rank average (analog + FP)"]], tab.drop(
        ["analog only", "rank average (analog + FP)"]).head(25)])
    summary = {"run": pick, "spec": specs[pick], "incumbent_kept": kept, "sets": sets, "fragment2": args.fragment2,
               **{f"{s} {c}": round(float(m.loc[pick, (c, s)]), 4) for s in sets for c in ("C1", "C2", "NP C2")},
               "worst_np_c2_gain": round(float(worst_np[pick]), 4), "worst_c1_change": round(float(worst_c1[pick]), 4)}
    (WORK_DIR / f"combiner_{args.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    res.to_parquet(WORK_DIR / f"combiners_{args.name}.parquet", index=False)
    report = "\n\n".join([
        f"# One combiner for {' and '.join(sets)} ({', '.join(args.sets)})",
        f"- {len(specs)} training-free configurations; MRR@25 mean over seeds {args.seeds}; Δ vs analog only, "
        f"n.m. = within ±{NOISE['NP C2']}; fragment v2: {args.fragment2 or 'not used'}",
        f"- qualifies: C1 at most {args.c1_loss} below analog only and NP C2 above it on every set; "
        f"{int(ok.sum())} of {len(ok)} do",
        "### Baselines and the 25 best by worst-case NP C2 gain", md_table(shown.reset_index()),
        f"- choice: **{pick}** (worst-case NP C2 gain {worst_np[pick]:+.3f}, worst-case C1 change "
        f"{worst_c1[pick]:+.3f}); configurations within {NOISE['NP C2']} NP C2 of the best: {len(close)}"
        + ("; the incumbent is among them and stays" if kept else
           f"; incumbent {args.incumbent!r} {'not among them' if args.incumbent else 'not given'}"),
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP_combiners_{args.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
