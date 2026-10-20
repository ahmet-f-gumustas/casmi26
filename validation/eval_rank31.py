"""The V33 ranker (``casmi.rank31``) trained on prvsiyan's public rows vs on our simulated rows.

1. Feature distributions: mean per feature over all rows and over answer rows, public ``rank_train.npz``
   vs our rows (r3 splits) vs our validation rows; the largest differences point at definitions that do
   not match.
2. Both rankers (``PRIORS`` × ``SEEDS`` HistGradientBoosting, ``USED`` features) scored on the v3 validation
   splits: MRR@25 per class (C1 / C2 / NP C2), mean over seeds, next to the analog-only baseline. Every NP
   example is a validation molecule in every split, so our rows (built from the r3 feature frames) are
   refitted per validation split without the molecules whose answer is one of that split's answers. A
   reference only: the public rows' 819 queries are library molecules we cannot identify, so they may
   overlap our validation molecules; the decision is the leaderboard's.

Report: ``notes/EXP_rank31.md``.

    python validation/eval_rank31.py --public external/prvsiyan/ranker_features/rank_train.npz \\
        --ours outputs/rank31_rows/rank31_rows.npz --ours-frames r3_seed10 r3_seed11 r3_seed12 r3_seed13 r3_seed14
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import rank31
from casmi.config import REPO_ROOT, TOP_K, WORK_DIR
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

FEATS = WORK_DIR / "rank31_features"


def class_mrr(df: pd.DataFrame, score: np.ndarray, val: pd.DataFrame) -> dict:
    d = df[["molecule_id", "key", "label"]].assign(s=score)
    d = d.sort_values(["molecule_id", "s", "key"], ascending=[True, False, True])
    d["pos"] = d.groupby("molecule_id").cumcount() + 1
    hit = d[d["label"] & (d["pos"] <= TOP_K)].set_index("molecule_id")["pos"]
    rr = val["molecule_id"].map(1.0 / hit).fillna(0.0)
    c2 = val["class"] == 2
    return {"C1": rr[val["class"] == 1].mean(), "C2": rr[c2].mean(),
            "NP C2": rr[c2 & (val["source"] == NP_EXAMPLES)].mean()}


def load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return z["X"], z["Y"], z["M"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--ours-frames", nargs="+", default=[f"r3_seed{s}" for s in range(10, 15)],
                    help="feature frames our rows come from (for the per-split held-out fit)")
    ap.add_argument("--val", nargs="+", default=["v3_seed0", "v3_seed1", "v3_seed2"])
    args = ap.parse_args()
    t0 = time.time()
    rows = {"public": load_rows(args.public), "ours": load_rows(args.ours)}
    vals = {s: pd.read_parquet(FEATS / f"{s}.parquet") for s in args.val}
    val_all = pd.concat(vals.values(), ignore_index=True)

    dist = []
    for i, name in enumerate(rank31.FEATURES):
        rec = {"feature": name}
        for tag, (X, Y, _) in rows.items():
            rec[f"{tag} mean"], rec[f"{tag} answer mean"] = X[:, i].mean(), X[Y == 1, i].mean()
        v = val_all[name].to_numpy()
        rec["our val mean"], rec["our val answer mean"] = v.mean(), v[val_all["label"].to_numpy()].mean()
        sd = rows["public"][0][:, i].std()
        rec["|Δ mean| / public sd"] = abs(rec["ours mean"] - rec["public mean"]) / sd if sd > 0 else 0.0
        dist.append(rec)
    dist = pd.DataFrame(dist)

    names = rank31.USED
    cols = rank31.columns(names)
    res = []
    t1 = time.time()
    X, Y, M = rows["public"]
    models = rank31.fit(X[:, cols], Y, M)
    for split, df in vals.items():
        val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
        res.append({"run": "ranker, public rows", "split": split,
                    **class_mrr(df, rank31.proba(models, df[names].to_numpy(np.float32)), val)})
    print(f"public: {len(models)} models on {len(Y):,} rows, {time.time() - t1:.0f}s", flush=True)
    ours = pd.concat([pd.read_parquet(FEATS / f"{s}.parquet") for s in args.ours_frames], ignore_index=True)
    ours = ours[ours["class"].isin([1, 2])]
    ours = ours[ours.groupby(["split", "molecule_id"])["label"].transform("any")]
    answer = ours[ours["label"]].set_index(["split", "molecule_id"])["key"]
    ours["answer"] = [answer.get(sm) for sm in zip(ours["split"], ours["molecule_id"])]
    n_groups = ours.groupby(["split", "molecule_id"]).ngroups
    held_info = []
    for split, df in vals.items():
        t1 = time.time()
        val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
        tr = ours[~ours["answer"].isin(set(val["inchikey14"]))]
        for tag, part in (("ranker, our rows (held out)", tr), ("leaky: our rows incl. the split's answers", ours)):
            m = rank31.fit(part[names].to_numpy(np.float32), part["label"].to_numpy(float),
                           (part["class"] == 2).to_numpy(float))
            res.append({"run": tag, "split": split,
                        **class_mrr(df, rank31.proba(m, df[names].to_numpy(np.float32)), val)})
        held_info.append(f"{split}: {tr.groupby(['split', 'molecule_id']).ngroups:,} of {n_groups:,} groups kept")
        print(f"{split}: held-out fit {time.time() - t1:.0f}s", flush=True)
    for split, df in vals.items():
        val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
        res.append({"run": "analog propagation (ap) alone", "split": split, **class_mrr(df, df["ap"].to_numpy(), val)})
        res.append({"run": "library (lv) + ap", "split": split,
                    **class_mrr(df, (df["lv"] * 100 + df["ap"]).to_numpy(), val)})
    res = pd.DataFrame(res)
    agg = res.groupby("run", sort=False)[["C1", "C2", "NP C2"]].agg(["mean", "std"])
    tab = pd.DataFrame(index=agg.index)
    for c in ("C1", "C2", "NP C2"):
        tab[c] = agg[(c, "mean")].map("{:.3f}".format) + " ± " + agg[(c, "std")].map("{:.3f}".format)
    sizes = {t: f"{len(r[1]):,} rows, {r[1].sum():.0f} answers, Class 2 share {r[2].mean():.2f}"
             for t, r in rows.items()}
    report = "\n\n".join([
        "# V33 ranker: public rows vs our rows",
        f"- rows: public {sizes['public']}; ours {sizes['ours']}; validation {', '.join(args.val)} "
        f"({len(val_all):,} rows)",
        f"- features used: {len(names)} (dropped: {', '.join(rank31.DROP)}); {len(rank31.PRIORS)} Class 1 weights "
        f"{rank31.PRIORS} × {len(rank31.SEEDS)} seeds; GBM {rank31.GBM}",
        f"- held-out fit of our rows: {'; '.join(held_info)}",
        "### MRR@25 on our validation splits (reference; the public rows may overlap these molecules)",
        md_table(tab.reset_index()),
        "### Feature means (all rows / answer rows)", md_table(dist.round(3)),
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / "EXP_rank31.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
