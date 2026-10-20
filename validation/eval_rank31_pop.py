"""Popularity prior on the rank31 ranker (plan item 4), on the v3 validation splits as a reference.

Popularity = log1p(PubChem substances) + log1p(PubMed records) per pool key (``popularity.parquet``,
from prvsiyan/pubchem-popularity-counts, CC0). Two uses:

- final score: V33's re-order of the top ``POP_TOP`` by z(p) + μ · pop where the best library match
  is below ``LIB_GATE`` (μ ∈ ``--mus``);
- ranker features ``pop_sid`` / ``pop_pmid``: only our rows can carry them (the public rows have no
  popularity), so for the public-row ranker they are a constant 0 column, i.e. neutral.

Rankers: public rows, and our r3 rows refitted per validation split without that split's answers
(``eval_rank31.py``). MRR@25 per class (C1 / C2 / NP C2), mean over seeds. Report: ``notes/EXP_rank31_pop.md``.

    python validation/eval_rank31_pop.py --public external/prvsiyan/ranker_features/rank_train.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import rank31
from casmi.candidates import POOL_DIR
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.report import md_table
from eval_rank31 import FEATS, class_mrr, load_rows

NOISE = 0.02


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", type=Path, required=True)
    ap.add_argument("--ours-frames", nargs="+", default=[f"r3_seed{s}" for s in range(10, 15)])
    ap.add_argument("--val", nargs="+", default=["v3_seed0", "v3_seed1", "v3_seed2"])
    ap.add_argument("--mus", type=float, nargs="+", default=[0.0, 0.15, 0.25, 0.35])
    args = ap.parse_args()
    t0 = time.time()
    table = pd.read_parquet(POOL_DIR / "popularity.parquet")
    names = rank31.USED
    Xp, Yp, Mp = load_rows(args.public)
    public = rank31.fit(Xp[:, rank31.columns(names)], Yp, Mp)
    ours = pd.concat([pd.read_parquet(FEATS / f"{s}.parquet") for s in args.ours_frames], ignore_index=True)
    ours = ours[ours["class"].isin([1, 2])]
    ours = ours[ours.groupby(["split", "molecule_id"])["label"].transform("any")]
    answer = ours[ours["label"]].set_index(["split", "molecule_id"])["key"]
    ours["answer"] = [answer.get(sm) for sm in zip(ours["split"], ours["molecule_id"])]
    ours[rank31.POP_FEATURES] = rank31.popularity_columns(ours["key"].tolist(), table)
    res = []
    for split in args.val:
        t1 = time.time()
        df = pd.read_parquet(FEATS / f"{split}.parquet")
        df[rank31.POP_FEATURES] = rank31.popularity_columns(df["key"].tolist(), table)
        val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
        pop = df[rank31.POP_FEATURES].sum(axis=1).to_numpy()
        tr = ours[~ours["answer"].isin(set(val["inchikey14"]))]
        y, m = tr["label"].to_numpy(float), (tr["class"] == 2).to_numpy(float)
        probs = {
            "public rows": rank31.proba(public, df[names].to_numpy(np.float32)),
            "our rows": rank31.proba(rank31.fit(tr[names].to_numpy(np.float32), y, m),
                                     df[names].to_numpy(np.float32)),
            "our rows + popularity features": rank31.proba(
                rank31.fit(tr[names + rank31.POP_FEATURES].to_numpy(np.float32), y, m),
                df[names + rank31.POP_FEATURES].to_numpy(np.float32)),
        }
        for run, p in probs.items():
            for mu in args.mus:
                s = rank31.popularity_scores(df["molecule_id"].to_numpy(), p, pop, df["lv_max"].to_numpy(), mu)
                res.append({"ranker": run, "μ": mu, "split": split, **class_mrr(df, s, val)})
        print(f"{split}: {time.time() - t1:.0f}s", flush=True)
    res = pd.DataFrame(res)
    agg = res.groupby(["ranker", "μ"], sort=False)[["C1", "C2", "NP C2"]].mean()
    base = agg.xs(0.0, level="μ")
    tab = agg.reset_index()
    for c in ("C1", "NP C2"):
        d = tab[c].to_numpy() - base.loc[tab["ranker"], c].to_numpy()
        tab[f"Δ {c} vs μ=0"] = [f"{x:+.3f}" + ("" if abs(x) > NOISE else " (n.m.)") for x in d]
    best = {}
    for run in base.index:
        sub = agg.loc[run]
        best[run] = float(sub["NP C2"].idxmax())
    report = "\n\n".join([
        "# Popularity prior on the rank31 ranker (v3, reference)",
        f"- popularity: log1p(substances) + log1p(PubMed) per pool key; re-order of the top {rank31.POP_TOP} by "
        f"z(p) + μ·pop where the best library match < {rank31.LIB_GATE}; validation {', '.join(args.val)}; "
        f"n.m. = within ±{NOISE}",
        md_table(tab.round(3)),
        "- best μ by NP C2: " + "; ".join(f"{k}: {v}" for k, v in best.items()),
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / "EXP_rank31_pop.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
