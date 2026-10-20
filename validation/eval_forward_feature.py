"""Forward scores as ranker features (plan item 5, second use) vs RRF, on the v3 validation splits (reference).

Training rows: the r3 frames' Class 1 / 2 molecules with the answer in the window. Their head is the top
``TOP_K`` by an out-of-fold rank31 probability (5 folds grouped by the answer, so no fold sees its own answer
structure); the head candidates get forward scores (cached in ``outputs/forward_scores/<split>_<model>.parquet``,
like ``eval_forward.py``). Stage 2 is the rank31 recipe on the head rows with ``USED`` + the stage-1 probability
and its rank + ``casmi.forward.score.FEATURES`` per model (nan without a score), refitted per validation split
without that split's answers. On v3 the stage-1 ranker and the v3 forward scores are ``eval_forward.py``'s (run it
first). Report: ``notes/EXP_forward_feature.md``.

    python validation/eval_forward_feature.py --models glacier iceberg
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import forward, rank31
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.forward import score as fs
from casmi.harness import load_val
from casmi.pipeline import clean_queries
from casmi.report import md_table
from eval_forward import NOISE, OUT, WEIGHTS, our_rows, stage1
from eval_rank31 import FEATS, class_mrr

FOLDS = 5


def oof_proba(ours: pd.DataFrame) -> np.ndarray:
    """Out-of-fold rank31 probability, folds grouped by the answer structure."""
    answers = np.array(sorted(ours["answer"].unique()))
    fold_of = dict(zip(answers, np.random.default_rng(0).permutation(len(answers)) % FOLDS))
    fold = ours["answer"].map(fold_of).to_numpy()
    p = np.zeros(len(ours))
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        models = rank31.fit(ours.loc[tr, rank31.USED].to_numpy(np.float32), ours.loc[tr, "label"].to_numpy(float),
                            (ours.loc[tr, "class"] == 2).to_numpy(float))
        p[te] = rank31.proba(models, ours.loc[te, rank31.USED].to_numpy(np.float32))
    return p


def stage2_columns(df: pd.DataFrame, group: np.ndarray, p1: np.ndarray,
                   fws: dict[str, np.ndarray]) -> tuple[pd.DataFrame, list[str]]:
    """Stage-2 features; ``group`` is the query (split and molecule: NP examples repeat across splits)."""
    mol = group
    out = df[rank31.USED].copy()
    out["p1"], out["p1_rank"] = p1, fs.head(mol, p1)
    names = [*rank31.USED, "p1", "p1_rank"]
    for name, fw in fws.items():
        cols = fs.feature_columns(mol, fw).add_prefix(f"{name}_")
        out[cols.columns] = cols.to_numpy()
        names += list(cols.columns)
    return out, names


def scores_for(split: str, name: str, heads: dict, data, models: dict, caches: dict, weights: Path) -> dict:
    path = OUT / f"{split}_{name}.parquet"
    if path.exists():
        got = pd.read_parquet(path)
        have = set(zip(got["molecule_id"], got["key"]))
        if all((m, k) in have for m, keys in heads.items() for k in keys[:1]):
            return dict(zip(zip(got["molecule_id"], got["key"]), got["fw"]))
    if name not in models:
        models[name] = forward.load_model(name, lambda f: str(weights / f))
    queries = clean_queries(data.queries).reset_index(drop=True)
    smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
    scores, info = fs.forward_scores(models[name], queries, heads, smiles_of, cache=caches.setdefault(name, {}))
    print(f"{split} {name}: {info}", flush=True)
    pd.DataFrame([(m, k, v) for (m, k), v in scores.items()], columns=["molecule_id", "key", "fw"]).to_parquet(
        path, index=False)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["glacier", "iceberg"], choices=sorted(forward.FILES))
    ap.add_argument("--ours-frames", nargs="+", default=[f"r3_seed{s}" for s in range(10, 15)])
    ap.add_argument("--val", nargs="+", default=["v3_seed0", "v3_seed1", "v3_seed2"])
    ap.add_argument("--rrf-k", type=int, default=fs.RRF_K)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ours = our_rows(args.ours_frames).reset_index(drop=True)
    p1 = oof_proba(ours)
    print(f"out-of-fold stage 1 on {len(ours):,} rows: {time.time() - t0:.0f}s", flush=True)
    gkey = ours["split"] + "|" + ours["molecule_id"]
    in_head = fs.head(gkey.to_numpy(), p1) <= fs.TOP_K
    models, caches = {}, {}
    fw_ours = {n: np.full(len(ours), np.nan) for n in args.models}
    for split in args.ours_frames:
        rows = (ours["split"] == split).to_numpy()
        sub = ours[rows]
        heads = fs.top_keys(sub["molecule_id"].to_numpy(), sub["key"].to_numpy(), p1[rows])
        data = load_val(WORK_DIR / "splits" / split)
        for name in args.models:
            scores = scores_for(split, name, heads, data, models, caches, args.weights)
            fw_ours[name][rows] = fs.column(sub["molecule_id"].to_numpy(), sub["key"].to_numpy(), scores)
    train, names = stage2_columns(ours, gkey.to_numpy(), p1,
                                  {n: np.where(in_head, v, np.nan) for n, v in fw_ours.items()})
    train, y = train[in_head], ours.loc[in_head, "label"].to_numpy(float)
    m2, answer = (ours.loc[in_head, "class"] == 2).to_numpy(float), ours.loc[in_head, "answer"].to_numpy()
    print(f"stage-2 rows {len(train):,} ({len(names)} features), answers {y.sum():.0f}, {time.time() - t0:.0f}s",
          flush=True)
    res = []
    for split in args.val:
        df = pd.read_parquet(FEATS / f"{split}.parquet")
        val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
        base = stage1(split, ours, df, val)
        mol, keys = df["molecule_id"].to_numpy(), df["key"].to_numpy()
        head_v = fs.head(mol, base) <= fs.TOP_K
        fws = {}
        for name in args.models:
            got = pd.read_parquet(OUT / f"{split}_{name}.parquet")
            fws[name] = np.where(head_v, fs.column(mol, keys, dict(zip(zip(got["molecule_id"], got["key"]),
                                                                        got["fw"]))), np.nan)
        test, _ = stage2_columns(df, mol, base, fws)
        keep = ~np.isin(answer, val["inchikey14"].to_numpy())
        st2 = rank31.fit(train[names].to_numpy(np.float32)[keep], y[keep], m2[keep])
        p2 = rank31.proba(st2, test.loc[head_v, names].to_numpy(np.float32))
        s2 = np.where(head_v, 0.0, 1.0 / (1.0 + fs.head(mol, base)))
        s2[head_v] = 2.0 + p2
        runs = {"ranker alone": base, f"RRF ranker + {' + '.join(fws)}, k={args.rrf_k}":
                fs.rrf(mol, base, list(fws.values()), args.rrf_k),
                f"stage-2 ranker with {' + '.join(fws)} features": s2,
                f"stage-2 + RRF {' + '.join(fws)}, k={args.rrf_k}": fs.rrf(mol, s2, list(fws.values()), args.rrf_k)}
        for run, s in runs.items():
            res.append({"run": run, "split": split, **class_mrr(df, s, val)})
        print(f"{split} done, {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(res)
    agg = res.groupby("run", sort=False)[["C1", "C2", "NP C2"]].mean()
    tab = agg.reset_index()
    for c in ("C1", "C2", "NP C2"):
        d = tab[c].to_numpy() - agg.loc["ranker alone", c]
        tab[f"Δ {c}"] = [f"{x:+.3f}" + ("" if abs(x) > NOISE else " (n.m.)") for x in d]
    report = "\n\n".join([
        "# Forward scores as ranker features vs RRF (v3, reference)",
        f"- training: head (top {fs.TOP_K} by out-of-fold rank31, {FOLDS} answer-grouped folds) of "
        f"{', '.join(args.ours_frames)}: {len(train):,} rows, {len(names)} features; refitted per split without "
        f"its answers; models {', '.join(args.models)}; n.m. = within ±{NOISE}",
        md_table(tab.round(3)), f"- runtime {time.time() - t0:.0f}s"]) + "\n"
    (REPO_ROOT / "notes" / "EXP_forward_feature.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
