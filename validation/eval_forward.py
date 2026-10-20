"""Forward-model rescoring of the rank31 ranker's top 50 (plan item 5) on the v3 validation splits (reference).

Stage 1 is the rank31 ranker on our r3 rows, refitted per validation split without that split's answers
(``eval_rank31.py``). Its top ``TOP_K`` candidates per molecule get forward scores from each model in
``--models`` (``casmi.forward.score``; predictions shared across splits through one cache per model). The scores
are written to ``outputs/forward_scores/<split>_<model>.parquet`` and reused on a rerun (``--reuse``).

Runs: the ranker alone; each model and all models together by RRF with k ∈ ``--rrf-k``; each model alone
inside the head (diagnostic). MRR@25 per class (C1 / C2 / NP C2), mean over the splits, with the answer's
presence in the head and the share of head rows with a forward score. Report: ``notes/EXP_forward.md``.

    python validation/eval_forward.py --models glacier iceberg
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
from eval_rank31 import FEATS, class_mrr

OUT = WORK_DIR / "forward_scores"
WEIGHTS = WORK_DIR / "forward_msg"
NOISE = 0.02


def stage1(split: str, ours: pd.DataFrame, df: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    tr = ours[~ours["answer"].isin(set(val["inchikey14"]))]
    models = rank31.fit(tr[rank31.USED].to_numpy(np.float32), tr["label"].to_numpy(float),
                        (tr["class"] == 2).to_numpy(float))
    return rank31.proba(models, df[rank31.USED].to_numpy(np.float32))


def our_rows(frames: list[str]) -> pd.DataFrame:
    ours = pd.concat([pd.read_parquet(FEATS / f"{s}.parquet") for s in frames], ignore_index=True)
    ours = ours[ours["class"].isin([1, 2])]
    ours = ours[ours.groupby(["split", "molecule_id"])["label"].transform("any")]
    answer = ours[ours["label"]].set_index(["split", "molecule_id"])["key"]
    ours["answer"] = [answer.get(sm) for sm in zip(ours["split"], ours["molecule_id"])]
    return ours


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["glacier", "iceberg"], choices=sorted(forward.FILES))
    ap.add_argument("--ours-frames", nargs="+", default=[f"r3_seed{s}" for s in range(10, 15)])
    ap.add_argument("--val", nargs="+", default=["v3_seed0", "v3_seed1", "v3_seed2"])
    ap.add_argument("--rrf-k", type=int, nargs="+", default=[5, 10, 30, 60])
    ap.add_argument("--top", type=int, default=fs.TOP_K)
    ap.add_argument("--reuse", action="store_true", help="read forward scores written by an earlier run")
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ours = our_rows(args.ours_frames)
    models, caches, timing = {}, {n: {} for n in args.models}, {n: [0.0, 0] for n in args.models}
    res, cover = [], []
    for split in args.val:
        t1 = time.time()
        df = pd.read_parquet(FEATS / f"{split}.parquet")
        data = load_val(WORK_DIR / "splits" / split)
        val = data.val
        base = stage1(split, ours, df, val)
        mol, keys = df["molecule_id"].to_numpy(), df["key"].to_numpy()
        heads = fs.top_keys(mol, keys, base, args.top)
        in_head = fs.head(mol, base) <= args.top
        fws = {}
        for name in args.models:
            path = OUT / f"{split}_{name}.parquet"
            if args.reuse and path.exists():
                got = pd.read_parquet(path)
                scores = dict(zip(zip(got["molecule_id"], got["key"]), got["fw"]))
            else:
                if name not in models:
                    models[name] = forward.load_model(name, lambda f: str(args.weights / f))
                queries = clean_queries(data.queries).reset_index(drop=True)
                smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
                scores, info = fs.forward_scores(models[name], queries, heads, smiles_of, cache=caches[name])
                timing[name][0] += info["seconds"]
                timing[name][1] += info["predictions"]
                print(f"{split} {name}: {info}", flush=True)
                pd.DataFrame([(m, k, v) for (m, k), v in scores.items()],
                             columns=["molecule_id", "key", "fw"]).to_parquet(path, index=False)
            fws[name] = fs.column(mol, keys, scores)
        label = df["label"].to_numpy(bool)
        cover.append({"split": split, "answer in head": df.loc[label & in_head, "molecule_id"].nunique()
                      / max(df.loc[label, "molecule_id"].nunique(), 1),
                      **{f"{n} scored head rows": float(np.isfinite(fws[n][in_head]).mean()) for n in args.models}})
        runs = {"ranker alone": base}
        for name, fw in fws.items():
            runs[f"{name} alone in the head"] = np.where(in_head, 2.0 + np.nan_to_num(fw, nan=-1.0), base)
            for k in args.rrf_k:
                runs[f"RRF ranker + {name}, k={k}"] = fs.rrf(mol, base, [fw], k, args.top)
        if len(fws) > 1:
            for k in args.rrf_k:
                runs[f"RRF ranker + {' + '.join(fws)}, k={k}"] = fs.rrf(mol, base, list(fws.values()), k, args.top)
        for run, s in runs.items():
            res.append({"run": run, "split": split, **class_mrr(df, s, val)})
        print(f"{split}: {time.time() - t1:.0f}s", flush=True)
    res = pd.DataFrame(res)
    agg = res.groupby("run", sort=False)[["C1", "C2", "NP C2"]].mean()
    tab = agg.reset_index()
    for c in ("C1", "C2", "NP C2"):
        d = tab[c].to_numpy() - agg.loc["ranker alone", c]
        tab[f"Δ {c}"] = [f"{x:+.3f}" + ("" if abs(x) > NOISE else " (n.m.)") for x in d]
    speed = "; ".join(f"{n}: {p:,} predictions in {s:.0f}s ({p / s:.1f}/s)" for n, (s, p) in timing.items() if p)
    report = "\n\n".join([
        "# Forward-model rescoring of the rank31 top 50 (v3, reference)",
        f"- stage 1: rank31 on our rows {', '.join(args.ours_frames)}, refitted per split without its answers; "
        f"head = top {args.top}; models {', '.join(args.models)} ({args.weights}); n.m. = within ±{NOISE}",
        "- MassSpecGym (MoNA / MassBank / GNPS) overlaps our library, so the forward models may have trained on "
        "spectra of validation structures; these gains are optimistic, the decision is the leaderboard's",
        md_table(tab.round(3)), "### Head coverage", md_table(pd.DataFrame(cover).round(3)),
        f"- speed (this machine): {speed or 'scores reused'}", f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / "EXP_forward.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
