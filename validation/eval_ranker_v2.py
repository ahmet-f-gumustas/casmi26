"""Ranker v2 (learned) vs the rank average, from ``outputs/ranker_features`` (``ranker_features.py``).

Training rows: the ranker-training splits r_seed10–14 (simulated validation sets with their own
library masks). Model: HistGradientBoostingClassifier on (molecule, candidate) rows, label = the
candidate is the answer; 4 bags, each a bootstrap over molecules (grouped by scorer key) with its
own seed; the score is the mean predicted probability.

1. Grouped 5-fold CV on the training splits (folds by answer key).
2. Validation: for each v2 seed, the bags are trained on training rows whose answer key is not
   in that seed's validation set, then rank its candidates. MRR@25 per class (all validation
   molecules, a molecule without its answer in the window scores 0), mean ± std over seeds, and
   the difference to the rank average with the noise labels (total ±0.015, C1 ±0.04, C2 ±0.02).
Feature sets: all; without popularity; without popularity and source flags. The training rows are
written to ``outputs/ranker_train/ranker_train.parquet`` for the notebook, which fits the ranker in place.

    python validation/eval_ranker_v2.py [--bags 4]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from casmi.config import REPO_ROOT, TOP_K, WORK_DIR
from casmi.ranker_v2 import FEATURE_SETS, PARAMS, fit_bags, predict
from casmi.report import md_table

TRAIN = [f"r_seed{s}" for s in range(10, 15)]
VAL = [f"v2_seed{s}" for s in range(3)]
NOISE = {"total": 0.015, "C1": 0.04, "C2": 0.02}


def load(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(WORK_DIR / "ranker_features" / f"{split}.parquet")
    val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
    df = df.merge(val[["molecule_id", "inchikey14"]].rename(columns={"inchikey14": "answer"}), on="molecule_id")
    return df, val


def class_mrr(df: pd.DataFrame, score: np.ndarray, val: pd.DataFrame) -> dict:
    """MRR@25 per class over every validation molecule; ties broken by analog score, then key."""
    d = df[["molecule_id", "key", "label", "analog"]].assign(s=score)
    d = d.sort_values(["molecule_id", "s", "analog", "key"], ascending=[True, False, False, True])
    d["pos"] = d.groupby("molecule_id").cumcount() + 1
    hit = d[d["label"] & (d["pos"] <= TOP_K)].set_index("molecule_id")["pos"]
    rr = val["molecule_id"].map(1.0 / hit).fillna(0.0)
    out = {f"C{c}": float(rr[val["class"] == c].mean()) for c in (1, 2, 3)}
    np_mask = (val["class"] == 2) & (val["source"] == "enveda-np-examples")
    out.update(total=float(rr.mean()), **{"C2 NP": float(rr[np_mask].mean())})
    return out


def rank_average(df: pd.DataFrame) -> np.ndarray:
    return (df["analog_pct"] + df["fingerprint_pct"]).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    train = [load(s) for s in TRAIN]
    tr = pd.concat([d for d, _ in train], ignore_index=True)
    tr_val = pd.concat([v.assign(molecule_id=s + "/" + v["molecule_id"]) for s, (_, v) in zip(TRAIN, train)],
                       ignore_index=True)
    tr["molecule_id"] = tr["split"] + "/" + tr["molecule_id"]
    ship = WORK_DIR / "ranker_train"
    ship.mkdir(parents=True, exist_ok=True)
    tr[["molecule_id", "answer", "label", *FEATURE_SETS["all"]]].to_parquet(ship / "ranker_train.parquet", index=False)

    cv_rows = []
    for name, feats in FEATURE_SETS.items():
        oof = np.zeros(len(tr))
        for fit_idx, test_idx in GroupKFold(n_splits=5).split(tr, groups=tr["answer"]):
            models = fit_bags(tr.iloc[fit_idx], feats, args.bags)
            oof[test_idx] = predict(models, tr.iloc[test_idx], feats)
        cv_rows.append({"run": f"ranker v2 [{name}]", **class_mrr(tr, oof, tr_val)})
    cv_rows.append({"run": "rank average (analog + fingerprint)", **class_mrr(tr, rank_average(tr), tr_val)})

    rows = []
    for split in VAL:
        df, val = load(split)
        rows.append({"run": "rank average (analog + fingerprint)", "seed": split, **class_mrr(df, rank_average(df), val)})
        rows.append({"run": "analog only", "seed": split, **class_mrr(df, df["analog_pct"].to_numpy(), val)})
        fit_df = tr[~tr["answer"].isin(set(val["inchikey14"]))]
        for name, feats in FEATURE_SETS.items():
            models = fit_bags(fit_df, feats, args.bags)
            rows.append({"run": f"ranker v2 [{name}]", "seed": split, **class_mrr(df, predict(models, df, feats), val)})
        print(f"{split} done ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    agg = res.groupby("run", sort=False)[["C1", "C2", "C2 NP", "total"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    base = agg.loc["rank average (analog + fingerprint)"]

    def verdict(row: pd.Series, col: str) -> str:
        d = row[f"{col}_mean"] - base[f"{col}_mean"]
        return f"{d:+.3f}" + ("" if abs(d) > NOISE[col] else " (n.m.)")

    for col in ("total", "C1", "C2"):
        agg[f"Δ {col} vs rank avg"] = agg.apply(verdict, axis=1, col=col)
    agg = agg.reset_index()
    best = agg[agg["run"].str.startswith("ranker v2")].sort_values("total_mean", ascending=False).iloc[0]
    gain = best["total_mean"] - base["total_mean"]
    decision = "ranker v2" if gain >= NOISE["total"] else "rank average"
    (WORK_DIR / "ranker_v2.json").write_text(json.dumps({
        "decision": decision, "best": best["run"], "gain_total": round(float(gain), 4),
        "features": FEATURE_SETS[best["run"].split("[")[1].rstrip("]")], "params": PARAMS, "bags": args.bags,
    }, indent=2) + "\n")
    report = "\n\n".join([
        "# Ranker v2 (learned) vs rank average",
        f"- training: {', '.join(TRAIN)} ({tr['molecule_id'].nunique()} molecules, {len(tr):,} rows); "
        f"{args.bags} bags, HistGradientBoosting {PARAMS}",
        "### Grouped 5-fold CV on the training splits (pooled molecules)", md_table(pd.DataFrame(cv_rows)),
        "### Validation (v2 seeds), MRR@25 mean ± std over seeds; n.m. = not measurable", md_table(agg),
        f"- best ranker: **{best['run']}**, Δ total {gain:+.3f} → decision: **{decision}** "
        f"(rule: keep the rank average unless the gain is ≥ {NOISE['total']})",
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / "EXP_ranker_v2.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
