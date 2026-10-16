"""Fragment channel v2 (``casmi.channels.fragment2``) against v1 on split v5.

Per validation split: v2 scores for every (molecule, candidate) row of the feature table (mean over
the molecule's spectra) under a few scoring settings, then MRR@25 per class (C1 / C2 / NP C2) for the
channel alone, for analog + 0.3·z(fragment) (the combiner without fingerprint, kernel v12) and for
the same with the fingerprint (the Oct 18 combiner), v1 vs v2. Time per candidate is measured. Scores
go to ``outputs/fragment2/<split>_<grid>.parquet``; the report to ``notes/EXP_fragment_v2[_<grid>][_<split>].md``
(the split suffix for prefixes other than v5). A v2 setting whose analog + 0.3·z(fragment) gains
≥ ``--lb-gain`` NP C2 over v1 is written to ``outputs/fragment2[_<split>].json``.

    python validation/eval_fragment_v2.py [--features outputs/ranker_features_e5] [--prefix v5_seed]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.channels.fragment2 import Fragment2Config, explained_weighted, explanations, fragment_tree, ion_levels
from casmi.combiners import combine_rows
from casmi.config import REPO_ROOT, TOP_K, WORK_DIR
from casmi.harness import load_val
from casmi.channels.library import LibraryConfig
from casmi.pipeline import clean_queries, to_spectra
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

GRIDS = {
    "main": {"0.01 Da": Fragment2Config(),
             "10 ppm": Fragment2Config(tol_da=0.003, tol_ppm=10.0),
             "10 ppm, sqrt intensity": Fragment2Config(tol_da=0.003, tol_ppm=10.0, sqrt_intensity=True)},
    "ablation": {"depth 2 + losses": Fragment2Config(max_depth=2, tol_da=0.003, tol_ppm=10.0),
                 "depth 3, no losses": Fragment2Config(tol_da=0.003, tol_ppm=10.0, use_losses=False),
                 "depth 2, no losses": Fragment2Config(max_depth=2, tol_da=0.003, tol_ppm=10.0, use_losses=False)},
}
SETTINGS: dict[str, Fragment2Config] = {}
NOISE = {"C1": 0.04, "C2": 0.02, "NP C2": 0.02}
_SMILES: dict[str, str] = {}


def _tree_key(cfg: Fragment2Config) -> tuple:
    return cfg.max_depth, cfg.use_losses


def _explain(key: str) -> tuple[str, dict, float]:
    """Explanations of one candidate for every distinct tree setting of the grid."""
    t0 = time.perf_counter()
    out = {}
    for cfg in SETTINGS.values():
        tk = _tree_key(cfg)
        if tk not in out:
            out[tk] = explanations(fragment_tree(_SMILES[key], cfg), cfg)
    return key, out, (time.perf_counter() - t0) / len(out)


def _score(args: tuple) -> list[tuple[str, str, dict[str, float]]]:
    mol_id, specs, keys, expl = args
    out = []
    for k in keys:
        per = {}
        for name, cfg in SETTINGS.items():
            e = expl[k][_tree_key(cfg)]
            vals = [explained_weighted(ion_levels(e, mode, cfg), peaks, cfg) for mode, peaks in specs]
            per[name] = float(np.mean(vals)) if vals else 0.0
        out.append((mol_id, k, per))
    return out


def class_mrr(df: pd.DataFrame, score: np.ndarray, val: pd.DataFrame) -> dict:
    d = df[["molecule_id", "key", "label", "analog"]].assign(s=score)
    d = d.sort_values(["molecule_id", "s", "analog", "key"], ascending=[True, False, False, True])
    d["pos"] = d.groupby("molecule_id").cumcount() + 1
    hit = d[d["label"] & (d["pos"] <= TOP_K)].set_index("molecule_id")["pos"]
    rr = val["molecule_id"].map(1.0 / hit).fillna(0.0)
    c2 = val["class"] == 2
    return {"C1": rr[val["class"] == 1].mean(), "C2": rr[c2].mean(), "NP C2": rr[c2 & (val["source"] == NP_EXAMPLES)].mean()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("outputs/ranker_features_e5"))
    ap.add_argument("--prefix", default="v5_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lb-gain", type=float, default=0.02)
    ap.add_argument("--grid", choices=sorted(GRIDS), default="main")
    args = ap.parse_args()
    SETTINGS.update(GRIDS[args.grid])
    report_name = "EXP_fragment_v2" if args.grid == "main" else f"EXP_fragment_v2_{args.grid}"
    suffix = "" if args.prefix == "v5_seed" else f"_{args.prefix.removesuffix('_seed')}"
    report_name += suffix
    t0 = time.time()
    splits = [f"{args.prefix}{s}" for s in args.seeds]
    feats = {s: pd.read_parquet(REPO_ROOT / args.features / f"{s}.parquet") for s in splits}
    vals = {s: load_val(WORK_DIR / "splits" / s) for s in splits}
    _SMILES.update(dict(zip(vals[splits[0]].pool.meta["key"], vals[splits[0]].pool.meta["smiles"])))
    keys = sorted(set().union(*(set(f["key"]) for f in feats.values())))
    with Pool(args.workers) as p:
        res = p.map(_explain, keys, chunksize=64)
    expl = {k: e for k, e, _ in res}
    secs = np.array([s for _, _, s in res])
    print(f"{len(keys):,} candidates explained: {1000 * secs.mean():.1f} ms mean, p99 {1000 * np.percentile(secs, 99):.0f} "
          f"ms ({time.time() - t0:.0f}s)", flush=True)
    rows, out_dir = [], WORK_DIR / "fragment2"
    out_dir.mkdir(exist_ok=True)
    for split in splits:
        data, df = vals[split], feats[split]
        queries = clean_queries(data.queries)
        groups = {m: to_spectra(g, LibraryConfig()) for m, g in queries.groupby("molecule_id")}
        cand = df.groupby("molecule_id", sort=False)["key"].agg(list)
        jobs = [(m, groups.get(m, []), ks, {k: expl[k] for k in ks}) for m, ks in cand.items()]
        t1 = time.time()
        with Pool(args.workers) as p:
            scored = [r for part in p.map(_score, jobs, chunksize=4) for r in part]
        s2 = pd.DataFrame([{"molecule_id": m, "key": k, **{f"fragment2 {n}": v for n, v in per.items()}}
                           for m, k, per in scored])
        s2.to_parquet(out_dir / f"{split}_{args.grid}.parquet", index=False)
        df = df.merge(s2, on=["molecule_id", "key"], how="left").fillna({c: 0.0 for c in s2.columns[2:]})
        print(f"{split}: scored {len(s2):,} rows in {time.time() - t1:.0f}s", flush=True)
        val = data.val
        runs = {"analog only": combine_rows(df, {"kind": "analog"}), "fragment v1 alone": df["fragment"].to_numpy()}
        for col, tag in [("fragment", "v1"), *((f"fragment2 {n}", f"v2 {n}") for n in SETTINGS)]:
            if tag != "v1":
                runs[f"fragment {tag} alone"] = df[col].to_numpy()
            runs[f"analog + 0.3·z(fragment {tag})"] = combine_rows(
                df, {"kind": "topk", "k": 100, "alpha": 0.5, "fp": None, "extra": {col: 0.3}})
            runs[f"analog + FP + 0.3·z(fragment {tag})"] = combine_rows(
                df, {"kind": "topk", "k": 100, "alpha": 0.5, "extra": {col: 0.3}})
        for name, s in runs.items():
            rows.append({"split": split, "run": name, **class_mrr(df, s, val)})
    res = pd.DataFrame(rows)
    agg = res.groupby("run", sort=False)[["C1", "C2", "NP C2"]].agg(["mean", "std"])
    tab = pd.DataFrame(index=agg.index)
    for c in ("C1", "C2", "NP C2"):
        tab[c] = agg[(c, "mean")].map("{:.3f}".format) + " ± " + agg[(c, "std")].map("{:.3f}".format)
    base = {"analog + 0.3·z": agg.loc["analog + 0.3·z(fragment v1)"], "analog + FP + 0.3·z": agg.loc[
        "analog + FP + 0.3·z(fragment v1)"]}
    for c in ("C1", "NP C2"):
        col = []
        for run in agg.index:
            b = next((v for k, v in base.items() if run.startswith(k + "(")), None)
            d = agg.loc[run, (c, "mean")] - b[(c, "mean")] if b is not None else np.nan
            col.append("" if np.isnan(d) else f"{d:+.3f}" + ("" if abs(d) > NOISE[c] else " (n.m.)"))
        tab[f"Δ {c} vs v1 (same combiner)"] = col
    gains = {n: agg.loc[f"analog + 0.3·z(fragment v2 {n})", ("NP C2", "mean")]
             - agg.loc["analog + 0.3·z(fragment v1)", ("NP C2", "mean")] for n in SETTINGS}
    best = max(gains, key=gains.get)
    if gains[best] >= args.lb_gain:
        cfg = SETTINGS[best]
        (WORK_DIR / f"fragment2{suffix}.json").write_text(json.dumps(
            {"setting": best, "np_c2_gain_vs_v1": round(float(gains[best]), 4),
             "config": {"max_depth": cfg.max_depth, "use_losses": cfg.use_losses, "tol_da": cfg.tol_da,
                        "tol_ppm": cfg.tol_ppm, "sqrt_intensity": cfg.sqrt_intensity}}, indent=2) + "\n")
        verdict = (f"**v2 {best}**: NP C2 {gains[best]:+.3f} over v1 (analog + 0.3·z) → LB queue "
                   f"(outputs/fragment2{suffix}.json)")
    else:
        verdict = f"best v2 setting {best}: NP C2 {gains[best]:+.3f} over v1 < {args.lb_gain} → not queued"
    report = "\n\n".join([
        f"# Fragment channel v2 vs v1 ({', '.join(splits)})",
        f"- v2: fragment tree to depth 3 (bridges / same-ring pairs), neutral losses with a composition check, "
        f"best explanation per peak, intensity-weighted; {len(keys):,} candidates, {1000 * secs.mean():.1f} ms each "
        f"(p99 {1000 * np.percentile(secs, 99):.0f} ms, one core)",
        "- combiners: analog top-100 re-rank 0.5·z(analog) [+ 0.5·z(FP v2 epoch 5)] + 0.3·z(fragment)",
        "### MRR@25, mean ± std over seeds", md_table(tab.reset_index()),
        f"- {verdict}", f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"{report_name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
