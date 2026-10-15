"""Recall diagnosis without the leaderboard → ``notes/DIAGNOSIS.md``.

a. Test columns: the 12 columns of the (placeholder) ``test.parquet`` against the train schema;
   the placeholder rows are enveda-180 training spectra, so joining them back to train shows what
   the train-only columns (``adduct_orig``, ``precursor_error_ppm``) would say about test-like rows.
   Also the per-molecule adduct mix and neutral-mass agreement, which bound the multi-adduct idea.
b. Base rate: share of Class 2 window candidates with library spectra (split mask), against the
   share among analog-only wrong top-1s. A large gap means the analog channel favours structures
   with library spectra (self-match bias).
c. Self-match test: the analog channel without each candidate's own spectra (``self_min_sim`` > 1),
   alone and with Class 1 protected by the library gate at τ = 0.8 (switch / two-stage / boost, see
   ``casmi.ranker``); and the per-candidate variant where a candidate's own spectra count only at an
   analog similarity ≥ s (s ∈ {0.7, 0.8, 0.9}). MRR@25 per class and on the NP subset, mean ± std
   over seeds; noise labels C1 ±0.04, C2 ±0.02, total ±0.015.

Library max similarity and ``in_library`` come from ``outputs/ranker_features/<split>.parquet``.

    python scripts/recall_diagnosis.py [--prefixes v2_seed v3_seed] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.channels.analog import AnalogConfig, AnalogIndex, library_row_keys
from casmi.chem import neutral_mass
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val, reciprocal_ranks
from casmi.pipeline import PipelineConfig, analog_channel, analog_hits, candidate_windows, clean_queries
from casmi.report import md_table

DATA = REPO_ROOT / "data"
NOISE = {"C1": 0.04, "C2": 0.02, "C2 NP": 0.02, "total": 0.015}
TAU = 0.8
CONFIGS = {
    "analog (with self match)": {"mode": "weighted", "weights": {"analog": 1.0}},
    "analog without self match": {"mode": "weighted", "weights": {"analog_noself": 1.0}},
    **{f"no self + gate τ={TAU} ({rule})": {"mode": "protect", "tau": TAU, "weights": {"analog_noself": 1.0},
                                             "rule": rule} for rule in ("switch", "two_stage", "boost")},
    **{f"own spectra only at sim ≥ {s}": {"mode": "weighted", "weights": {f"analog_self{s}": 1.0}}
       for s in (0.7, 0.8, 0.9)},
}


def test_columns() -> str:
    test = pd.read_parquet(DATA / "test.parquet")
    schema = pq.read_schema(DATA / "train.parquet").names
    rows = []
    for c in test.columns:
        v = test[c]
        listy = not isinstance(v.iloc[0], (str, float, int, np.floating, np.integer))
        rows.append({"column": c, "dtype": "list" if listy else str(v.dtype),
                     "n unique": "" if listy else v.nunique(),
                     "example": f"len {len(v.iloc[0])}" if listy else str(v.iloc[0])[:30],
                     "in train": c in schema})
    cols = pd.DataFrame(rows)
    missing = [c for c in schema if c not in test.columns]

    # placeholder rows back to their training rows (exact precursor + adduct + base peak intensity)
    lib = pd.read_parquet(CLEAN_LIBRARY, columns=["ingest_lib", "adduct", "adduct_orig", "precursor_mz",
                                                  "precursor_error_ppm", "base_peak_intensity", "key"])
    j = test.merge(lib, on=["precursor_mz", "adduct", "base_peak_intensity"], how="left")
    matched = j.dropna(subset=["key"]).drop_duplicates("spectrum_id")
    same = (matched["adduct_orig"] == matched["adduct"]).mean()
    ppm = matched["precursor_error_ppm"].abs()

    # adduct mix and neutral-mass agreement per molecule
    t = test.assign(mass=[neutral_mass(p, a) for p, a in zip(test["precursor_mz"], test["adduct"])])
    per = t.groupby("molecule_id").agg(n_adducts=("adduct", "nunique"), n=("adduct", "size"),
                                       lo=("mass", "min"), hi=("mass", "max"), mid=("mass", "median"))
    spread = (per["hi"] - per["lo"]) / per["mid"] * 1e6
    multi = per["n_adducts"] >= 2
    adducts = test["adduct"].value_counts().rename_axis("adduct").reset_index(name="spectra")
    return "\n\n".join([
        "## a. Test columns",
        f"- `test.parquet`: {test.shape[0]:,} rows × {test.shape[1]} columns, {test['molecule_id'].nunique()} molecules",
        md_table(cols),
        f"- train columns missing from test: {', '.join(f'`{c}`' for c in missing)}",
        "- **No structure-derived column**: no SMILES / InChIKey / formula, no `precursor_error_ppm` (needs the "
        "true structure) and no `adduct_orig`. `molecule_id` / `spectrum_id` are random hex ids. "
        "`base_peak_intensity` is the absolute intensity of the base peak (acquisition, not structure); it only "
        "identifies the placeholder rows (they are training spectra).",
        f"- placeholder rows matched back to train: {len(matched):,} / {len(test):,} spectra "
        f"({', '.join(f'{k} {v}' for k, v in matched['ingest_lib'].value_counts().items())}); "
        f"`adduct_orig` == `adduct` for {same:.1%}; |precursor_error_ppm| median {ppm.median():.2f}, "
        f"p99 {ppm.quantile(0.99):.2f}, max {ppm.max():.2f}",
        "### Adducts in the placeholder test", md_table(adducts),
        f"- molecules with ≥ 2 adduct types: {multi.sum()} / {len(per)} ({multi.mean():.0%}); their neutral masses "
        f"agree to median {spread[multi].median():.2f} ppm (p99 {spread[multi].quantile(0.99):.2f}, max "
        f"{spread[multi].max():.2f}). Single-adduct molecules: {(~multi).sum()}",
    ])


def top1(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Top-1 row per molecule by ``score`` (ties: key)."""
    return df.sort_values(["molecule_id", score, "key"], ascending=[True, False, True]).groupby(
        "molecule_id", sort=False).head(1)


def base_rate(splits: list[str]) -> str:
    rows = []
    for split in splits:
        f = pd.read_parquet(WORK_DIR / "ranker_features" / f"{split}.parquet",
                            columns=["molecule_id", "key", "analog", "in_library", "label", "class", "source"])
        for subset, d in (("all C2", f[f["class"] == 2]), ("NP C2", f[(f["class"] == 2)
                                                                        & (f["source"] == "enveda-np-examples")])):
            t1 = top1(d[d["analog"] > 0], "analog")
            wrong = t1[~t1["label"]]
            non_answer = d[~d["label"]]
            rows.append({"split": split, "subset": subset, "molecules": d["molecule_id"].nunique(),
                         "window candidates in library (pooled)": non_answer["in_library"].mean(),
                         "per-molecule mean": non_answer.groupby("molecule_id")["in_library"].mean().mean(),
                         "analog top-1 in library (all)": t1["in_library"].mean(),
                         "analog wrong top-1 in library": wrong["in_library"].mean(),
                         "wrong top-1 / molecules": len(wrong) / max(1, d["molecule_id"].nunique())})
    df = pd.DataFrame(rows)
    df["prefix"] = df["split"].str.replace(r"\d+$", "", regex=True)
    agg = df.groupby(["prefix", "subset"], sort=False).mean(numeric_only=True).reset_index()
    return "\n\n".join([
        "## b. Base rate: library structures among Class 2 window candidates",
        "- in-library share among the non-answer window candidates (what a random wrong top-1 would show) vs "
        "the share among the analog channel's wrong top-1s (the 81 % of EXP-017)",
        md_table(agg.drop(columns=["molecules"])),
        "### Per split", md_table(df.drop(columns=["prefix"])),
    ])


def self_match(splits: list[str], workers: int) -> tuple[str, pd.DataFrame]:
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    rows, shares = [], []
    for split in splits:
        data = load_val(WORK_DIR / "splits" / split)
        queries = clean_queries(data.queries).reset_index(drop=True)
        windows = candidate_windows(queries, data.pool, PipelineConfig())
        allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
        allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
        hits = analog_hits(queries, index, allowed=allowed, workers=workers)
        f = pd.read_parquet(WORK_DIR / "ranker_features" / f"{split}.parquet",
                            columns=["molecule_id", "key", "library_max", "in_library"])
        channels = {
            "analog": analog_channel(queries, windows, data.pool, index, row_key, hits=hits),
            "analog_noself": analog_channel(queries, windows, data.pool, index, row_key,
                                            cfg=AnalogConfig(self_min_sim=2.0), hits=hits),
            **{f"analog_self{s}": analog_channel(queries, windows, data.pool, index, row_key,
                                                 cfg=AnalogConfig(self_min_sim=s), hits=hits) for s in (0.7, 0.8, 0.9)},
            "library_max": {m: dict(zip(g["key"], g["library_max"])) for m, g in f.groupby("molecule_id")},
        }
        in_lib = set(f.loc[f["in_library"] > 0, "key"])
        cls, is_np = data.val["class"].to_numpy(), (data.val["source"] == "enveda-np-examples").to_numpy()
        for name, cfg in CONFIGS.items():
            rr = data.val["molecule_id"].map(reciprocal_ranks(windows, channels, cfg, data.val)).fillna(0).to_numpy()
            rows.append({"run": name, "split": split, "C1": rr[cls == 1].mean(), "C2": rr[cls == 2].mean(),
                         "C2 NP": rr[(cls == 2) & is_np].mean(), "total": rr.mean()})
        for name in ("analog", "analog_noself", "analog_self0.8"):
            c2 = data.val.loc[data.val["class"] == 2]
            t1 = [max(channels[name].get(m, {}).items(), key=lambda kv: (kv[1], kv[0]), default=(None, 0))
                  for m in c2["molecule_id"]]
            wrong = [k for (k, s), a in zip(t1, c2["inchikey14"]) if k is not None and s > 0 and k != a]
            shares.append({"split": split, "channel": name, "wrong top-1 in library": np.mean([k in in_lib
                                                                                                for k in wrong])})
        print(f"{split} done", flush=True)
    res = pd.DataFrame(rows)
    res["prefix"] = res["split"].str.replace(r"\d+$", "", regex=True)
    parts = ["## c. Self-match test (analog channel without each candidate's own spectra)"]
    for prefix, d in res.groupby("prefix", sort=False):
        agg = d.groupby("run", sort=False)[["C1", "C2", "C2 NP", "total"]].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        base = agg.iloc[0]
        for col in ("C1", "C2", "C2 NP", "total"):
            delta = agg[f"{col}_mean"] - base[f"{col}_mean"]
            agg[f"Δ {col}"] = [f"{x:+.3f}" + ("" if abs(x) > NOISE[col] else " (n.m.)") for x in delta]
        parts += [f"### {prefix} {sorted(d['split'].unique())}", md_table(agg.reset_index())]
    sh = pd.DataFrame(shares)
    sh["prefix"] = sh["split"].str.replace(r"\d+$", "", regex=True)
    parts += ["### Class 2 wrong top-1s with library spectra", md_table(
        sh.groupby(["prefix", "channel"], sort=False)["wrong top-1 in library"].mean().reset_index())]
    parts.append(f"- noise labels: C1 ±{NOISE['C1']}, C2 / C2 NP ±{NOISE['C2']}, total ±{NOISE['total']}; "
                 "Δ vs analog with self match")
    return "\n\n".join(parts), res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefixes", nargs="+", default=["v2_seed", "v3_seed"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    splits = [f"{p}{s}" for p in args.prefixes for s in args.seeds]
    sections = [test_columns(), base_rate(splits)]
    sm, res = self_match(splits, args.workers)
    res.to_parquet(WORK_DIR / "self_match.parquet", index=False)
    report = "\n\n".join(["# Recall diagnosis (item 1)", *sections, sm,
                          f"- runtime {time.time() - t0:.0f}s"]) + "\n"
    (REPO_ROOT / "notes" / "DIAGNOSIS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
