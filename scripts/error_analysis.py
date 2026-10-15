"""Class 2 error analysis of the analog-only pipeline on split v2.

Per validation C2 molecule (every seed):

- ``rank``: the answer's rank in the full analog ranking (``rr`` = 1/rank within 25, else 0),
  ``n_candidates`` in its mass window and ``n_close`` candidates with Tanimoto > 0.8 to the answer;
- ``t_top1``: Tanimoto(top-1 candidate, answer) when the top-1 is wrong, ``t_wrong1`` the same for
  the best-ranked wrong candidate (also defined for hits);
- ``t_analog1``: Tanimoto(answer, structure of the best spectral analog), ``t_analog10`` the best
  over the top-10 analogs of each spectrum; ``sim1`` that analog's spectral similarity;
- query side: number of spectra, mean collision energy, mean peak count (after the test peak rule),
  instrument; ``nn_tanimoto`` (nearest library structure) from the split; ``tims_share10``: share of
  timsTOF spectra among the retrieved top-10 analogs (the library mixes instruments).

Failures (answer not at rank 1) are put in two clusters and a remainder:

- (a) near-isomer confusion: ``t_top1`` > 0.8;
- (b) retrieval failure: ``t_top1`` < 0.5 and ``t_analog10`` < 0.5 (no retrieved analog resembles the
  answer). Split further by ``nn_tanimoto``: a library analog exists (≥ 0.5) but was not retrieved
  (spectral mismatch), or none exists;
- (c) intermediate: the rest.

    python scripts/error_analysis.py [--prefix v2_seed] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from casmi.channels.analog import AnalogIndex, library_row_keys
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.errors import ANALOG_FAR, CLOSE, FAR, NN_EXISTS, c2_error_rows, row_instruments
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val
from casmi.pipeline import PipelineConfig, analog_channel, analog_hits, candidate_windows, clean_queries
from casmi.report import md_table


def analyse_split(split: str, index: AnalogIndex, row_key: dict[int, str], row_tims: dict[int, bool],
                  workers: int) -> pd.DataFrame:
    data = load_val(WORK_DIR / "splits" / split)
    queries = clean_queries(data.queries).reset_index(drop=True)
    windows = candidate_windows(queries, data.pool, PipelineConfig())
    allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    hits = analog_hits(queries, index, allowed=allowed, workers=workers)
    scores = analog_channel(queries, windows, data.pool, index, row_key, allowed=allowed, hits=hits)
    return c2_error_rows(split, data.val, queries, windows, scores, hits, data.pool.meta, data.pool.fps, row_key,
                         row_tims)


def cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    n_miss = int((df["rank"] != 1).sum())
    g = df.groupby("cluster")
    tab = pd.DataFrame({
        "n": g.size(), "share of C2": g.size() / len(df),
        "share of failures": g.size().where(g.size().index != "hit (rank 1)") / max(n_miss, 1),
        "MRR": g["rr"].mean(), "answer in top-25": g["rr"].apply(lambda s: float((s > 0).mean())),
        "median rank (scored)": g["rank"].apply(lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan),
        "t_top1": g["t_top1"].median(), "t_analog10": g["t_analog10"].median(), "sim1": g["sim1"].median(),
        "nn_tanimoto": g["nn_tanimoto"].median(), "n_candidates": g["n_candidates"].median(),
        "n_close": g["n_close"].median(), "tims_share10": g["tims_share10"].mean(), "n_spectra": g["n_spectra"].median(), "ce": g["ce"].median(),
        "n_peaks": g["n_peaks"].median(),
    })
    return tab.reset_index()


def seed_shares(df: pd.DataFrame) -> pd.DataFrame:
    per = df.groupby(["split", "cluster"]).size().unstack(fill_value=0)
    per = per.div(per.sum(axis=1), axis=0)
    return pd.DataFrame({"cluster": per.columns, "share mean": per.mean().to_numpy(),
                         "share std": per.std().to_numpy()})


def bin_grid(df: pd.DataFrame) -> pd.DataFrame:
    miss = df[df["rank"] != 1]
    tb = pd.cut(miss["t_top1"].fillna(-1), [-1.01, -0.5, 0.3, 0.5, 0.8, 1.0],
                labels=["no top-1", "<0.3", "0.3–0.5", "0.5–0.8", ">0.8"])
    ab = pd.cut(miss["t_analog10"], [-0.01, 0.3, 0.5, 0.7, 1.0], labels=["<0.3", "0.3–0.5", "0.5–0.7", ">0.7"])
    return pd.crosstab(tb, ab, margins=True).reset_index().rename(columns={"t_top1": "t_top1 \\ t_analog10"})


def feature_contrast(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["sim1", "t_analog1", "t_analog10", "nn_tanimoto", "n_candidates", "n_close", "tims_share10", "n_spectra",
            "ce", "n_peaks"]
    hit, miss = df[df["rank"] == 1], df[df["rank"] != 1]
    out = []
    for c in cols:
        out.append({"feature": c, "hit median": hit[c].median(), "failure median": miss[c].median(),
                    "Spearman(rr, ·)": df["rr"].corr(df[c], method="spearman")})
    return pd.DataFrame(out)


def section(df: pd.DataFrame, title: str) -> list[str]:
    return [f"## {title} (n = {len(df)} molecule-seeds, MRR {df['rr'].mean():.3f}, hits@1 {(df['rank'] == 1).mean():.1%})",
            "### Clusters", md_table(cluster_table(df)),
            "### Cluster share per seed (mean ± std over seeds)", md_table(seed_shares(df)),
            "### Failures: t_top1 × t_analog10 (counts)", md_table(bin_grid(df)),
            "### Hits vs failures", md_table(feature_contrast(df))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="v2_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    row_tims = row_instruments(CLEAN_LIBRARY)
    parts = []
    for seed in args.seeds:
        parts.append(analyse_split(f"{args.prefix}{seed}", index, row_key, row_tims, args.workers))
        print(f"{args.prefix}{seed} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(WORK_DIR / f"error_analysis_{args.prefix}.parquet", index=False)
    np_df = df[df["source"] == "enveda-np-examples"]
    by_inst = df.groupby("instrument").agg(n=("rr", "size"), MRR=("rr", "mean"),
                                           near_isomer=("cluster", lambda s: float((s == "(a) near-isomer").mean())),
                                           retrieval=("cluster", lambda s: float(s.str.startswith("(b").mean())))
    report = "\n\n".join([
        f"# Class 2 error analysis, analog only ({args.prefix} {args.seeds})",
        f"- Clusters over failures (answer not rank 1): (a) t_top1 > {CLOSE}; (b) t_top1 < {FAR} and t_analog10 < "
        f"{ANALOG_FAR}, split by nn_tanimoto ≥ {NN_EXISTS} (b1) / < {NN_EXISTS} (b2); (c) the rest.",
        "- t_top1: Tanimoto(top-1 candidate, answer). t_analog10: best Tanimoto(answer, top-10 spectral analogs of "
        "any spectrum). sim1: best analog spectral similarity. Morgan r2 2048 on the stereo-free canonical tautomer.",
        *section(df, "All Class 2"),
        *section(np_df, "NP subset (enveda-np-examples)"),
        "## By query instrument", md_table(by_inst.reset_index()),
        f"- per-molecule rows: outputs/error_analysis_{args.prefix}.parquet; runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    name = "ERROR_ANALYSIS.md" if args.prefix == "v2_seed" else f"ERROR_ANALYSIS_{args.prefix}.md"
    (REPO_ROOT / "notes" / name).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
