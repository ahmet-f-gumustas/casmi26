"""Split v4 calibration: which nn_tanimoto profile gives the LB-implied analog-only Class 2 MRR?

The public LB of analog only (0.285) with Class 1 at its validation level implies a Class 2 MRR of
about 0.32–0.34. Per profile (Class 2 shares for nn_tanimoto 0–0.3 / 0.3–0.5 / 0.5–0.7 / 0.7+) and
seed, ``make_split.py --v4-from v3_seed<N> --profile ...`` builds ``splits/v4scan_<profile>_seed<N>``
(skipped when present); the analog channel alone is then scored per class, with the Class 2 failure
clusters (``casmi.errors``), the Class 2 window size and the library size. Split v3 is the reference row.
The chosen profile (analog-only C2 within ``--target`` ± ``--tol``, closest first) goes to
``outputs/v4_profile.json``; the report to ``notes/V4_CALIBRATION.md``.

    python validation/calibrate_v4.py [--profiles 40,45,15,0 50,40,10,0 60,35,5,0] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from casmi.channels.analog import AnalogIndex, library_row_keys
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.errors import CLUSTERS, c2_error_rows, row_instruments
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val, reciprocal_ranks
from casmi.pipeline import PipelineConfig, analog_channel, analog_hits, candidate_windows, clean_queries
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

PROFILES = ["40,45,15,0", "50,40,10,0", "60,35,5,0", "75,25,0,0", "100,0,0,0"]
SHORT = {c: c.split(")")[0] + ")" if c.startswith("(") else "hit" for c in CLUSTERS}


def build(profile: str, seed: int) -> str:
    name = f"v4scan_{profile.replace(',', '-')}_seed{seed}"
    out = WORK_DIR / "splits" / name
    if not (out / "val_ids.parquet").exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "validation" / "make_split.py"), "--v4-from", f"v3_seed{seed}",
                        "--profile", *profile.split(","), "--out-dir", str(out)], check=True, capture_output=True)
    return name


def score(split: str, index: AnalogIndex, row_key: dict, row_tims: dict, workers: int) -> dict:
    data = load_val(WORK_DIR / "splits" / split)
    queries = clean_queries(data.queries).reset_index(drop=True)
    windows = candidate_windows(queries, data.pool, PipelineConfig())
    allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    hits = analog_hits(queries, index, allowed=allowed, workers=workers)
    scores = analog_channel(queries, windows, data.pool, index, row_key, allowed=allowed, hits=hits)
    rr = reciprocal_ranks(windows, {"analog": scores}, {"analog": 1.0}, data.val)
    val = data.val.set_index("molecule_id")
    cls, src = val["class"], val["source"]
    c2 = cls[cls == 2].index
    err = c2_error_rows(split, data.val, queries, windows, scores, hits, data.pool.meta, data.pool.fps, row_key,
                        row_tims)
    shares = err["cluster"].value_counts(normalize=True)
    np_c2 = [m for m in c2 if src[m] == NP_EXAMPLES]
    return {"C1": rr[cls[cls == 1].index].mean(), "C2": rr[c2].mean(), "NP C2": rr[np_c2].mean(),
            "C2 window median": float(np.median([len(windows.get(m, [])) for m in c2])),
            "library spectra (M)": data.in_library.sum() / 1e6,
            **{f"share {SHORT[c]}": float(shares.get(c, 0.0)) for c in CLUSTERS},
            **{f"MRR {SHORT[c]}": float(err.loc[err["cluster"] == c, "rr"].mean()) for c in CLUSTERS}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", nargs="+", default=PROFILES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--target", type=float, default=0.34)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    row_tims = row_instruments(CLEAN_LIBRARY)
    rows = []
    for profile in ["v3", *args.profiles]:
        for seed in args.seeds:
            split = f"v3_seed{seed}" if profile == "v3" else build(profile, seed)
            rows.append({"profile": profile, "seed": seed, **score(split, index, row_key, row_tims, args.workers)})
            r = rows[-1]
            print(f"{profile} seed {seed}: C1 {r['C1']:.3f} C2 {r['C2']:.3f} NP C2 {r['NP C2']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(WORK_DIR / "v4_calibration.parquet", index=False)
    cols = [c for c in df.columns if c not in ("profile", "seed")]
    agg = df.groupby("profile", sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["|C2 − target|"] = (agg["C2_mean"] - args.target).abs()
    ok = agg[(agg["profile"] != "v3") & (agg["|C2 − target|"] <= args.tol)].sort_values("|C2 − target|")
    main_cols = ["profile", "C1_mean", "C1_std", "C2_mean", "C2_std", "NP C2_mean", "NP C2_std", "|C2 − target|",
                 "C2 window median_mean", "library spectra (M)_mean"]
    cluster_cols = ["profile", *[f"share {SHORT[c]}_mean" for c in CLUSTERS], *[f"MRR {SHORT[c]}_mean" for c in CLUSTERS]]
    if len(ok):
        best = ok.iloc[0]
        choice = {"profile": [float(x) for x in best["profile"].split(",")], "C1": round(best["C1_mean"], 4),
                  "C2": round(best["C2_mean"], 4), "NP C2": round(best["NP C2_mean"], 4), "target": args.target}
        (WORK_DIR / "v4_profile.json").write_text(json.dumps(choice, indent=2) + "\n")
        verdict = f"choice: **{best['profile']}** (analog-only C2 {best['C2_mean']:.3f}) → outputs/v4_profile.json"
    else:
        verdict = f"no profile within {args.target} ± {args.tol}; nothing written"
    report = "\n\n".join([
        f"# Split v4 calibration (analog only; seeds {args.seeds}; target C2 {args.target} ± {args.tol})",
        "- profile: Class 2 shares for nn_tanimoto 0–0.3 / 0.3–0.5 / 0.5–0.7 / 0.7+, reached by removing each "
        "answer's library neighbours at or above its bin's upper edge (on top of the v3 decoys)",
        "### MRR@25 per class, mean ± std over seeds", md_table(agg[main_cols]),
        "### Class 2 failure clusters of analog only (share of C2 molecules, MRR inside the cluster)",
        md_table(agg[cluster_cols]),
        f"- {verdict}", f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / "V4_CALIBRATION.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
