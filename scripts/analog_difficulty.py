"""Analog difficulty of validation Class 2 molecules, and the target profile implied by the LB.

For every C2 molecule of each split seed: ``nn_tanimoto`` (nearest library structure,
its own excluded) and the number of analogs (Tanimoto ≥ 0.6, ±200 Da), next to the
analog channel's reciprocal rank. C2 MRR is reported per nn_tanimoto bin. The same
nn_tanimoto distribution is computed for enveda-np-examples (the test proxy, each
molecule against the full library minus itself). Finally the bin mixture closest to the
NP-proxy profile (exponential tilt, minimum KL) whose C2 MRR equals the LB-implied
value is solved for; it is saved as the target profile for split v2.

    python scripts/analog_difficulty.py [--seeds 0 1 2] [--target 0.32]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.candidates import CandidatePool
from casmi.channels.analog import library_row_keys
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.difficulty import BIN_LABELS, nn_bin, nn_tanimoto, tilt_to_target
from casmi.harness import CLEAN_LIBRARY, compute_channels, library_keys, load_val, reciprocal_ranks
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

PROFILE = WORK_DIR / "difficulty_profile.json"


def structure_arrays(pool: CandidatePool, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    row = dict(zip(pool.meta["key"], range(len(pool.meta))))
    idx = np.array([row[k] for k in keys])
    return np.asarray(pool.fps[idx]), pool.meta["exact_mass"].to_numpy()[idx]


def difficulty(pool: CandidatePool, query_keys: list[str], ref_keys: list[str]) -> pd.DataFrame:
    qf, qm = structure_arrays(pool, query_keys)
    rf, rm = structure_arrays(pool, ref_keys)
    nn, n_analogs = nn_tanimoto(qf, query_keys, qm, rf, ref_keys, rm)
    return pd.DataFrame({"key": query_keys, "nn_tanimoto": nn, "n_analogs": n_analogs})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", type=Path, default=WORK_DIR / "splits")
    ap.add_argument("--prefix", default="seed", help="seed (split v1, writes the target profile) or v2_seed")
    ap.add_argument("--target", type=float, default=0.32, help="LB-implied C2 MRR@25")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()

    pool = CandidatePool.load()
    pool_keys = set(pool.meta["key"])
    rows = []
    for seed in args.seeds:
        data = load_val(args.splits / f"{args.prefix}{seed}")
        windows, channels = compute_channels(data, channels=("analog",), workers=args.workers)
        rr = reciprocal_ranks(windows, channels, {"analog": 1.0}, data.val)
        c2 = data.val[data.val["class"] == 2]
        lib = sorted(library_keys(data.in_library) & pool_keys)
        d = difficulty(pool, c2["inchikey14"].tolist(), lib)
        rows.append(c2[["molecule_id", "inchikey14", "source"]].assign(seed=seed, rr=c2["molecule_id"].map(rr).to_numpy(),
                                                                         nn_tanimoto=d["nn_tanimoto"].to_numpy(),
                                                                         n_analogs=d["n_analogs"].to_numpy()))
        print(f"seed {seed}: {len(c2)} C2 molecules, library {len(lib):,} structures ({time.time() - t0:.0f}s)", flush=True)
    c2 = pd.concat(rows, ignore_index=True)
    c2["bin"] = np.array(BIN_LABELS)[nn_bin(c2["nn_tanimoto"].to_numpy())]

    # NP proxy: every NP example against the full library minus itself.
    row_key = library_row_keys(CLEAN_LIBRARY)
    all_lib = sorted(set(row_key.dropna()) & pool_keys)
    lib_df = pd.read_parquet(CLEAN_LIBRARY, columns=["ingest_lib", "key"])
    np_keys = sorted(set(lib_df.loc[lib_df["ingest_lib"] == NP_EXAMPLES, "key"].dropna()) & pool_keys)
    npd = difficulty(pool, np_keys, all_lib)
    npd["bin"] = np.array(BIN_LABELS)[nn_bin(npd["nn_tanimoto"].to_numpy())]
    npd.to_parquet(WORK_DIR / "nn_tanimoto_np_proxy.parquet", index=False)

    def per_bin(df: pd.DataFrame, name: str) -> pd.DataFrame:
        g = df.groupby("bin", observed=False)
        return pd.DataFrame({"n": g.size(), "share": g.size() / len(df), "c2_mrr": g["rr"].mean(),
                             "median_analogs": g["n_analogs"].median()}).reindex(BIN_LABELS).assign(group=name)

    tables = [per_bin(c2, "val C2 (all, 3 seeds)")] + [per_bin(c2[c2["source"] == s], f"val C2 [{s}]")
                                                      for s in sorted(c2["source"].unique())]
    by_bin = pd.concat(tables).reset_index(names="bin")
    pooled = tables[0]
    bin_mrr = pooled["c2_mrr"].fillna(0.0).to_numpy()
    np_shares = npd["bin"].value_counts(normalize=True).reindex(BIN_LABELS).fillna(0.0).to_numpy()
    v1_shares = pooled["share"].to_numpy()

    solutions = []
    for miss in (0.0, 0.2, 0.4):  # share of hidden C2 answers missing from the pool
        target = args.target / (1 - miss)
        try:
            profile, lam = tilt_to_target(np_shares, bin_mrr, target)
            solutions.append({"pool_miss_share": miss, "in_pool_c2_target": target, "lambda": lam,
                              **dict(zip(BIN_LABELS, profile.round(3)))})
        except ValueError as err:
            solutions.append({"pool_miss_share": miss, "in_pool_c2_target": target, "lambda": np.nan,
                              **dict.fromkeys(BIN_LABELS, np.nan), "note": str(err)})
    sol = pd.DataFrame(solutions)
    main_profile = sol.iloc[0][list(BIN_LABELS)].to_numpy(dtype=float)
    if args.prefix == "seed":
        PROFILE.write_text(json.dumps({
            "bins": BIN_LABELS, "edges": [0.0, 0.3, 0.5, 0.7, 1.0], "target_c2_mrr": args.target,
            "profile": dict(zip(BIN_LABELS, main_profile.round(4).tolist())),
            "bin_c2_mrr_v1": dict(zip(BIN_LABELS, bin_mrr.round(4).tolist())),
            "np_proxy_shares": dict(zip(BIN_LABELS, np_shares.round(4).tolist())),
        }, indent=2) + "\n")

    expected = pd.DataFrame([
        {"profile": "val v1 C2 (as sampled)", **dict(zip(BIN_LABELS, v1_shares.round(3))),
         "expected_c2_mrr": float(v1_shares @ bin_mrr)},
        {"profile": "NP proxy (enveda-np-examples)", **dict(zip(BIN_LABELS, np_shares.round(3))),
         "expected_c2_mrr": float(np_shares @ bin_mrr)},
        {"profile": f"target (tilted NP proxy, C2 = {args.target})", **dict(zip(BIN_LABELS, main_profile.round(3))),
         "expected_c2_mrr": float(main_profile @ bin_mrr)},
    ])
    corr = c2[["nn_tanimoto", "n_analogs", "rr"]].corr(method="spearman")["rr"]
    report = "\n\n".join([
        "# Analog difficulty of Class 2",
        f"- {len(c2)} C2 molecule-seeds ({args.prefix} {args.seeds}); analog channel p3/K100/mean; "
        f"NP proxy: {len(npd)} enveda-np-examples structures vs {len(all_lib):,} library structures",
        f"- Spearman with reciprocal rank: nn_tanimoto {corr['nn_tanimoto']:.2f}, n_analogs {corr['n_analogs']:.2f}",
        "### C2 MRR by nn_tanimoto bin", md_table(by_bin[["group", "bin", "n", "share", "c2_mrr", "median_analogs"]]),
        "### NP proxy nn_tanimoto", md_table(pd.DataFrame([npd["nn_tanimoto"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])])),
        "### Profiles and the C2 MRR they imply (per-bin MRR from val v1)", md_table(expected),
        f"### Bin mix that gives the LB-implied C2 MRR, closest to the NP proxy", md_table(sol),
        "> If part of the hidden C2 answers is missing from our pool (PubChem-only), the in-pool molecules must "
        "score higher to give the same LB, so the required profile is easier (rows with pool_miss_share > 0).",
        f"- target profile {'saved to ' + str(PROFILE) if args.prefix == 'seed' else 'not rewritten'}; "
        f"runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"ANALOG_DIFFICULTY_{args.prefix}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
