"""Self-match variants of the analog channel, alone and with Class 1 protected by the library gate.

A candidate with library spectra is its own best analog (Tanimoto 1). ``AnalogConfig.self_min_sim``
counts a candidate's own spectra only at a spectral similarity of at least that value (0 = always,
the current channel; 2 = never). Each variant runs alone and behind the library gate (τ = 0.8, rule
switch: a molecule whose best library match is ≥ τ is ranked by the current analog channel).

Per variant: MRR@25 for C1 / C2 / NP C2 and, inside Class 2, per failure cluster of the current
analog ranking (``casmi.errors``: hit, (a) near-isomer, (b1) / (b2) retrieval, (c) intermediate);
mean ± std over seeds, Δ vs analog only with the noise labels (C1 ±0.04, C2 / NP C2 / clusters ±0.02).
A protected variant whose NP C2 gain reaches ``--lb-gain`` is written to ``outputs/self_match_<prefix>.json``.

    python validation/eval_self_match.py --prefix v4_seed [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from casmi.channels.analog import AnalogConfig, AnalogIndex, library_row_keys
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.errors import CLUSTERS, c2_error_rows, row_instruments
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val, reciprocal_ranks
from casmi.pipeline import PipelineConfig, analog_channel, analog_hits, candidate_windows, clean_queries, library_channel
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

TAU = 0.8
SELF = {"0.7": 0.7, "0.9": 0.9, "never": 2.0}
NOISE = {"C1": 0.04, "C2": 0.02, "NP C2": 0.02}
SHORT = {c: c.split(")")[0] + ")" if c.startswith("(") else "hit" for c in CLUSTERS}


def variants() -> dict[str, dict]:
    out = {"analog only (self 0)": {"mode": "weighted", "weights": {"analog": 1.0}}}
    for name in SELF:
        out[f"self {name}"] = {"mode": "weighted", "weights": {f"analog_{name}": 1.0}}
    for name in SELF:
        out[f"self {name} + gate τ={TAU}"] = {"mode": "protect", "tau": TAU, "rule": "switch",
                                              "weights": {f"analog_{name}": 1.0}}
    return out


def split_rr(split: str, index: AnalogIndex, row_key: dict, row_tims: dict, workers: int) -> pd.DataFrame:
    data = load_val(WORK_DIR / "splits" / split)
    queries = clean_queries(data.queries).reset_index(drop=True)
    cfg = PipelineConfig(workers=workers)
    windows = candidate_windows(queries, data.pool, cfg)
    allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    hits = analog_hits(queries, index, allowed=allowed, workers=workers)
    ch = {"analog": analog_channel(queries, windows, data.pool, index, row_key, allowed=allowed, hits=hits)}
    for name, s in SELF.items():
        ch[f"analog_{name}"] = analog_channel(queries, windows, data.pool, index, row_key, allowed=allowed, hits=hits,
                                              cfg=AnalogConfig(self_min_sim=s))
    _, ch["library_max"] = library_channel(queries, windows, CLEAN_LIBRARY, cfg, data.in_library, both=True)
    err = c2_error_rows(split, data.val, queries, windows, ch["analog"], hits, data.pool.meta, data.pool.fps, row_key,
                        row_tims).set_index("molecule_id")
    val = data.val.set_index("molecule_id")
    df = pd.DataFrame({run: reciprocal_ranks(windows, ch, cfg_r, data.val) for run, cfg_r in variants().items()})
    df["class"], df["source"] = val["class"].reindex(df.index), val["source"].reindex(df.index)
    df["cluster"] = err["cluster"].reindex(df.index)
    df["split"] = split
    return df


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, sub in df.groupby("split", sort=False):
        c2 = sub[sub["class"] == 2]
        for run in variants():
            rows.append({"split": split, "run": run, "C1": sub.loc[sub["class"] == 1, run].mean(), "C2": c2[run].mean(),
                         "NP C2": c2.loc[c2["source"] == NP_EXAMPLES, run].mean(),
                         **{SHORT[c]: c2.loc[c2["cluster"] == c, run].mean() for c in CLUSTERS}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="v4_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lb-gain", type=float, default=0.02, help="NP C2 gain over analog only that earns an LB probe")
    args = ap.parse_args()
    t0 = time.time()
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    row_tims = row_instruments(CLEAN_LIBRARY)
    df = pd.concat([split_rr(f"{args.prefix}{s}", index, row_key, row_tims, args.workers) for s in args.seeds])
    df.to_parquet(WORK_DIR / f"self_match_{args.prefix}.parquet")
    seeds = per_seed(df)
    cols = ["C1", "C2", "NP C2", *(SHORT[c] for c in CLUSTERS)]
    agg = seeds.groupby("run", sort=False)[cols].agg(["mean", "std"])
    base = agg.loc["analog only (self 0)"]
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        out[c] = agg[(c, "mean")].round(3).astype(str) + " ± " + agg[(c, "std")].round(3).astype(str)
    for c in cols:
        d = agg[(c, "mean")] - base[(c, "mean")]
        out[f"Δ {c}"] = [f"{x:+.3f}" + ("" if abs(x) > NOISE.get(c, 0.02) else " (n.m.)") for x in d]
    c2 = df[(df["class"] == 2) & (df["split"] == f"{args.prefix}{args.seeds[0]}")]
    shares = c2["cluster"].value_counts(normalize=True).reindex(CLUSTERS).fillna(0)
    gated = [r for r in agg.index if "gate" in r]
    gain = (agg.loc[gated, ("NP C2", "mean")] - base[("NP C2", "mean")]).sort_values(ascending=False)
    best, best_gain = gain.index[0], float(gain.iloc[0])
    if best_gain >= args.lb_gain:
        cfg = variants()[best]
        (WORK_DIR / f"self_match_{args.prefix}.json").write_text(json.dumps(
            {"run": best, "self_min_sim": SELF[best.split()[1]], "tau": TAU, "rule": "switch",
             "np_c2_gain": round(best_gain, 4), "ranker": cfg}, indent=2) + "\n")
        verdict = f"**{best}**: NP C2 {best_gain:+.3f} ≥ {args.lb_gain} → LB probe (outputs/self_match_{args.prefix}.json)"
    else:
        verdict = f"best protected variant {best}: NP C2 {best_gain:+.3f} < {args.lb_gain} → no LB probe"
    report = "\n\n".join([
        f"# Self-match variants ({args.prefix} {args.seeds})",
        "- self s: a candidate's own library spectra count as its analogs only at spectral similarity ≥ s "
        "(never = not at all); gate: library max ≥ τ ranks the molecule by the current analog channel",
        "- cluster columns: Class 2 MRR inside each failure cluster of the current analog ranking; n.m. = not "
        "measurable (C1 ±0.04, others ±0.02)",
        "### MRR@25, mean ± std over seeds", md_table(out[cols].reset_index()),
        "### Δ vs analog only", md_table(out[[f"Δ {c}" for c in cols]].reset_index()),
        "### Class 2 cluster shares (first seed)", md_table(shares.rename("share").rename_axis("cluster").reset_index()),
        f"- {verdict}", f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP_self_match_{args.prefix}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
