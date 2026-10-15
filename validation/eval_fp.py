"""Fingerprint channel (our SpecFormer) on split v2: alone and rank-averaged with the analog channel.

The model must be trained with ``--family v2`` (no validation molecule of any v2 seed seen).
Per seed: candidate windows, analog scores and fingerprint scores (f·z and soft-Tanimoto);
MRR@25 per class for each ranker configuration, mean ± std over seeds, and the difference to
analog only with the noise labels (total ±0.015, C1 ±0.04, C2 ±0.02). Fingerprint scores are
saved to ``outputs/channels/<split>/fingerprint_<tag>.parquet``.

"protected" rows put the library gate first (``casmi.ranker`` protect mode, τ = 0.8, rule switch:
a molecule whose best library match is ≥ τ is ranked by analog alone); the library max similarity
comes from the split's ranker features (``validation/ranker_features.py``).

    python validation/eval_fp.py --model outputs/fp_models/v2_full [--prefix v2_seed] [--seeds 0 1 2]
    python validation/eval_fp.py --model outputs/fp_models/np_v2 --prefix v3_seed
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.channels.analog import AnalogIndex, library_row_keys
from casmi.channels.fingerprint import fingerprint_channel
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.fpmodel import CasmiBundle
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, class_mrr, load_val, predict
from casmi.pipeline import PipelineConfig, analog_channel, candidate_windows, clean_queries, fingerprint_logits
from casmi.report import md_table

NOISE = {"total": 0.015, "C1": 0.04, "C2": 0.02, "C2 NP": 0.02}
PROTECT = {"mode": "protect", "tau": 0.8, "rule": "switch"}
CONFIGS = {
    "analog only": {"analog": 1.0},
    "fingerprint only (f·z)": {"fingerprint": 1.0},
    "fingerprint only (soft-Tanimoto)": {"fingerprint_st": 1.0},
    "analog + fingerprint (rank average)": {"analog": 1.0, "fingerprint": 1.0},
    "analog + 0.5 fingerprint": {"analog": 1.0, "fingerprint": 0.5},
    "analog + 2 fingerprint": {"analog": 1.0, "fingerprint": 2.0},
    "analog + fingerprint soft-T (rank average)": {"analog": 1.0, "fingerprint_st": 1.0},
    "protected: gate τ=0.8, then analog + fingerprint": {**PROTECT, "weights": {"analog": 1.0, "fingerprint": 1.0}},
    "protected: gate τ=0.8, then analog + fingerprint soft-T": {**PROTECT, "weights": {"analog": 1.0,
                                                                                      "fingerprint_st": 1.0}},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="folder with fp_casmi*.pt")
    ap.add_argument("--prefix", default="v2_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--device", default=None, help="fingerprint model device (default: cuda if available)")
    ap.add_argument("--tag", default=None, help="report name suffix (default: model folder name)")
    args = ap.parse_args()
    t0 = time.time()
    model = CasmiBundle.load(sorted(args.model.glob("fp_casmi*.pt")), device=args.device)
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    rows, fp_cache = [], {}
    for seed in args.seeds:
        split = f"{args.prefix}{seed}"
        data = load_val(WORK_DIR / "splits" / split)
        queries = clean_queries(data.queries).reset_index(drop=True)
        windows = candidate_windows(queries, data.pool, PipelineConfig())
        allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
        allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
        smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
        t1 = time.time()
        logits = fingerprint_logits(queries, windows, model)
        t_fp = time.time() - t1
        feats = pd.read_parquet(WORK_DIR / "ranker_features" / f"{split}.parquet",
                                columns=["molecule_id", "key", "library_max"])
        channels = {
            "library_max": {m: dict(zip(g["key"], g["library_max"])) for m, g in feats.groupby("molecule_id")},
            "analog": analog_channel(queries, windows, data.pool, index, row_key, allowed=allowed,
                                     workers=args.workers),
            "fingerprint": fingerprint_channel(logits, windows, smiles_of, "casmi6311", workers=args.workers,
                                               cache=fp_cache),
            "fingerprint_st": fingerprint_channel(logits, windows, smiles_of, "casmi6311", workers=args.workers,
                                                  cache=fp_cache, how="soft_tanimoto"),
        }
        out = WORK_DIR / "channels" / split
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([(m, k, s, channels["fingerprint_st"][m][k]) for m, ks in channels["fingerprint"].items()
                      for k, s in ks.items()], columns=["molecule_id", "key", "fingerprint", "fingerprint_st"]
                     ).to_parquet(out / f"fingerprint_{args.tag or args.model.name}.parquet", index=False)
        ids = data.val["molecule_id"]
        for label, w in CONFIGS.items():
            rows.append({**class_mrr(predict(windows, channels, w, smiles_of, ids), data.val, label), "seed": seed})
            np_val = data.val[data.val["source"] == "enveda-np-examples"]
            rows[-1]["C2 NP"] = class_mrr(predict(windows, channels, w, smiles_of, np_val["molecule_id"]),
                                          np_val, label)["C2"]
        print(f"{split} done: {len(queries)} spectra, fingerprint logits {t_fp:.1f}s ({time.time() - t0:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("run", sort=False)[["C1", "C2", "C2 NP", "total"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    base = agg.loc["analog only"]

    def verdict(row: pd.Series, col: str) -> str:
        d = row[f"{col}_mean"] - base[f"{col}_mean"]
        return f"{d:+.3f}" + ("" if abs(d) > NOISE[col] else " (n.m.)")

    for col in ("total", "C1", "C2", "C2 NP"):
        agg[f"Δ {col}"] = agg.apply(verdict, axis=1, col=col)
    agg = agg.reset_index()
    tag = args.tag or args.model.name
    report = "\n\n".join([
        f"# Fingerprint channel ({tag}; {args.prefix} {args.seeds})",
        "- n.m. = not measurable (|Δ| ≤ 0.015 total, 0.04 C1, 0.02 C2 / C2 NP) vs analog only; C3 is 0 everywhere",
        "### MRR@25, mean ± std over seeds", md_table(agg),
        f"- runtime {time.time() - t0:.0f}s; fingerprint scores saved to outputs/channels/<split>/fingerprint_{tag}.parquet",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP_fp_{tag}_{args.prefix}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
