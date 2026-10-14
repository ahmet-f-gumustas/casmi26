"""Evaluate the analog channel (mass-shifted analog propagation) on a validation split.

Grid: power p ∈ {2,3,4}, top-K ∈ {50,100}, aggregation ∈ {max, mean}. The analog
search runs once per query spectrum (K = 100) against the full-library Flash index
with the split's library mask applied; smaller K is a prefix of the same list.
Per-candidate scores of the chosen configuration are saved to
``outputs/channels/<split>/analog.parquet`` for the ranker.

    python validation/eval_analog.py [--split outputs/splits/seed0] [--index outputs/analog_index]
"""

from __future__ import annotations

import argparse
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.channels.analog import AnalogConfig, AnalogIndex, aggregate, library_row_keys, rank_candidates, score_molecule
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import CLEAN_LIBRARY, class_mrr, load_val
from casmi.pipeline import PipelineConfig, candidate_windows, clean_queries, to_spectra
from casmi.report import md_table

POWERS = (2.0, 3.0, 4.0)
TOP_KS = (50, 100)
AGGS = ("max", "mean")
STATE: dict = {}


def search_one(i: int) -> tuple[np.ndarray, np.ndarray]:
    s = STATE
    mode, prec, peaks = s["queries"][i]
    return s["index"].search(mode, prec, peaks, s["cfg"], s["allowed"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=WORK_DIR / "splits" / "seed0")
    ap.add_argument("--index", type=Path, default=WORK_DIR / "analog_index")
    ap.add_argument("--method", default="hybrid", choices=["hybrid", "open"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--save", default="p3/k100/mean", help="configuration whose scores are saved, p<p>/k<K>/<agg>")
    args = ap.parse_args()
    t0 = time.time()

    data = load_val(args.split)
    cfg = AnalogConfig(top_k=max(TOP_KS), method=args.method)
    queries = clean_queries(data.queries).reset_index(drop=True)
    windows = candidate_windows(queries, data.pool, PipelineConfig())

    index = AnalogIndex.load(args.index)
    n_rows = int(data.in_library.index.max()) + 1
    allowed = np.zeros(n_rows, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    raw = [(m, float(p), np.column_stack([mz, it]).astype(np.float32))
           for m, p, mz, it in zip(queries["ionization_mode"], queries["precursor_mz"], queries["ms2_mzs"],
                                   queries["ms2_normalized_intensities"])]
    STATE.update(index=index, queries=raw, cfg=cfg, allowed=allowed)
    t1 = time.time()
    with Pool(args.workers) as pool:
        hits = pool.map(search_one, range(len(raw)), chunksize=4)
    STATE.clear()
    search_s = time.time() - t1

    # library row -> fingerprint row in the full pool (analog structures are library structures)
    meta = data.pool.meta
    fp_row_of_key = dict(zip(meta["key"], range(len(meta))))
    row_keys = library_row_keys(CLEAN_LIBRARY)
    used_rows = {int(r) for rows, _ in hits for r in rows}
    row_fp = {r: fp_row_of_key[k] for r, k in row_keys.loc[sorted(used_rows)].items() if k in fp_row_of_key}
    fps = data.pool.fps
    smiles_of = dict(zip(meta["key"], meta["smiles"]))

    by_mol = queries.groupby("molecule_id").indices
    scores = {}  # (p, K, agg) -> {mol: (keys, scores)}
    for mol_id in data.val["molecule_id"]:
        keys = windows.get(mol_id, [])
        cand_fps = fps[[fp_row_of_key[k] for k in keys]] if keys else np.zeros((0, fps.shape[1]), np.uint8)
        mol_hits = [hits[i] for i in by_mol.get(mol_id, [])]
        for k in TOP_KS:
            per = score_molecule(mol_hits, cand_fps, row_fp, fps, POWERS, top_k=k)
            for p, agg in itertools.product(POWERS, AGGS):
                scores.setdefault((p, k, agg), {})[mol_id] = (keys, aggregate(per[p], agg))

    rows, by_source = [], []
    for (p, k, agg), mol_scores in sorted(scores.items()):
        preds = {m: [smiles_of[x] for x in rank_candidates(keys, sc)] for m, (keys, sc) in mol_scores.items()}
        rows.append(class_mrr(preds, data.val, f"analog/p{p:g}/k{k}/{agg}"))
        for src in sorted(data.val["source"].unique()):
            by_source.append(class_mrr(preds, data.val[data.val["source"] == src], f"analog/p{p:g}/k{k}/{agg} [{src}]"))
    results = pd.DataFrame(rows).sort_values("total", ascending=False)
    best = results["run"].iloc[0]
    per_source = pd.DataFrame([r for r in by_source if r["run"].startswith(best + " ")])

    p_s, k_s, agg_s = args.save.split("/")
    chosen = scores[(float(p_s[1:]), int(k_s[1:]), agg_s)]
    out = WORK_DIR / "channels" / args.split.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(m, key, float(s)) for m, (keys, sc) in chosen.items() for key, s in zip(keys, sc)],
                 columns=["molecule_id", "key", "score"]).to_parquet(out / "analog.parquet", index=False)

    n_analogs = np.array([len(r) for r, _ in hits])
    top_sim = np.array([s[0] if len(s) else 0.0 for _, s in hits])
    report = "\n\n".join([
        f"# Analog channel ({args.split.name}, {args.method} search)",
        f"- {len(data.val)} molecules, {len(raw)} query spectra; search {search_s:.0f}s "
        f"({search_s / max(len(raw), 1) * 1e3 * args.workers:.0f} ms/spectrum/core); total {time.time() - t0:.0f}s",
        f"- analogs per spectrum: median {np.median(n_analogs):.0f}; best analog similarity median "
        f"{np.median(top_sim):.3f}; analog structures with a pool fingerprint: {len(row_fp)}/{len(used_rows)}",
        "### MRR@25 (sorted by total)", md_table(results),
        f"### Best configuration by source", md_table(per_source),
        f"- saved `{args.save}` scores to {out / 'analog.parquet'}",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP002_analog_{args.split.name}_{args.method}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
