"""Evaluate the fragmentation channel on a validation split.

Fragment masses are computed once per candidate structure (pool representative SMILES)
in parallel and cached; each query spectrum is scored by explained intensity, then
aggregated per molecule (max / mean). Reports MRR@25 per class, time per candidate,
and saves the mean-aggregated scores to ``outputs/channels/<split>/fragment.parquet``.

    python validation/eval_fragment.py [--split outputs/splits/seed0]
"""

from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.channels.analog import rank_candidates
from casmi.channels.fragment import FragmentConfig, explained_intensity, fragment_masses, ion_masses
from casmi.channels.library import prepare
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import class_mrr, load_val
from casmi.pipeline import PipelineConfig, candidate_windows, clean_queries
from casmi.report import md_table

CACHE = WORK_DIR / "fragment_cache.pkl"
CFG = FragmentConfig()


def timed_fragments(smiles: str) -> tuple[np.ndarray, float]:
    t = time.perf_counter()
    f = fragment_masses(smiles, CFG)
    return f, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=WORK_DIR / "splits" / "seed0")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()

    data = load_val(args.split)
    queries = clean_queries(data.queries)
    windows = candidate_windows(queries, data.pool, PipelineConfig())
    smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
    keys = sorted({k for ks in windows.values() for k in ks})

    cache: dict[str, np.ndarray] = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    todo = [k for k in keys if k not in cache]
    times = np.zeros(0)
    if todo:
        with Pool(args.workers) as pool:
            res = pool.map(timed_fragments, [smiles_of[k] for k in todo], chunksize=50)
        cache.update({k: f for k, (f, _) in zip(todo, res)})
        times = np.array([t for _, t in res])
        CACHE.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
    frag_s = time.time() - t0

    spectra = {m: [(mode, prepare(np.asarray(mz), np.asarray(it), prec))
                   for mode, mz, it, prec in zip(g["ionization_mode"], g["ms2_mzs"], g["ms2_normalized_intensities"],
                                                 g["precursor_mz"])]
               for m, g in queries.groupby("molecule_id")}
    t1 = time.time()
    per_mol = {}
    for mol_id in data.val["molecule_id"]:
        cand = windows.get(mol_id, [])
        mat = np.zeros((len(spectra.get(mol_id, [])), len(cand)))
        for i, (mode, peaks) in enumerate(spectra.get(mol_id, [])):
            for j, k in enumerate(cand):
                mat[i, j] = explained_intensity(ion_masses(cache[k], mode, CFG), peaks, CFG.tol_da)
        per_mol[mol_id] = (cand, mat)
    score_s = time.time() - t1

    rows = []
    for agg in ("max", "mean"):
        preds = {m: [smiles_of[k] for k in rank_candidates(c, mat.max(0) if agg == "max" else mat.mean(0))]
                 for m, (c, mat) in per_mol.items() if mat.size}
        rows.append(class_mrr(preds, data.val, f"fragment/{agg}"))
        for src in sorted(data.val["source"].unique()):
            rows.append(class_mrr(preds, data.val[data.val["source"] == src], f"fragment/{agg} [{src}]"))

    # Correlation with the analog channel (per-molecule reciprocal rank of the answer).
    corr_lines = []
    analog_file = WORK_DIR / "channels" / args.split.name / "analog.parquet"
    if analog_file.exists():
        analog = pd.read_parquet(analog_file)
        a_scores = {m: dict(zip(g["key"], g["score"])) for m, g in analog.groupby("molecule_id")}
        key_of = dict(zip(data.val["molecule_id"], data.val["inchikey14"]))

        def rr(ranked: list[str], truth: str) -> float:
            return 1.0 / (ranked.index(truth) + 1) if truth in ranked else 0.0

        rec = []
        for m, (c, mat) in per_mol.items():
            if not mat.size:
                continue
            f_rank = rank_candidates(c, mat.mean(0))
            a = a_scores.get(m, {})
            a_rank = rank_candidates(c, np.array([a.get(k, 0.0) for k in c]))
            rec.append((m, rr(f_rank, key_of[m]), rr(a_rank, key_of[m])))
        rr_df = pd.DataFrame(rec, columns=["molecule_id", "fragment", "analog"]).merge(
            data.val[["molecule_id", "class", "source"]], on="molecule_id")
        for cls in (1, 2):
            sub = rr_df[rr_df["class"] == cls]
            corr_lines.append(
                f"- C{cls}: Pearson r(fragment RR, analog RR) = {sub['fragment'].corr(sub['analog']):.2f}, Spearman "
                f"{sub['fragment'].corr(sub['analog'], method='spearman'):.2f}; fragment rank-1 where analog is not: "
                f"{int(((sub['fragment'] == 1) & (sub['analog'] < 1)).sum())}/{len(sub)}, the reverse: "
                f"{int(((sub['analog'] == 1) & (sub['fragment'] < 1)).sum())}/{len(sub)}")

    out = WORK_DIR / "channels" / args.split.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(m, k, float(s)) for m, (c, mat) in per_mol.items() if mat.size for k, s in zip(c, mat.mean(0))],
                 columns=["molecule_id", "key", "score"]).to_parquet(out / "fragment.parquet", index=False)
    n_frag = np.array([len(cache[k]) for k in keys])
    timing = (f"fragment generation for {len(todo):,} new candidates: {frag_s:.0f}s wall on {args.workers} cores, "
              f"per candidate median {np.median(times) * 1e3:.1f} ms, p99 {np.quantile(times, 0.99) * 1e3:.0f} ms, "
              f"max {times.max():.2f} s" if len(times) else f"all {len(keys):,} candidates cached")
    report = "\n\n".join([
        f"# Fragment channel ({args.split.name})",
        f"- {len(keys):,} candidate structures, fragments per candidate median {np.median(n_frag):.0f} "
        f"(p90 {np.quantile(n_frag, 0.9):.0f}); tolerance {CFG.tol_da} Da, H shifts {CFG.h_shifts}",
        f"- {timing}; scoring {score_s:.0f}s single core; total {time.time() - t0:.0f}s",
        "### MRR@25", md_table(pd.DataFrame(rows)),
        "### Agreement with the analog channel", "\n".join(corr_lines) or "analog scores not found",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP003_fragment_{args.split.name}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
