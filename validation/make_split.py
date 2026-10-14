"""Build the molecule-level validation split (see ``casmi.split`` for the class contract).

Writes to ``WORK_DIR/splits/seed{seed}/`` (or ``splits/final/`` with ``--final``):
- ``val_ids.parquet``: molecule_id, inchikey14 (scorer key), class, smiles, source, exact_mass, n_query
- ``library_mask.parquet``: one row per train spectrum (train order): row, in_library, query_molecule_id
- ``config.json``
and a summary to ``notes/SPLIT_seed{seed}.md``. Scorer keys and masses per unique
SMILES are cached in ``WORK_DIR/structure_keys.parquet``.

    python validation/make_split.py [--seed 0] [--max-drug-share 0.5]
    python validation/make_split.py --seed 0 --difficulty   # split v2 → splits/v2_seed0
    python validation/make_split.py --final   # submission: nothing leaves the library
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from casmi.chem import parse_adduct, ppm_error, precursor_mz
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.data import read_table
from casmi.report import md_table
from casmi.split import (
    NP_EXAMPLES,
    Split,
    SplitConfig,
    candidate_pool_keys,
    ce_pattern,
    check_leakage,
    eligible_queries,
    final_split,
    make_split,
)
from casmi.structures import structure_table
from casmi.candidates import CandidatePool
from casmi.channels.analog import library_row_keys
from casmi.difficulty import BIN_LABELS, library_nn
from casmi.harness import CLEAN_LIBRARY


def load_rows(workers: int) -> pd.DataFrame:
    cols = ["ingest_lib", "normalized_smiles", "adduct", "precursor_mz", "base_peak_intensity", "collision_energy_ev"]
    t = read_table("train", cols)
    rows = t.drop(["collision_energy_ev"]).to_pandas()
    rows["n_ce"] = pc.list_value_length(t.column("collision_energy_ev")).to_numpy(zero_copy_only=False)
    keys = structure_table(rows["normalized_smiles"].unique(), workers).set_index("normalized_smiles")
    rows = rows.join(keys, on="normalized_smiles")
    rows["ppm"] = np.nan
    for adduct, idx in rows.groupby("adduct").groups.items():
        if parse_adduct(adduct) is None:
            continue
        sub = rows.loc[idx]
        rows.loc[idx, "ppm"] = ppm_error(sub["precursor_mz"].to_numpy(), precursor_mz(sub["exact_mass"].to_numpy(), adduct))
    return rows


def nn_table(rows: pd.DataFrame, cfg: SplitConfig, workers: int) -> pd.Series:
    """nn_tanimoto per eligible molecule vs the full cleaned library minus itself (cached)."""
    pool = CandidatePool.load()
    keys = sorted(set(rows.loc[eligible_queries(rows, cfg), "key"]))
    lib = sorted(set(library_row_keys(CLEAN_LIBRARY).dropna()))
    key_row = dict(zip(pool.meta["key"], range(len(pool.meta))))
    df = library_nn(keys, pool.fps, pool.meta["exact_mass"].to_numpy(), key_row, lib,
                    cache=WORK_DIR / "nn_tanimoto_train.parquet", chunk=2000)
    return df.set_index("key")["nn_tanimoto"]


def summary(rows: pd.DataFrame, split: Split, cfg: SplitConfig) -> str:
    """Validation queries side by side with the NP proxy and the host's hidden-test description."""
    val, mask = split.val, split.library_mask
    qmask = mask["query_molecule_id"].notna().to_numpy()
    q = rows[qmask].assign(molecule_id=mask.loc[qmask, "query_molecule_id"].to_numpy())
    proxy = rows[eligible_queries(rows, cfg) & (rows["ingest_lib"] == NP_EXAMPLES).to_numpy()]
    proxy_mol = proxy.groupby("key")["exact_mass"].median()

    def shares(col_q: pd.Series, col_p: pd.Series, name: str) -> pd.DataFrame:
        df = pd.concat([col_q.value_counts(normalize=True).rename("val queries"),
                        col_p.value_counts(normalize=True).rename("NP proxy")], axis=1).fillna(0.0)
        return df.sort_values("NP proxy", ascending=False).reset_index(names=name)

    def mass_row(name: str, m: pd.Series) -> dict:
        return {"group": name, "n": len(m), "min": m.min(), "p5": m.quantile(0.05), "p50": m.median(),
                "p95": m.quantile(0.95), "max": m.max()}

    mass = pd.DataFrame([
        mass_row("val (all)", val["exact_mass"]),
        *[mass_row(f"val {src}", val.loc[val["source"] == src, "exact_mass"]) for src in sorted(val["source"].unique())],
        mass_row("NP proxy", proxy_mol),
        {"group": "hidden test (host)", "n": "~400", "min": 157.0, "p5": "", "p50": 348.0, "p95": "", "max": 1159.0},
    ])
    spm = pd.DataFrame([
        {"group": "val", **q.groupby("molecule_id").size().describe(percentiles=[0.5, 0.9]).to_dict()},
        {"group": "NP proxy", **proxy.groupby("key").size().describe(percentiles=[0.5, 0.9]).to_dict()},
        {"group": "hidden test (host)", "min": 1, "50%": 3, "max": 16},
    ])
    classes = val.groupby(["class", "source"]).size().unstack(fill_value=0).reset_index()
    c2_bins = None
    if "nn_tanimoto" in val.columns and cfg.c2_profile is not None:
        c2 = val[val["class"] == 2]
        b = pd.cut(c2["nn_tanimoto"], cfg.nn_edges, right=False, labels=BIN_LABELS)
        c2_bins = pd.crosstab(b, c2["source"]).reindex(BIN_LABELS, fill_value=0)
        c2_bins["share"] = c2_bins.sum(axis=1) / len(c2)
        c2_bins["target"] = cfg.c2_profile
        c2_bins = c2_bins.reset_index(names="nn_tanimoto")
    removed = int((~mask["in_library"]).sum())
    el = rows[eligible_queries(rows, cfg)].groupby(["key", "ingest_lib"])["exact_mass"].median().reset_index()
    bins = pd.cut(el["exact_mass"], cfg.mass_bins, include_lowest=True)
    avail = pd.crosstab(bins, el["ingest_lib"]).join(
        pd.cut(val["exact_mass"], cfg.mass_bins, include_lowest=True).value_counts().rename("in val"))
    avail = avail.reset_index(names="mass bin").astype(str)
    return "\n\n".join([
        f"# Validation split (seed {cfg.seed})",
        f"- config: `{json.dumps(dataclasses.asdict(cfg))}`",
        f"- {len(val):,} molecules ({(val['source'] != NP_EXAMPLES).mean():.0%} enveda-180), {len(q):,} query "
        f"spectra; {removed:,} train spectra ({removed / len(mask):.2%}) leave the library; C3 removes "
        f"{int((val['class'] == 3).sum())} structures from the candidate pool",
        "### Classes × source", md_table(classes),
        "### Neutral mass (Da)", md_table(mass.map(lambda v: f"{v:.1f}" if isinstance(v, float) else str(v))),
        "### Eligible molecules per mass bin (enveda-180 has nothing above 600 Da)", md_table(avail),
        "### Adduct share (spectra)", md_table(shares(q["adduct"], proxy["adduct"], "adduct")),
        "### CE pattern share", md_table(shares(ce_pattern(q["n_ce"]), ce_pattern(proxy["n_ce"]), "ce_pattern")),
        *(["### Class 2 by nn_tanimoto bin (split v2)", md_table(c2_bins)] if c2_bins is not None else []),
        "### Spectra per molecule",
        md_table(spm.map(lambda v: f"{v:.2g}" if isinstance(v, float) and v == v else ("" if v != v else str(v)))),
        "> Leakage checks passed: no query or C2/C3 spectrum in the library, C1 keeps only non-timsTOF "
        "references, C3 structures not in the candidate pool, molecules grouped by scorer key.",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-drug-share", type=float, default=0.5, help="cap on enveda-180 molecules in val")
    ap.add_argument("--shares", type=float, nargs=3, default=(0.16, 0.45, 0.39), metavar=("C1", "C2", "C3"))
    ap.add_argument("--final", action="store_true", help="no validation: every spectrum stays in the library")
    ap.add_argument("--difficulty", action="store_true", help="split v2: Class 2 stratified by nn_tanimoto")
    ap.add_argument("--c2-profile", type=float, nargs=4, default=(0.0, 0.5, 0.35, 0.15),
                    metavar=("B0_03", "B03_05", "B05_07", "B07_1"), help="split v2 Class 2 share per nn bin")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = SplitConfig(max_drug_share=args.max_drug_share, seed=args.seed, class_shares=tuple(args.shares),
                      c2_profile=tuple(args.c2_profile) if args.difficulty else None)
    name = "final" if args.final else (f"v2_seed{cfg.seed}" if args.difficulty else f"seed{cfg.seed}")
    out_dir = args.out_dir or WORK_DIR / "splits" / name
    rows = load_rows(args.workers)
    if args.final:
        split = final_split(len(rows))
        report = f"# Final split\n\n- {len(rows):,} train spectra, all in the library; no validation molecules\n"
    else:
        nn = nn_table(rows, cfg, args.workers) if args.difficulty else None
        split = make_split(rows, cfg, nn)
        check_leakage(rows, split, candidate_pool_keys(rows["key"].dropna(), split))
        report = summary(rows, split, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    split.val.to_parquet(out_dir / "val_ids.parquet", index=False)
    split.library_mask.to_parquet(out_dir / "library_mask.parquet", index=False)
    (out_dir / "config.json").write_text(json.dumps({**dataclasses.asdict(cfg), "final": args.final}, indent=2) + "\n")
    note = REPO_ROOT / "notes" / f"SPLIT_{out_dir.name}.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(report)
    print(report)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
