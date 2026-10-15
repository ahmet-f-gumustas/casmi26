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
    python validation/make_split.py --decoys-from v2_seed0   # split v3 → splits/v3_seed0
    python validation/make_split.py --v4-from v3_seed0 --profile 40 45 15 0   # split v4 → splits/v4_seed0

Split v3 (``--decoys-from``) copies an existing split and also removes the spectra of a random
``--decoy-frac`` of the non-validation library structures (``casmi.split.add_decoys``); the pool
is unchanged. Output names: ``v2_seedN`` → ``v3_seedN``, ``r_seedN`` → ``r3_seedN``; the decoy keys
go to ``decoy_keys.parquet``.

Split v4 (``--v4-from``) takes a split v3 and gives every Class 2 molecule an nn_tanimoto bin drawn
by ``--profile`` (shares for 0–0.3 / 0.3–0.5 / 0.5–0.7 / 0.7+, per source; ``casmi.split.nn_caps``).
Library structures at or above the bin's upper edge from the answer leave the library
(``casmi.split.neighbour_decoys``); validation structures stay, and so does the pool. The eligible
molecules have almost no nn_tanimoto below 0.4 against the full library, so a profile can only be
reached by removing neighbours. ``val_ids.parquet`` gains ``nn_cap`` and ``nn_tanimoto_full``
(before removal); ``nn_tanimoto`` is the Class 2 answer's nn against the remaining library.
Output names: ``v3_seedN`` → ``v4_seedN``, ``r3_seedN`` → ``r4_seedN``; removed keys go to
``neighbour_keys.parquet``.
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
    add_decoys,
    candidate_pool_keys,
    ce_pattern,
    check_leakage,
    eligible_queries,
    final_split,
    make_split,
    neighbour_decoys,
    nn_caps,
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


def v3_name(name: str) -> str:
    """Split-v3 directory name of a split derived from ``name``."""
    if name.startswith("v2_"):
        return "v3_" + name[3:]
    if name.startswith("r_"):
        return "r3_" + name[2:]
    return name + "_v3"


def decoy_split(source: str, frac: float, out_dir: Path | None) -> None:
    """Split v3 from an existing split (see the module docstring)."""
    src = WORK_DIR / "splits" / source
    cfg = json.loads((src / "config.json").read_text())
    split = Split(val=pd.read_parquet(src / "val_ids.parquet"), library_mask=pd.read_parquet(src / "library_mask.parquet"))
    row_key = library_row_keys(CLEAN_LIBRARY)
    v3, decoys = add_decoys(split, row_key, frac, cfg["seed"])
    assert not set(decoys) & set(split.val["inchikey14"]), "a validation answer became a decoy"
    out_dir = out_dir or WORK_DIR / "splits" / v3_name(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    v3.val.to_parquet(out_dir / "val_ids.parquet", index=False)
    v3.library_mask.to_parquet(out_dir / "library_mask.parquet", index=False)
    pd.DataFrame({"key": decoys}).to_parquet(out_dir / "decoy_keys.parquet", index=False)
    (out_dir / "config.json").write_text(json.dumps({**cfg, "decoys_from": source, "decoy_frac": frac}, indent=2) + "\n")
    before, after = int(split.library_mask["in_library"].sum()), int(v3.library_mask["in_library"].sum())
    lib = set(row_key.reindex(v3.library_mask.loc[v3.library_mask["in_library"], "row"].to_numpy()).dropna())
    report = "\n\n".join([
        f"# Split v3: {out_dir.name} (from {source}, decoy fraction {frac})",
        f"- {len(decoys):,} decoy structures leave the library ({before - after:,} spectra; library {before:,} → "
        f"{after:,} spectra, {len(lib):,} structures); the candidate pool is unchanged",
        f"- validation molecules, queries and classes are those of {source}",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"SPLIT_{out_dir.name}.md").write_text(report)
    print(report)


def v4_name(name: str) -> str:
    """Split-v4 directory name of a split derived from ``name``."""
    if name.startswith("v3_"):
        return "v4_" + name[3:]
    if name.startswith("r3_"):
        return "r4_" + name[3:]
    return name + "_v4"


def v4_split(source: str, profile: list[float], out_dir: Path | None) -> None:
    """Split v4 from a split v3 (see the module docstring)."""
    src = WORK_DIR / "splits" / source
    cfg = json.loads((src / "config.json").read_text())
    split = Split(val=pd.read_parquet(src / "val_ids.parquet"), library_mask=pd.read_parquet(src / "library_mask.parquet"))
    row_key = library_row_keys(CLEAN_LIBRARY)
    pool = CandidatePool.load()
    key_row = dict(zip(pool.meta["key"], range(len(pool.meta))))
    edges = tuple(cfg.get("nn_edges") or SplitConfig().nn_edges)
    shares = np.asarray(profile, dtype=float) / np.sum(profile)
    caps = nn_caps(split.val, shares, edges, cfg["seed"])
    v4, removed, nn = neighbour_decoys(split, row_key, caps, pool.fps, key_row)
    val = v4.val.copy()
    val["nn_tanimoto_full"] = val["nn_tanimoto"] if "nn_tanimoto" in val.columns else np.nan
    val["nn_cap"] = val["molecule_id"].map(caps)
    val["nn_tanimoto"] = val["molecule_id"].map(nn).fillna(val["nn_tanimoto_full"])
    out_dir = out_dir or WORK_DIR / "splits" / v4_name(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    val.to_parquet(out_dir / "val_ids.parquet", index=False)
    v4.library_mask.to_parquet(out_dir / "library_mask.parquet", index=False)
    pd.DataFrame({"key": removed}).to_parquet(out_dir / "neighbour_keys.parquet", index=False)
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg, "v4_from": source, "nn_profile": shares.round(4).tolist(), "nn_edges": list(edges)}, indent=2) + "\n")
    before, after = int(split.library_mask["in_library"].sum()), int(v4.library_mask["in_library"].sum())
    c2 = val[val["class"] == 2]

    def bins(col: str, sub: pd.DataFrame) -> pd.Series:
        return pd.cut(sub[col], list(edges), right=False, labels=BIN_LABELS).value_counts(normalize=True).sort_index()

    tab = pd.DataFrame({"C2 before": bins("nn_tanimoto_full", c2), "C2 after": bins("nn_tanimoto", c2),
                        "NP C2 before": bins("nn_tanimoto_full", c2[c2["source"] == NP_EXAMPLES]),
                        "NP C2 after": bins("nn_tanimoto", c2[c2["source"] == NP_EXAMPLES]),
                        "profile": pd.Series(shares, index=list(BIN_LABELS))}).reset_index(names="nn_tanimoto")
    report = "\n\n".join([
        f"# Split v4: {out_dir.name} (from {source}, nn profile {shares.round(3).tolist()})",
        f"- {len(removed):,} neighbour structures leave the library ({before - after:,} spectra; library {before:,} → "
        f"{after:,} spectra); the candidate pool is unchanged",
        f"- validation molecules, queries and classes are those of {source}",
        "### Class 2 nn_tanimoto bins (shares) against the library before / after", md_table(tab),
    ]) + "\n"
    (REPO_ROOT / "notes" / f"SPLIT_{out_dir.name}.md").write_text(report)
    print(report)


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
    ap.add_argument("--decoys-from", default=None, help="split v3: derive from this split (e.g. v2_seed0)")
    ap.add_argument("--decoy-frac", type=float, default=0.3)
    ap.add_argument("--v4-from", default=None, help="split v4: derive from this split v3 (e.g. v3_seed0)")
    ap.add_argument("--profile", type=float, nargs=4, default=None, metavar=("B0_03", "B03_05", "B05_07", "B07_1"),
                    help="split v4 Class 2 share per nn bin")
    args = ap.parse_args()
    if args.v4_from:
        v4_split(args.v4_from, args.profile, args.out_dir)
        return
    if args.decoys_from:
        decoy_split(args.decoys_from, args.decoy_frac, args.out_dir)
        return

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
