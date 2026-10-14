"""Clean train.parquet the way the test spectra were cleaned → outputs/train_clean.parquet.

Per spectrum (rules and evidence in ``casmi.clean``):
1. precursor mass vs the labelled structure, with our ppm: |ppm| ≤ 20 keeps; exactly
   one other same-polarity adduct within 20 ppm → relabelled (``adduct_fixed_from``
   keeps the old label); anything else is rejected with a reason;
2. peaks above precursor + 2 Da removed, intensities renormalised to a base peak of 1,
   ``base_peak_intensity`` rescaled to the remaining base peak;
3. no peaks left → rejected; base peak < 1,000 raw counts → rejected (a null base
   peak — libraries that shipped normalised intensities — is kept).

``row`` is the spectrum's position in train.parquet, so split masks join on it.

Outputs: ``WORK_DIR/train_clean.parquet`` (train columns + row, key, exact_mass, ppm,
adduct_fixed_from), ``WORK_DIR/train_rejected.parquet`` (raw rows + the same extras and
``reason``), ``notes/CLEAN.md``.

    python scripts/clean_train.py [--workers 24]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi.clean import FIXED, MIN_BASE_PEAK, OK, mass_verdict, strip_peaks
from casmi.config import DATA_DIR, REPO_ROOT, WORK_DIR
from casmi.data import read_table
from casmi.report import md_table
from casmi.structures import structure_table


def flat_lists(arr: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    arr = arr.combine_chunks()
    lengths = pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return offsets, pc.list_flatten(arr).to_numpy(zero_copy_only=False).astype(np.float64)


def process_group(t: pa.Table, start: int, keys: pd.Series, masses: pd.Series) -> tuple[pa.Table, pa.Table, dict]:
    n = t.num_rows
    row = np.arange(start, start + n, dtype=np.int64)
    smiles = t.column("normalized_smiles").to_pandas()
    mapped = smiles.map(keys)
    key = mapped.where(mapped.notna(), None).to_numpy(dtype=object)
    mass = smiles.map(masses).to_numpy(dtype=np.float64)
    adduct = np.asarray(t.column("adduct").to_pylist(), dtype=object)
    prec = t.column("precursor_mz").to_numpy()
    ppm, adduct_new, status = mass_verdict(prec, mass, adduct)

    off_mz, mz = flat_lists(t.column("ms2_mzs"))
    off_in, inten = flat_lists(t.column("ms2_normalized_intensities"))
    assert np.array_equal(off_mz, off_in), "m/z and intensity arrays are not aligned"
    new_off, new_mz, new_int, scale = strip_peaks(off_mz, mz, inten, prec)
    bpi = t.column("base_peak_intensity").to_numpy(zero_copy_only=False).astype(np.float64)
    new_bpi = bpi * scale
    counts = np.diff(new_off)

    passed = np.isin(status, [OK, FIXED])
    status[passed & (counts == 0)] = "no peaks left"
    passed = np.isin(status, [OK, FIXED])
    status[passed & (new_bpi < MIN_BASE_PEAK)] = "base peak < 1000"
    good = np.flatnonzero(np.isin(status, [OK, FIXED]))
    bad = np.flatnonzero(~np.isin(status, [OK, FIXED]))

    mz_list = pa.ListArray.from_arrays(pa.array(new_off.astype(np.int32)), pa.array(new_mz))
    in_list = pa.ListArray.from_arrays(pa.array(new_off.astype(np.int32)), pa.array(new_int))
    replaced = {
        "ms2_mzs": mz_list, "ms2_normalized_intensities": in_list,
        "num_peaks": pa.array(counts.astype(np.int64)), "base_peak_intensity": pa.array(new_bpi, from_pandas=True),
        "adduct": pa.array(adduct_new.tolist(), type=pa.string()),
    }
    cols = [replaced.get(name, t.column(name).combine_chunks()) for name in t.column_names]
    fixed_from = np.where(status == FIXED, adduct, None)
    extras = {
        "row": pa.array(row), "key": pa.array(key.tolist(), type=pa.string()), "exact_mass": pa.array(mass),
        "ppm": pa.array(ppm, from_pandas=True),
    }
    clean = pa.Table.from_arrays(cols + list(extras.values()) + [pa.array(fixed_from.tolist(), type=pa.string())],
                                 names=t.column_names + list(extras) + ["adduct_fixed_from"]).take(good)
    rejected = pa.Table.from_arrays(
        [t.column(name).combine_chunks() for name in t.column_names] + list(extras.values())
        + [pa.array(status.tolist(), type=pa.string())],
        names=t.column_names + list(extras) + ["reason"],
    ).take(bad)

    lib = np.asarray(t.column("ingest_lib").to_pylist(), dtype=object)
    kept_rows = np.zeros(n, dtype=bool)
    kept_rows[good] = True
    stats = {
        "lib": lib, "key": key, "kept": kept_rows, "status": status,
        "fixed": Counter(zip(adduct[status == FIXED], adduct_new[status == FIXED])),
        "peaks_removed": int((np.diff(off_mz) - counts)[good].sum()),
        "spectra_trimmed": int(((np.diff(off_mz) - counts) > 0)[good].sum()),
    }
    return clean, rejected, stats


def report(all_stats: list[dict]) -> str:
    lib = np.concatenate([s["lib"] for s in all_stats])
    key = np.concatenate([s["key"] for s in all_stats])
    kept = np.concatenate([s["kept"] for s in all_stats])
    status = np.concatenate([s["status"] for s in all_stats])
    df = pd.DataFrame({"lib": lib, "key": key, "kept": kept, "status": status})
    per_lib = df.groupby("lib").agg(
        rows_before=("kept", "size"), rows_after=("kept", "sum"),
        structures_before=("key", "nunique"), structures_after=("key", lambda s: s[df.loc[s.index, "kept"]].nunique()),
    )
    tot = pd.DataFrame([{
        "rows_before": len(df), "rows_after": int(kept.sum()),
        "structures_before": df["key"].nunique(), "structures_after": df.loc[df["kept"], "key"].nunique(),
    }], index=["ALL"])
    per_lib = pd.concat([per_lib.sort_values("rows_before", ascending=False), tot])
    per_lib["rows_kept"] = per_lib["rows_after"] / per_lib["rows_before"]
    reasons = pd.crosstab(df["lib"], df["status"])
    reasons = reasons[[c for c in ["fixed", *sorted(set(reasons.columns) - {"ok", "fixed"})] if c in reasons]]
    fixed = Counter()
    for s in all_stats:
        fixed.update(s["fixed"])
    fixed_df = pd.DataFrame([(a, b, n) for (a, b), n in fixed.most_common(20)], columns=["from", "to", "n"])
    lost = set(df["key"].dropna()) - set(df.loc[df["kept"], "key"].dropna())
    return "\n\n".join([
        "# Train cleaning",
        f"- rows: {len(df):,} → {int(kept.sum()):,} ({kept.mean():.2%} kept); structures (scorer key): "
        f"{df['key'].nunique():,} → {df.loc[df['kept'], 'key'].nunique():,} ({len(lost):,} lose every spectrum)",
        f"- adduct relabelled (ppm-verified): {int((status == 'fixed').sum()):,} rows",
        f"- peaks above precursor + 2 Da removed: {sum(s['peaks_removed'] for s in all_stats):,} peaks in "
        f"{sum(s['spectra_trimmed'] for s in all_stats):,} kept spectra",
        "### Rows and structures per library", md_table(per_lib.reset_index(names="ingest_lib")),
        "### Fixed / rejected rows per library", md_table(reasons.reset_index(names="ingest_lib")),
        "### Adduct relabels (top 20)", md_table(fixed_df),
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=WORK_DIR)
    args = ap.parse_args()

    smiles = read_table("train", ["normalized_smiles"]).column("normalized_smiles").unique().to_pylist()
    table = structure_table(smiles, args.workers).set_index("normalized_smiles")
    keys, masses = table["key"], table["exact_mass"]

    pf = pq.ParquetFile(DATA_DIR / "train.parquet")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_path, rej_path = args.out_dir / "train_clean.parquet", args.out_dir / "train_rejected.parquet"
    clean_w = rej_w = None
    start, all_stats = 0, []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i)
        clean, rejected, stats = process_group(t, start, keys, masses)
        start += t.num_rows
        if clean_w is None:
            clean_w = pq.ParquetWriter(clean_path, clean.schema, compression="zstd")
            rej_w = pq.ParquetWriter(rej_path, rejected.schema, compression="zstd")
        clean_w.write_table(clean, row_group_size=len(clean))
        rej_w.write_table(rejected)
        all_stats.append(stats)
        print(f"row group {i + 1}/{pf.num_row_groups}: kept {clean.num_rows:,} / {t.num_rows:,}", flush=True)
    clean_w.close()
    rej_w.close()

    text = report(all_stats) + (
        f"\nFiles: {clean_path.name} {clean_path.stat().st_size / 1e9:.2f} GB, "
        f"{rej_path.name} {rej_path.stat().st_size / 1e9:.2f} GB\n"
    )
    note = REPO_ROOT / "notes" / "CLEAN.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
