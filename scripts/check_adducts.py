"""Adduct coverage and precursor-mass consistency for train/test.

For every train spectrum, compares the observed precursor m/z with the value
implied by the labelled structure and adduct (via ``casmi.chem``). Spectra off
by more than ``--ppm`` are matched against other adducts of the same polarity
and a 13C-isotope precursor to find likely mislabels.

Writes ``notes/ADDUCTS.md`` and caches per-structure masses to
``WORK_DIR/structure_masses.parquet`` (reused by EDA and validation).

    python scripts/check_adducts.py [--ppm 20] [--workers 24]
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.chem import ADDUCTS, TEST_ADDUCTS, adduct_spec, exact_mass, parse_adduct, ppm_error, precursor_mz
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.data import read_frame

C13_SHIFT = 1.0033548378
LOW_RES_DA = 0.5


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    """Minimal markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def structure_masses(smiles: pd.Series, workers: int) -> pd.DataFrame:
    """Neutral-atom exact mass per unique SMILES, cached under WORK_DIR."""
    cache = WORK_DIR / "structure_masses.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    uniq = smiles.dropna().unique().tolist()
    with Pool(workers) as pool:
        masses = pool.map(exact_mass, uniq, chunksize=2000)
    out = pd.DataFrame({"normalized_smiles": uniq, "exact_mass": np.array(masses, dtype=np.float64)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def adduct_status(name: str) -> str:
    if name in ADDUCTS:
        return "ADDUCTS"
    return "parsed" if parse_adduct(name) else "UNKNOWN"


def coverage_section(train: pd.DataFrame, test: pd.DataFrame) -> str:
    tr = train["adduct"].value_counts().rename("train")
    te = test["adduct"].value_counts().rename("test")
    cov = pd.concat([tr, te], axis=1).fillna(0).astype(int).sort_values("train", ascending=False)
    cov["status"] = [adduct_status(a) for a in cov.index]
    cov = cov.reset_index(names="adduct")
    unknown = cov[cov["status"] == "UNKNOWN"]
    in_test_missing = [a for a in TEST_ADDUCTS if a not in ADDUCTS]
    return "\n\n".join(
        [
            "## 1. Adduct coverage",
            f"- train: {len(tr)} distinct adducts, test (visible): {len(te)}",
            f"- resolvable by `casmi.chem`: {int((cov['status'] != 'UNKNOWN').sum())}/{len(cov)}"
            f" ({cov.loc[cov['status'] != 'UNKNOWN', 'train'].sum() / len(train):.5%} of train rows)",
            f"- unknown: {', '.join(f'`{a}` ({n})' for a, n in zip(unknown['adduct'], unknown['train'])) or 'none'}",
            f"- documented test adducts missing from ADDUCTS: {in_test_missing or 'none'}",
            md_table(cov),
        ]
    )


def compute_ppm(train: pd.DataFrame) -> pd.Series:
    ppm = pd.Series(np.nan, index=train.index)
    for adduct, idx in train.groupby("adduct").groups.items():
        if parse_adduct(adduct) is None and adduct not in ADDUCTS:
            continue
        rows = train.loc[idx]
        ppm.loc[idx] = ppm_error(rows["precursor_mz"].to_numpy(), precursor_mz(rows["M"].to_numpy(), adduct))
    return ppm


def classify_bad(bad: pd.DataFrame, threshold: float) -> pd.Series:
    """Explain each off-mass row: another same-polarity adduct, a 13C precursor,
    low mass accuracy (|Δ| < LOW_RES_DA, typical of nominal-mass libraries), or nothing."""
    alts = list({spec: name for name, spec in reversed(ADDUCTS.items())}.values())[::-1]  # one name per ion
    out = pd.Series("unexplained", index=bad.index, dtype=object)
    for adduct, idx in bad.groupby("adduct").groups.items():
        rows = bad.loc[idx]
        mz = rows["precursor_mz"].to_numpy()
        m = rows["M"].to_numpy()
        n_m, _, charge = adduct_spec(adduct)
        names, errs = [], []
        for alt in alts:
            if alt == adduct or np.sign(ADDUCTS[alt][2]) != np.sign(charge):
                continue
            names.append(f"→ {alt}")
            errs.append(np.abs(ppm_error(mz, precursor_mz(m, alt))))
        names.append("13C precursor")
        errs.append(np.abs(ppm_error(mz, precursor_mz(m, adduct) + C13_SHIFT / abs(charge))))
        errs = np.vstack(errs)
        best = errs.argmin(axis=0)
        ok = errs[best, np.arange(errs.shape[1])] <= threshold
        low_res = np.abs(mz - precursor_mz(m, adduct)) < LOW_RES_DA
        label = np.where(ok, np.array(names, dtype=object)[best], np.where(low_res, "low mass accuracy", "unexplained"))
        out.loc[idx] = label
    return out


def ppm_section(train: pd.DataFrame, threshold: float) -> str:
    ppm = train["ppm"]
    absppm = ppm.abs()
    q = absppm.quantile([0.5, 0.9, 0.95, 0.99, 0.999])
    good = absppm < threshold

    bad = train[absppm > threshold].copy()
    bad["cause"] = classify_bad(bad, threshold)
    bad["delta_da"] = (bad["precursor_mz"] - [precursor_mz(m, a) for m, a in zip(bad["M"], bad["adduct"])]).round(2)

    g = train.assign(abs_ppm=absppm, is_bad=absppm > threshold).groupby("adduct")
    per_adduct = (
        g.agg(n=("ppm", "size"), median_ppm=("ppm", "median"), p99_abs=("abs_ppm", lambda s: s.quantile(0.99)),
              n_bad=("is_bad", "sum"))
        .assign(frac_bad=lambda d: d["n_bad"] / d["n"])
        .sort_values("n", ascending=False)
        .head(25)
        .reset_index()
    )
    per_lib = (
        train.assign(is_bad=absppm > threshold, ppm_ok=ppm.where(good))
        .groupby("ingest_lib")
        .agg(n=("is_bad", "size"), n_bad=("is_bad", "sum"), median_ppm=("ppm_ok", "median"), std_ppm=("ppm_ok", "std"))
        .assign(frac_bad=lambda d: d["n_bad"] / d["n"])
        .sort_values("frac_bad", ascending=False)
        .reset_index()
    )
    causes = (
        bad.groupby("cause").agg(n=("cause", "size"), structures=("inchikey14", "nunique"))
        .assign(frac=lambda d: d["n"] / len(bad)).sort_values("n", ascending=False).reset_index()
    )
    mislabel = (
        bad[bad["cause"].str.startswith("→")].groupby(["adduct", "cause"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False).head(20)
    )
    unexplained = bad[bad["cause"] == "unexplained"]
    top_delta = unexplained["delta_da"].value_counts().head(15).rename("n").reset_index()
    top_delta["delta_da"] = top_delta["delta_da"].map("{:+.2f}".format)

    # Host precursor_error_ppm vs ours, on rows we consider fine.
    host = train.loc[good & train["precursor_error_ppm"].notna()]
    host_cmp = (
        host.assign(diff=host["precursor_error_ppm"] - host["ppm"])
        .groupby(["adduct", "ingest_lib"])
        .agg(n=("diff", "size"), ours_median=("ppm", "median"), host_median=("precursor_error_ppm", "median"),
             median_diff=("diff", "median"))
        .reset_index()
    )
    host_cmp = host_cmp[host_cmp["adduct"].isin(TEST_ADDUCTS) & (host_cmp["n"] >= 1000)]
    host_cmp = host_cmp.sort_values(["adduct", "n"], ascending=[True, False])

    return "\n\n".join(
        [
            f"## 2. Precursor mass consistency (threshold |ppm| > {threshold:g})",
            f"- rows with computable ppm: {ppm.notna().sum():,} / {len(train):,}"
            f" (structure unparseable: {train['M'].isna().sum():,}, adduct unknown: "
            f"{(train['M'].notna() & ppm.isna()).sum():,})",
            "- |ppm| quantiles: " + ", ".join(f"p{p * 100:g}={v:.2f}" for p, v in q.items()),
            f"- **|ppm| > {threshold:g}: {len(bad):,} rows ({len(bad) / ppm.notna().sum():.2%})**, "
            f"{bad['inchikey14'].nunique():,} structures",
            "### Causes of |ppm| > threshold",
            md_table(causes),
            "### Per library (median/std ppm over rows within threshold)",
            md_table(per_lib),
            "### Per adduct (top 25 by count)",
            md_table(per_adduct),
            "### Likely adduct mislabels: labelled → matching adduct (top 20)",
            md_table(mislabel),
            "### Unexplained rows: most common Δ (Da, observed − expected)",
            md_table(top_delta),
            "### Host `precursor_error_ppm` vs ours (test adducts, rows within threshold, n ≥ 1000)",
            md_table(host_cmp),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppm", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "notes" / "ADDUCTS.md")
    args = ap.parse_args()

    cols = ["ingest_lib", "normalized_smiles", "inchikey14", "adduct", "precursor_mz", "precursor_error_ppm"]
    train = read_frame("train", cols)
    test = read_frame("test", ["molecule_id", "adduct"])

    masses = structure_masses(train["normalized_smiles"], args.workers)
    train["M"] = train["normalized_smiles"].map(masses.set_index("normalized_smiles")["exact_mass"])
    train["ppm"] = compute_ppm(train)

    report = "\n\n".join(["# Adduct & precursor-mass check", coverage_section(train, test),
                          ppm_section(train, args.ppm)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
