"""Exploratory data analysis of train / test / sample_submission → notes/EDA.md.

Train is read column-wise; the peak arrays are only touched row group by row
group for the "peaks above precursor" check. Needs the structure-mass cache
from ``scripts/check_adducts.py``.

    python scripts/eda.py [--ppm 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi.chem import mass_window_indices, neutral_mass
from casmi.config import DATA_DIR, REPO_ROOT, WORK_DIR
from casmi.data import read_frame, read_table
from casmi.report import md_table, share_table

TIMSTOF_LIBS = ("enveda-180", "enveda-np-examples")
NP_PROXY = "enveda-np-examples"
# Hidden test set, from the competition data page.
HIDDEN_TEST = {"spectra": "~1,500", "molecules": "~400", "spectra/mol": "1–16 (median 3)",
               "mass": "157–1,159 Da (median 348)", "instrument": "timsTOF"}


def quantiles(s: pd.Series | np.ndarray, qs: tuple[float, ...] = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) -> dict:
    s = pd.Series(s).dropna()
    return {"n": len(s), **{f"p{q * 100:g}": float(s.quantile(q)) for q in qs}, "max": float(s.max())}


def bins(s: pd.Series, edges: list[float], name: str) -> pd.DataFrame:
    cut = pd.cut(s, edges, right=False)
    out = cut.value_counts(sort=False).rename_axis(name).reset_index(name="n")
    out[name] = out[name].astype(str)
    out["share"] = out["n"] / s.notna().sum()
    out["bar"] = (out["share"] * 60).round().astype(int).map(lambda k: "█" * k)
    return out


def ce_frame(table: pa.Table) -> pd.DataFrame:
    """Per spectrum: number of CE values and their mean (eV)."""
    ce = table.column("collision_energy_ev").combine_chunks()
    n_ce = pc.list_value_length(ce).to_numpy(zero_copy_only=False).astype(np.float64)
    flat = pc.list_flatten(ce).to_numpy(zero_copy_only=False)
    parent = pc.list_parent_indices(ce).to_numpy(zero_copy_only=False)
    sums = np.bincount(parent, weights=flat, minlength=len(ce))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_ce = np.where(n_ce > 0, sums / n_ce, np.nan)
    return pd.DataFrame({"n_ce": n_ce, "mean_ce": mean_ce})


def ce_pattern(n_ce: pd.Series) -> pd.Series:
    return pd.Series(np.select([n_ce.isna(), n_ce == 0, n_ce == 1], ["null", "null", "single"], "merged"),
                     index=n_ce.index)


def peaks_above_precursor(path: Path, margin: float = 2.0) -> pd.DataFrame:
    """Per library: share of spectra with any peak above precursor_mz + margin."""
    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ingest_lib", "precursor_mz", "ms2_mzs"])
        mzs = t.column("ms2_mzs").combine_chunks()
        flat = pc.list_flatten(mzs).to_numpy(zero_copy_only=False)
        parent = pc.list_parent_indices(mzs).to_numpy(zero_copy_only=False)
        prec = t.column("precursor_mz").to_numpy()
        above = np.zeros(len(t), dtype=bool)
        above[np.unique(parent[flat > prec[parent] + margin])] = True
        parts.append(pd.DataFrame({"ingest_lib": t.column("ingest_lib").to_numpy(zero_copy_only=False), "above": above}))
    df = pd.concat(parts, ignore_index=True)
    out = df.groupby("ingest_lib")["above"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "frac_above"})
    out.loc["ALL"] = [len(df), df["above"].mean()]
    return out.reset_index()


def train_section(train: pd.DataFrame, struct: pd.DataFrame) -> list[str]:
    per_lib = (
        train.groupby("ingest_lib")
        .agg(spectra=("inchikey14", "size"), structures=("inchikey14", "nunique"),
             instrument_null=("instrument_type", lambda s: s.isna().mean()),
             bpi_null=("base_peak_intensity", lambda s: s.isna().mean()),
             median_peaks=("num_peaks", "median"))
        .assign(spectra_per_structure=lambda d: d["spectra"] / d["structures"])
        .sort_values("spectra", ascending=False)
        .reset_index()
    )
    per_ik = train.groupby("inchikey14").size()
    libs_per_ik = train.groupby("inchikey14")["ingest_lib"].nunique()
    ce = train[["n_ce", "mean_ce"]]
    out = [
        "## 1. Train",
        f"- {len(train):,} spectra, {train['inchikey14'].nunique():,} unique InChIKey14, "
        f"{train['normalized_smiles'].nunique():,} unique SMILES",
        "### Libraries",
        md_table(per_lib),
        f"### Instrument type ({train['instrument_type'].nunique()} distinct strings, "
        f"{train['instrument_type'].isna().mean():.1%} null)",
        md_table(share_table(train["instrument_type"], "instrument_type", top=20)),
        "### Ionization mode",
        md_table(share_table(train["ionization_mode"], "ionization_mode")),
        "### Adduct (top 15)",
        md_table(share_table(train["adduct"], "adduct", top=15)),
        "### Collision energy",
        md_table(share_table(train["collision_energy_orig_units"], "units")),
        md_table(share_table(ce_pattern(ce["n_ce"]), "ce_pattern")),
        "Mean CE (eV) per spectrum:",
        md_table(bins(ce["mean_ce"], [0, 10, 20, 30, 40, 50, 60, 80, 100, 150, np.inf], "ev")),
        "### Spectra per InChIKey14",
        md_table(pd.DataFrame([quantiles(per_ik)])),
        md_table(bins(per_ik, [1, 2, 3, 6, 11, 21, 51, 101, np.inf], "spectra")),
        f"- structures in ≥ 2 libraries: {(libs_per_ik >= 2).mean():.1%}",
        "### num_peaks",
        md_table(pd.DataFrame([quantiles(train["num_peaks"])])),
        f"- < 5 peaks: {(train['num_peaks'] < 5).mean():.1%}, > 128: {(train['num_peaks'] > 128).mean():.1%}, "
        f"> 1000: {(train['num_peaks'] > 1000).mean():.2%}",
        "### precursor_mz",
        md_table(bins(train["precursor_mz"], [0, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, np.inf], "mz")),
        "### Structure neutral mass (unique InChIKey14)",
        md_table(pd.DataFrame([quantiles(struct["exact_mass"])])),
        f"- inside hidden-test range 157–1159 Da: {struct['exact_mass'].between(157, 1159).mean():.1%}",
        "### base_peak_intensity",
        f"- null: {train['base_peak_intensity'].isna().mean():.1%}; below 1,000 raw counts (test drops these): "
        f"{(train['base_peak_intensity'] < 1000).mean():.1%}",
    ]
    return out


def compare_section(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    groups = {
        "train (all)": train,
        "timsTOF libs": train[train["ingest_lib"].isin(TIMSTOF_LIBS)],
        NP_PROXY: train[train["ingest_lib"] == NP_PROXY],
        "visible test": test,
    }

    def shares(col: str, values: list[str]) -> pd.DataFrame:
        rows = {}
        for name, df in groups.items():
            s = df[col].value_counts(normalize=True)
            rows[name] = [s.get(v, 0.0) for v in values] + [1 - sum(s.get(v, 0.0) for v in values)]
        return pd.DataFrame(rows, index=values + ["(other)"]).reset_index(names=col)

    top_adducts = test["adduct"].value_counts().index.tolist()
    for name, df in groups.items():
        groups[name] = df.assign(ce_pattern=ce_pattern(df["n_ce"]))

    def per_mol(df: pd.DataFrame, key: str) -> dict:
        return quantiles(df.groupby(key).size(), (0.5, 0.9))

    spm = pd.DataFrame({
        "train (per ik14)": per_mol(train, "inchikey14"),
        "timsTOF libs (per ik14)": per_mol(groups["timsTOF libs"], "inchikey14"),
        f"{NP_PROXY} (per ik14)": per_mol(groups[NP_PROXY], "inchikey14"),
        "visible test (per molecule_id)": per_mol(test, "molecule_id"),
    }).T.reset_index(names="group")

    mass = pd.DataFrame({
        name: quantiles(df.groupby(key)["M_obs"].median(), (0.05, 0.5, 0.95))
        for name, df, key in [
            ("timsTOF libs", groups["timsTOF libs"], "inchikey14"),
            (NP_PROXY, groups[NP_PROXY], "inchikey14"),
            ("visible test", test, "molecule_id"),
        ]
    }).T.reset_index(names="group")

    return [
        "## 2. Test (visible) and distribution shift",
        "> **The visible `test.parquet` is a placeholder:** all 400 molecules are enveda-180 training spectra "
        "(every row matches a train row on precursor_mz + adduct + base peak intensity). The hidden test set "
        "replaces it at re-run. Shift vs. train below is therefore shift of *enveda-180 drug-like* chemistry; "
        f"`{NP_PROXY}` is the closest public proxy for the hidden NP test chemistry.",
        "Hidden test (data page): " + ", ".join(f"{k}: {v}" for k, v in HIDDEN_TEST.items()),
        f"- visible test: {len(test):,} spectra, {test['molecule_id'].nunique()} molecules, "
        f"instrument {test['instrument_type'].unique().tolist()}, CE units {test['collision_energy_orig_units'].unique().tolist()}",
        "### Spectra per molecule",
        md_table(spm),
        "### Adduct share",
        md_table(shares("adduct", top_adducts)),
        "### Ionization mode share",
        md_table(shares("ionization_mode", ["positive", "negative"])),
        "### CE pattern share (single energy vs merged multi-energy)",
        md_table(shares("ce_pattern", ["single", "merged"])),
        "### Visible test collision_energy_orig",
        md_table(share_table(test["collision_energy_orig"], "collision_energy_orig")),
        "### num_peaks",
        md_table(pd.DataFrame({n: quantiles(df["num_peaks"], (0.1, 0.5, 0.9)) for n, df in groups.items()}).T
                 .reset_index(names="group")),
        "### Molecule neutral mass (median over its spectra, from precursor_mz + adduct)",
        md_table(mass),
    ]


def mass_match_section(test: pd.DataFrame, struct: pd.DataFrame, ppm: float) -> list[str]:
    masses = np.sort(struct["exact_mass"].to_numpy())
    mol = test.groupby("molecule_id")["M_obs"].agg(["median", "min", "max"])
    mol["spread_ppm"] = (mol["max"] - mol["min"]) / mol["median"] * 1e6
    rows = []
    for label, shift_ppm in [("as measured", 0.0), ("−2 ppm timsTOF offset", -2.0)]:
        m = mol["median"] * (1 + shift_ppm * 1e-6)
        n_cand = np.array([mass_window_indices(masses, x, ppm).size for x in m])
        rows.append({"mass": label, "any_match": float((n_cand > 0).mean()),
                     "median_candidates": float(np.median(n_cand)), "p90_candidates": float(np.quantile(n_cand, 0.9))})
    return [
        f"## 3. Test molecules with a train structure within ±{ppm:g} ppm",
        f"- train structures (unique InChIKey14): {len(masses):,}",
        f"- per-molecule spread of neutral mass across its spectra: median {mol['spread_ppm'].median():.2f} ppm, "
        f"p99 {mol['spread_ppm'].quantile(0.99):.2f} ppm",
        md_table(pd.DataFrame(rows)),
        "> Caveat: a mass match only says an isobaric structure exists in train, so this share is an **upper "
        "bound** on \"answer is in train\", not a lower bound on Class 1. On the visible placeholder test it is "
        "trivially ~100% because those molecules come from train; rerun on the hidden test is impossible offline. "
        "`median_candidates` is the more useful number: the isomer pool a train-only retrieval must rank.",
    ]


def submission_section(sub: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    guesses = sub["smiles"].str.split(";")
    n = guesses.str.len()
    return [
        "## 4. sample_submission.csv format",
        f"- columns: {list(sub.columns)}; {len(sub)} rows, one per molecule_id; "
        f"ids match test molecule_ids: {set(sub['molecule_id']) == set(test['molecule_id'])}; duplicates: "
        f"{sub['molecule_id'].duplicated().sum()}",
        f"- `smiles` = up to 25 SMILES joined by `;`, best first. Guesses per row: min {n.min()}, median "
        f"{int(n.median())}, max {n.max()}; distinct SMILES used: {sorted(set(guesses.explode()))[:5]}",
        "- scorer rejects: missing `molecule_id`/`smiles`, empty file, nulls, repeated molecule_id, > 25 guesses. "
        "Empty guesses (`;;`) are dropped before ranking; invalid / duplicate SMILES keep their slot.",
        "```\n" + sub.head(3).to_csv(index=False) + "```",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "notes" / "EDA.md")
    args = ap.parse_args()

    meta = ["ingest_lib", "normalized_smiles", "inchikey14", "ionization_mode", "instrument_type", "adduct",
            "precursor_mz", "num_peaks", "base_peak_intensity", "collision_energy_ev", "collision_energy_orig_units"]
    t = read_table("train", meta)
    train = t.drop(["collision_energy_ev"]).to_pandas().join(ce_frame(t))
    del t
    tt = read_table("test")
    test = tt.drop(["ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]).to_pandas().join(ce_frame(tt))
    test["num_peaks"] = pc.list_value_length(tt.column("ms2_mzs")).to_numpy(zero_copy_only=False)
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    cache = WORK_DIR / "structure_masses.parquet"
    if not cache.exists():
        raise SystemExit(f"{cache} missing: run scripts/check_adducts.py first")
    smiles_mass = pd.read_parquet(cache).set_index("normalized_smiles")["exact_mass"]
    struct = (train[["inchikey14", "normalized_smiles"]].drop_duplicates("inchikey14")
              .assign(exact_mass=lambda d: d["normalized_smiles"].map(smiles_mass)))

    for df in (train, test):
        df["M_obs"] = np.nan
        for adduct, idx in df.groupby("adduct").groups.items():
            try:
                df.loc[idx, "M_obs"] = neutral_mass(df.loc[idx, "precursor_mz"].to_numpy(), adduct)
            except KeyError:
                pass

    above = peaks_above_precursor(DATA_DIR / "train.parquet")
    sections = [
        "# EDA — CASMI 2026",
        f"Files: train.parquet {(DATA_DIR / 'train.parquet').stat().st_size / 1e9:.2f} GB "
        f"({len(train):,} rows), test.parquet {len(test):,} rows, sample_submission.csv {len(sub)} rows.",
        *train_section(train, struct),
        "### Spectra with a peak above precursor + 2 Da (removed in test, kept in train)",
        md_table(above.sort_values("frac_above", ascending=False)),
        *compare_section(train, test),
        *mass_match_section(test, struct, args.ppm),
        *submission_section(sub, test),
    ]
    report = "\n\n".join(sections) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
