"""Molecule-level validation split with simulated novelty classes.

A validation *molecule* is a scorer-key group (stereo-free, tautomer-canonical
InChIKey14, see ``casmi.metric.scorer_key``). Grouping on the train
``inchikey14`` column is not enough: it differs from the scorer key for ~1.7 %
of structures, so two tautomers could otherwise straddle train and validation.

Classes follow the hidden-test definitions (LB-estimated shares 16/45/39):

- **C1**: the molecule's timsTOF spectra (queries included) leave the library;
  its spectra from other libraries stay as public references.
- **C2**: every spectrum of the molecule leaves the library; the structure stays
  in the candidate pool.
- **C3**: every spectrum leaves the library and the structure leaves the pool.

Query spectra mimic the test: timsTOF, a test adduct, consistent precursor mass,
base peak ≥ 1000 raw counts, molecular mass inside the hidden-test range. The
number of spectra per molecule is drawn from the visible test, and molecules and
spectra are sampled with adduct / CE-pattern weights that match its shares.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from casmi.chem import TEST_ADDUCTS

TIMSTOF_LIBS = ("enveda-180", "enveda-np-examples")
NP_EXAMPLES = "enveda-np-examples"
ROW_COLUMNS = ("ingest_lib", "key", "normalized_smiles", "adduct", "ppm", "base_peak_intensity", "n_ce",
               "exact_mass")


@dataclass(frozen=True)
class SplitConfig:
    n_val: int = 2000
    class_shares: tuple[float, float, float] = (0.16, 0.45, 0.39)
    seed: int = 0
    include_np_examples: bool = True  # every eligible enveda-np-examples molecule goes to validation
    max_abs_ppm: float = 20.0
    min_base_peak: float = 1000.0
    mass_range: tuple[float, float] = (157.0, 1159.0)
    smoothing: float = 0.01  # added to both shares in the importance weights


@dataclass(frozen=True)
class Split:
    val: pd.DataFrame  # molecule_id, inchikey14 (scorer key), class, smiles, source, exact_mass, n_query
    library_mask: pd.DataFrame  # row, in_library, query_molecule_id


def ce_pattern(n_ce: pd.Series) -> pd.Series:
    """``single`` (one energy), ``merged`` (several) or ``null`` per spectrum."""
    n = n_ce.fillna(0).to_numpy()
    return pd.Series(np.select([n == 0, n == 1], ["null", "single"], "merged"), index=n_ce.index)


def _weighted_order(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    """Order for sampling without replacement ∝ ``weights`` (Efraimidis–Spirakis); take a prefix."""
    keys = np.log(rng.random(len(weights))) / np.maximum(weights, 1e-12)
    return np.argsort(-keys, kind="stable")


def _importance(values: pd.Series, target: pd.Series, alpha: float) -> pd.Series:
    """Per-value weight target_share / pool_share, both smoothed by ``alpha``."""
    pool = values.value_counts(normalize=True)
    w = {v: (target.get(v, 0.0) + alpha) / (pool.get(v, 0.0) + alpha) for v in pool.index}
    return values.map(w).astype(np.float64)


def make_split(rows: pd.DataFrame, test: pd.DataFrame, cfg: SplitConfig = SplitConfig()) -> Split:
    """Build the split. ``rows`` is one row per train spectrum, in train order, with
    ``ROW_COLUMNS`` (``key`` = scorer key, ``ppm`` = precursor error vs the labelled
    structure). ``test`` needs ``molecule_id``, ``adduct`` and ``n_ce``."""
    missing = set(ROW_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"rows is missing columns {sorted(missing)}")
    rng = np.random.default_rng(cfg.seed)
    rows = rows.reset_index(drop=True)
    is_tims = rows["ingest_lib"].isin(TIMSTOF_LIBS).to_numpy()
    pattern = ce_pattern(rows["n_ce"])
    eligible = (
        is_tims
        & rows["adduct"].isin(TEST_ADDUCTS).to_numpy()
        & (rows["ppm"].abs() <= cfg.max_abs_ppm).to_numpy()
        & (rows["base_peak_intensity"] >= cfg.min_base_peak).to_numpy()
        & rows["key"].notna().to_numpy()
    )

    # Row weights: bring the eligible pool's adduct and CE-pattern mix towards the test's.
    test_adduct = test["adduct"].value_counts(normalize=True)
    test_ce = ce_pattern(test["n_ce"]).value_counts(normalize=True)
    el = rows.loc[eligible, ["key", "adduct", "ingest_lib", "exact_mass", "normalized_smiles"]].copy()
    el["w"] = (_importance(el["adduct"], test_adduct, cfg.smoothing)
               * _importance(pattern[eligible], test_ce, cfg.smoothing))

    has_other = set(rows.loc[~is_tims & rows["key"].notna().to_numpy(), "key"])
    mol = (
        el.groupby("key", sort=True)
        .agg(smiles=("normalized_smiles", "min"), exact_mass=("exact_mass", "median"), weight=("w", "mean"),
             n_eligible=("w", "size"), is_np=("ingest_lib", lambda s: bool((s == NP_EXAMPLES).any())))
        .reset_index()
    )
    lo, hi = cfg.mass_range
    mol = mol[mol["exact_mass"].between(lo, hi)].reset_index(drop=True)
    mol["is_np"] = mol["is_np"].astype(bool)
    mol["c1_eligible"] = mol["key"].isin(has_other)

    # Class quotas. NP examples (optional) are spread over the classes first; the remaining
    # C1 slots are drawn from molecules with non-timsTOF reference spectra, C2/C3 from the
    # rest. Every draw is importance-weighted.
    s1, _, s3 = cfg.class_shares
    n1, n3 = round(s1 * cfg.n_val), round(s3 * cfg.n_val)
    n2 = cfg.n_val - n1 - n3
    cls = pd.Series(0, index=mol.index)  # 0 = not in validation
    if cfg.include_np_examples:
        np_idx = rng.permutation(mol.index[mol["is_np"]].to_numpy())
        if len(np_idx) > cfg.n_val:
            raise ValueError(f"n_val={cfg.n_val} is smaller than the {len(np_idx)} NP-example molecules")
        np_c1 = [i for i in np_idx if mol.at[i, "c1_eligible"]][: round(s1 * len(np_idx))]
        np_rest = [i for i in np_idx if i not in set(np_c1)]
        k3 = round(s3 * len(np_idx))
        cls.loc[np_c1] = 1
        cls.loc[np_rest[:k3]] = 3
        cls.loc[np_rest[k3:]] = 2

    def draw(candidates: np.ndarray, k: int, what: str) -> np.ndarray:
        if k < 0 or k > len(candidates):
            raise ValueError(f"only {len(candidates)} {what} molecules, need {k}")
        return candidates[_weighted_order(rng, mol.loc[candidates, "weight"].to_numpy())[:k]]

    cls.loc[draw(mol.index[(cls == 0) & mol["c1_eligible"]].to_numpy(), n1 - int((cls == 1).sum()),
                 "C1-eligible")] = 1
    need2, need3 = n2 - int((cls == 2).sum()), n3 - int((cls == 3).sum())
    c23 = rng.permutation(draw(mol.index[cls == 0].to_numpy(), need2 + need3, "eligible"))
    cls.loc[c23[:need3]] = 3
    cls.loc[c23[need3:]] = 2
    val = mol[cls > 0].assign(**{"class": cls[cls > 0].to_numpy()}).reset_index(drop=True)
    val["molecule_id"] = [f"v_{i:05d}" for i in range(len(val))]

    # Query spectra: count from the visible test's spectra-per-molecule, rows ∝ weight.
    spm = test.groupby("molecule_id").size().to_numpy()
    by_key = el.groupby("key").indices
    el_rows = el.index.to_numpy()
    el_w = el["w"].to_numpy()
    query_mol = np.full(len(rows), None, dtype=object)
    n_query = []
    for key, mol_id in zip(val["key"], val["molecule_id"]):
        idx = by_key[key]
        take = idx[_weighted_order(rng, el_w[idx])[: min(int(rng.choice(spm)), len(idx))]]
        query_mol[el_rows[take]] = mol_id
        n_query.append(len(take))
    val["n_query"] = n_query

    in_library = np.ones(len(rows), dtype=bool)
    key = rows["key"]
    in_library[key.isin(set(val["key"])).to_numpy()] = False
    in_library[(key.isin(set(val.loc[val["class"] == 1, "key"])).to_numpy()) & ~is_tims] = True

    val["source"] = np.where(val["is_np"], NP_EXAMPLES, "enveda-180")
    val = val.rename(columns={"key": "inchikey14"})[
        ["molecule_id", "inchikey14", "class", "smiles", "source", "exact_mass", "n_query"]
    ]
    mask = pd.DataFrame({"row": np.arange(len(rows)), "in_library": in_library, "query_molecule_id": query_mol})
    return Split(val=val, library_mask=mask)


def candidate_pool_keys(all_keys: Iterable[str], split: Split) -> set[str]:
    """Structure keys retrieval may rank: everything except Class 3 answers."""
    return set(all_keys) - set(split.val.loc[split.val["class"] == 3, "inchikey14"])


def check_leakage(rows: pd.DataFrame, split: Split, pool_keys: set[str]) -> None:
    """Raise ``AssertionError`` if any class contract is violated."""
    val, mask = split.val, split.library_mask
    assert len(mask) == len(rows), "mask must align with train rows"
    assert val["inchikey14"].is_unique, "a scorer key appears in two validation molecules"
    key = rows["key"].to_numpy()
    lib = mask["in_library"].to_numpy()
    is_tims = rows["ingest_lib"].isin(TIMSTOF_LIBS).to_numpy()

    q = mask["query_molecule_id"].notna().to_numpy()
    assert not (q & lib).any(), "a query spectrum is in the library"
    expected = val.set_index("molecule_id")["inchikey14"]
    assert (key[q] == expected.loc[mask.loc[q, "query_molecule_id"]].to_numpy()).all(), "query/key mismatch"
    assert set(mask.loc[q, "query_molecule_id"]) == set(val["molecule_id"]), "molecule without queries"

    lib_keys = set(key[lib])
    for c in (2, 3):
        leaked = set(val.loc[val["class"] == c, "inchikey14"]) & lib_keys
        assert not leaked, f"C{c} molecules have library spectra: {sorted(leaked)[:3]}"
    c1 = set(val.loc[val["class"] == 1, "inchikey14"])
    assert not (c1 & set(key[lib & is_tims])), "C1 molecules have timsTOF spectra in the library"
    assert c1 <= set(key[lib & ~is_tims]), "a C1 molecule has no reference spectra in the library"
    c3_in_pool = set(val.loc[val["class"] == 3, "inchikey14"]) & pool_keys
    assert not c3_in_pool, f"C3 structures in the candidate pool: {sorted(c3_in_pool)[:3]}"
