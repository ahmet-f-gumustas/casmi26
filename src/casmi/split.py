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

The target distribution is the hidden test as described by the host (timsTOF,
the ten test adducts, neutral mass 157–1159 Da) with the chemistry, adduct and
CE mix of ``enveda-np-examples``, the closest public proxy. Every eligible NP
example is a validation molecule; enveda-180 molecules are added up to
``max_drug_share`` of the set, drawn with per-mass-bin quotas matching the NP
mass distribution and adduct / CE-pattern importance weights within each bin.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from casmi.chem import TEST_ADDUCTS

TIMSTOF_LIBS = ("enveda-180", "enveda-np-examples")
NP_EXAMPLES = "enveda-np-examples"
DRUG_LIB = "enveda-180"
ROW_COLUMNS = ("ingest_lib", "key", "normalized_smiles", "adduct", "ppm", "base_peak_intensity", "n_ce",
               "exact_mass")
VAL_COLUMNS = ["molecule_id", "inchikey14", "class", "smiles", "source", "exact_mass", "n_query"]
V2_COLUMNS = [*VAL_COLUMNS, "nn_tanimoto"]


@dataclass(frozen=True)
class SplitConfig:
    max_drug_share: float = 0.5  # cap on enveda-180 molecules in validation
    class_shares: tuple[float, float, float] = (0.16, 0.45, 0.39)
    seed: int = 0
    max_abs_ppm: float = 20.0
    min_base_peak: float = 1000.0
    mass_range: tuple[float, float] = (157.0, 1159.0)
    mass_bins: tuple[float, ...] = (157.0, 250.0, 350.0, 450.0, 600.0, 800.0, 1159.0)
    max_spectra: int = 16  # hidden test has 1–16 spectra per molecule
    smoothing: float = 0.01  # added to shares in importance weights and bin targets
    # Split v2: Class 2 sampled to this share per nn_tanimoto bin (``nn_edges``); None = split v1.
    c2_profile: tuple[float, ...] | None = None
    nn_edges: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0001)


@dataclass(frozen=True)
class Split:
    val: pd.DataFrame  # VAL_COLUMNS; inchikey14 is the scorer key
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


def _bin_quotas(avail: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Split ``k`` draws over bins ∝ ``target``, capped by ``avail``; shortfalls move to open bins."""
    if k > avail.sum():
        raise ValueError(f"only {avail.sum()} candidates for {k} draws")
    quota = np.zeros(len(avail), dtype=int)
    while (left := k - quota.sum()) > 0:
        open_ = avail > quota
        share = np.where(open_, target, 0.0)
        share = share / share.sum() if share.sum() > 0 else open_ / open_.sum()
        add = np.minimum(np.floor(share * left).astype(int), avail - quota)
        if add.sum() == 0:
            add[np.argmax(np.where(open_, share, -1.0))] = 1
        quota += add
    return quota


def _stratified_draw(rng: np.random.Generator, candidates: np.ndarray, bins: np.ndarray, weights: np.ndarray,
                     target: np.ndarray, k: int, what: str) -> np.ndarray:
    """``k`` of ``candidates``: per-bin quotas from ``target``, weighted draws inside each bin."""
    avail = np.bincount(bins, minlength=len(target))
    if k > avail.sum():
        raise ValueError(f"only {avail.sum()} {what} molecules, need {k}")
    quota = _bin_quotas(avail, target, k)
    picks = []
    for b, q in enumerate(quota):
        idx = np.flatnonzero(bins == b)
        picks.append(candidates[idx[_weighted_order(rng, weights[idx])[:q]]])
    return np.concatenate(picks) if picks else np.array([], dtype=candidates.dtype)


def eligible_queries(rows: pd.DataFrame, cfg: SplitConfig = SplitConfig()) -> np.ndarray:
    """Spectra that look like test spectra: timsTOF, test adduct, clean precursor, base peak, mass range."""
    lo, hi = cfg.mass_range
    return (
        rows["ingest_lib"].isin(TIMSTOF_LIBS).to_numpy()
        & rows["adduct"].isin(TEST_ADDUCTS).to_numpy()
        & (rows["ppm"].abs() <= cfg.max_abs_ppm).to_numpy()
        & (rows["base_peak_intensity"] >= cfg.min_base_peak).to_numpy()
        & rows["exact_mass"].between(lo, hi).to_numpy()
        & rows["key"].notna().to_numpy()
    )


def make_split(rows: pd.DataFrame, cfg: SplitConfig = SplitConfig(), nn: pd.Series | None = None) -> Split:
    """Build the split. ``rows`` is one row per train spectrum, in train order, with
    ``ROW_COLUMNS`` (``key`` = scorer key, ``ppm`` = precursor error vs the labelled structure).

    With ``cfg.c2_profile`` (split v2), ``nn`` maps scorer key → nn_tanimoto (nearest library
    structure) and Class 2 is drawn to match the profile over ``cfg.nn_edges`` bins."""
    missing = set(ROW_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"rows is missing columns {sorted(missing)}")
    rng = np.random.default_rng(cfg.seed)
    rows = rows.reset_index(drop=True)
    is_tims = rows["ingest_lib"].isin(TIMSTOF_LIBS).to_numpy()
    pattern = ce_pattern(rows["n_ce"])
    eligible = eligible_queries(rows, cfg)
    el = rows.loc[eligible, ["key", "adduct", "ingest_lib", "exact_mass", "normalized_smiles"]].copy()
    el["pattern"] = pattern[eligible]
    el["is_np"] = el["ingest_lib"] == NP_EXAMPLES
    if not el["is_np"].any():
        raise ValueError("no eligible enveda-np-examples spectra: they define the target distribution")

    # Targets from the NP proxy: adduct and CE-pattern mix (spectra), mass bins and spectra per molecule.
    np_rows = el[el["is_np"]]
    el["w"] = (_importance(el["adduct"], np_rows["adduct"].value_counts(normalize=True), cfg.smoothing)
               * _importance(el["pattern"], np_rows["pattern"].value_counts(normalize=True), cfg.smoothing))
    np_spm = np.minimum(np_rows.groupby("key").size().to_numpy(), cfg.max_spectra)

    has_other = set(rows.loc[~is_tims & rows["key"].notna().to_numpy(), "key"])
    mol = (
        el.groupby("key", sort=True)
        .agg(smiles=("normalized_smiles", "min"), exact_mass=("exact_mass", "median"), weight=("w", "mean"),
             is_np=("is_np", "any"))
        .reset_index()
    )
    mol["c1_eligible"] = mol["key"].isin(has_other).to_numpy()
    edges = np.asarray(cfg.mass_bins)
    mol["bin"] = np.clip(np.searchsorted(edges, mol["exact_mass"].to_numpy(), side="right") - 1, 0, len(edges) - 2)
    np_bins = np.bincount(mol.loc[mol["is_np"], "bin"], minlength=len(edges) - 1) / mol["is_np"].sum()
    target_bins = (np_bins + cfg.smoothing) / (np_bins + cfg.smoothing).sum()

    # Sizes: every NP molecule, plus enveda-180 up to max_drug_share of the whole set.
    np_idx = rng.permutation(mol.index[mol["is_np"]].to_numpy())
    n_np = len(np_idx)
    n_val = int(round(n_np / (1.0 - cfg.max_drug_share)))
    s1, _, s3 = cfg.class_shares
    n1, n3 = round(s1 * n_val), round(s3 * n_val)
    n2 = n_val - n1 - n3

    # NP molecules are spread over the classes first (C1 needs non-timsTOF references).
    cls = pd.Series(0, index=mol.index)
    np_c1 = [i for i in np_idx if mol.at[i, "c1_eligible"]][: round(s1 * n_np)]
    np_rest = [i for i in np_idx if i not in set(np_c1)]
    k3 = round(s3 * n_np)
    cls.loc[np_c1] = 1
    if cfg.c2_profile is None:
        cls.loc[np_rest[:k3]] = 3
        cls.loc[np_rest[k3:]] = 2
    else:
        if nn is None:
            raise ValueError("split v2 (c2_profile) needs nn_tanimoto per key")
        nn_edges = np.asarray(cfg.nn_edges)
        mol["nn_bin"] = np.where(mol["key"].map(nn).notna(), np.clip(
            np.searchsorted(nn_edges, mol["key"].map(nn).fillna(0).to_numpy(), side="right") - 1, 0, len(nn_edges) - 2),
            -1)
        c2_quota = _bin_quotas(np.full(len(nn_edges) - 1, n2), np.asarray(cfg.c2_profile, dtype=float), n2)
        # NP molecules fill Class 2 within the per-bin quotas (up to their usual share); the rest go to C3.
        taken = np.zeros_like(c2_quota)
        np_c2: list[int] = []
        for i in np_rest:
            b = mol.at[i, "nn_bin"]
            if b >= 0 and taken[b] < c2_quota[b] and len(np_c2) < len(np_rest) - k3:
                np_c2.append(i)
                taken[b] += 1
        np_c3 = [i for i in np_rest if i not in set(np_c2)]
        overflow = max(0, len(np_c3) - n3)  # never exceed the C3 quota with NP molecules alone
        np_c2 += np_c3[:overflow]
        np_c3 = np_c3[overflow:]
        cls.loc[np_c2] = 2
        cls.loc[np_c3] = 3

    # enveda-180 fills the remaining quotas, mass-stratified towards the NP mass distribution.
    drug = mol.index[~mol["is_np"]].to_numpy()

    def draw(pool: np.ndarray, k: int, what: str) -> np.ndarray:
        if k < 0:
            raise ValueError(f"NP examples already exceed the {what} quota")
        return _stratified_draw(rng, pool, mol.loc[pool, "bin"].to_numpy(), mol.loc[pool, "weight"].to_numpy(),
                                target_bins, k, what)

    c1_pool = drug[mol.loc[drug, "c1_eligible"].to_numpy()]
    cls.loc[draw(c1_pool, n1 - int((cls == 1).sum()), "C1-eligible enveda-180")] = 1
    need2, need3 = n2 - int((cls == 2).sum()), n3 - int((cls == 3).sum())
    if cfg.c2_profile is None:
        c23 = rng.permutation(draw(drug[(cls.loc[drug] == 0).to_numpy()], need2 + need3, "enveda-180"))
        cls.loc[c23[:need3]] = 3
        cls.loc[c23[need3:]] = 2
    else:
        # enveda-180 Class 2: per-nn-bin deficits, weighted draws inside each bin.
        nb = len(cfg.nn_edges) - 1
        have = np.bincount(mol.loc[cls == 2, "nn_bin"].clip(lower=0), minlength=nb)
        deficit = np.clip(c2_quota - have, 0, None).astype(float)
        free = drug[((cls.loc[drug] == 0) & (mol.loc[drug, "nn_bin"] >= 0)).to_numpy()]
        target = deficit / deficit.sum() if deficit.sum() > 0 else np.full(nb, 1.0 / nb)
        if need2 < 0:
            raise ValueError("NP examples already exceed the C2 quota")
        c2 = _stratified_draw(rng, free, mol.loc[free, "nn_bin"].to_numpy(), mol.loc[free, "weight"].to_numpy(),
                              target, need2, "enveda-180 C2")
        cls.loc[c2] = 2
        cls.loc[draw(drug[(cls.loc[drug] == 0).to_numpy()], need3, "enveda-180")] = 3
    val = mol[cls > 0].assign(**{"class": cls[cls > 0].to_numpy()}).reset_index(drop=True)
    val["molecule_id"] = [f"v_{i:05d}" for i in range(len(val))]

    # Query spectra: NP molecules keep all eligible spectra (≤ max_spectra); enveda-180
    # molecules take a count drawn from the NP spectra-per-molecule distribution. Rows ∝ weight.
    by_key = el.groupby("key").indices
    el_rows = el.index.to_numpy()
    el_w = el["w"].to_numpy()
    query_mol = np.full(len(rows), None, dtype=object)
    n_query = []
    for key, mol_id, is_np in zip(val["key"], val["molecule_id"], val["is_np"]):
        idx = by_key[key]
        n = cfg.max_spectra if is_np else int(rng.choice(np_spm))
        take = idx[_weighted_order(rng, el_w[idx])[: min(n, len(idx))]]
        query_mol[el_rows[take]] = mol_id
        n_query.append(len(take))
    val["n_query"] = n_query

    in_library = np.ones(len(rows), dtype=bool)
    key = rows["key"]
    in_library[key.isin(set(val["key"])).to_numpy()] = False
    in_library[(key.isin(set(val.loc[val["class"] == 1, "key"])).to_numpy()) & ~is_tims] = True

    val["source"] = np.where(val["is_np"], NP_EXAMPLES, DRUG_LIB)
    val["nn_tanimoto"] = val["key"].map(nn).to_numpy() if nn is not None else np.nan
    val = val.rename(columns={"key": "inchikey14"})[V2_COLUMNS if nn is not None else VAL_COLUMNS]
    mask = pd.DataFrame({"row": np.arange(len(rows)), "in_library": in_library, "query_molecule_id": query_mol})
    return Split(val=val, library_mask=mask)


def final_split(n_rows: int) -> Split:
    """Submission mode: no validation molecules, every train spectrum (NP examples too) in the library."""
    val = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        VAL_COLUMNS, ["object", "object", "int64", "object", "object", "float64", "int64"])})
    mask = pd.DataFrame({"row": np.arange(n_rows), "in_library": np.ones(n_rows, dtype=bool),
                         "query_molecule_id": np.full(n_rows, None, dtype=object)})
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
