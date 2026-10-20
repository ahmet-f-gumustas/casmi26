"""End-to-end ranking pipeline shared by local validation and the Kaggle notebook.

queries (one row per spectrum, with ``molecule_id``) → neutral mass per molecule →
candidate structures from the pool's ppm window → channel scores → top-25 SMILES.
Validation passes its split's queries, library mask and Class-3 exclusions; the
Kaggle notebook passes ``test.parquet`` and the full cleaned library.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from casmi.candidates import CandidatePool, molecule_masses
from casmi.channels.library import LibraryConfig, Spectrum, prepare, rank, score_molecule
from casmi.clean import strip_peaks

QUERY_COLUMNS = ["molecule_id", "ionization_mode", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
                 "instrument_type", "collision_energy_ev"]
LIBRARY_COLUMNS = ["row", "key", "ionization_mode", "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"]


@dataclass(frozen=True)
class PipelineConfig:
    ppm: float = 10.0
    offset_ppm: float = 2.0
    aggregation: str = "mean"
    workers: int = max(1, len(os.sched_getaffinity(0)))  # CPUs this process may use (container-aware)
    library: LibraryConfig = LibraryConfig()


def clean_queries(queries: pd.DataFrame) -> pd.DataFrame:
    """Apply the test-side peak rule (drop peaks above precursor + 2 Da) to query spectra."""
    lengths = queries["ms2_mzs"].map(len).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    mz = np.concatenate([np.asarray(a, dtype=np.float64) for a in queries["ms2_mzs"]]) if lengths.sum() else np.zeros(0)
    it = (np.concatenate([np.asarray(a, dtype=np.float64) for a in queries["ms2_normalized_intensities"]])
          if lengths.sum() else np.zeros(0))
    new_off, new_mz, new_it, _ = strip_peaks(offsets, mz, it, queries["precursor_mz"].to_numpy(np.float64))
    out = queries.copy()
    out["ms2_mzs"] = [new_mz[a:b] for a, b in zip(new_off[:-1], new_off[1:])]
    out["ms2_normalized_intensities"] = [new_it[a:b] for a, b in zip(new_off[:-1], new_off[1:])]
    return out


def to_spectra(df: pd.DataFrame, cfg: LibraryConfig) -> list[Spectrum]:
    return [(mode, prepare(np.asarray(mz), np.asarray(it), prec, cfg))
            for mode, mz, it, prec in zip(df["ionization_mode"], df["ms2_mzs"], df["ms2_normalized_intensities"],
                                          df["precursor_mz"])]


def candidate_windows(queries: pd.DataFrame, pool: CandidatePool, cfg: PipelineConfig) -> dict[str, list[str]]:
    """Scorer keys of the pool structures in each molecule's mass window."""
    keys = pool.meta["key"].to_numpy()
    masses = molecule_masses(queries)
    return {m: keys[pool.window(masses[m], cfg.ppm, cfg.offset_ppm)].tolist() for m in masses.index}


def load_references(library: Path, keys: Iterable[str], in_library: pd.Series | None = None,
                    cfg: LibraryConfig = LibraryConfig()) -> dict[str, list[Spectrum]]:
    """Prepared reference spectra per scorer key from the cleaned library parquet.

    ``in_library`` (indexed by train ``row``) restricts the library to a split's mask.
    """
    table = ds.dataset(library).to_table(columns=LIBRARY_COLUMNS, filter=pc.field("key").isin(list(set(keys))))
    lib = table.to_pandas()
    if in_library is not None:
        lib = lib[lib["row"].map(in_library).fillna(False).to_numpy(dtype=bool)]
    return {key: to_spectra(grp, cfg) for key, grp in lib.groupby("key")}


# Worker state, inherited through fork.
_STATE: dict = {}


def _score(mol_id: str) -> tuple[str, pd.DataFrame]:
    s = _STATE
    return mol_id, score_molecule(s["queries"].get(mol_id, []), s["windows"].get(mol_id, []), s["refs"], s["cfg"])


def library_scores(queries: pd.DataFrame, windows: Mapping[str, list[str]], refs: Mapping[str, list[Spectrum]],
                   cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Library-channel score table per molecule."""
    _STATE.update(queries={m: to_spectra(g, cfg.library) for m, g in queries.groupby("molecule_id")},
                  windows=windows, refs=refs, cfg=cfg.library)
    ids = list(windows)
    if cfg.workers > 1:
        with Pool(cfg.workers) as pool:
            out = dict(pool.map(_score, ids, chunksize=1))
    else:
        out = dict(map(_score, ids))
    _STATE.clear()
    return out


def predictions(scores: Mapping[str, pd.DataFrame], smiles_of: Mapping[str, str], by: str = "mean") -> dict[str, list[str]]:
    """Top-25 SMILES per molecule from its score table."""
    return {m: [smiles_of[k] for k in rank(s, by)] for m, s in scores.items()}


def run_library_pipeline(queries: pd.DataFrame, pool: CandidatePool, library: Path, cfg: PipelineConfig = PipelineConfig(),
                         in_library: pd.Series | None = None) -> tuple[dict[str, list[str]], dict[str, pd.DataFrame]]:
    """Library-only pipeline: returns ``(predictions, score tables)``."""
    queries = clean_queries(queries)
    windows = candidate_windows(queries, pool, cfg)
    refs = load_references(library, {k for ks in windows.values() for k in ks}, in_library, cfg.library)
    scores = library_scores(queries, windows, refs, cfg)
    smiles_of = dict(zip(pool.meta["key"], pool.meta["smiles"]))
    return predictions(scores, smiles_of, cfg.aggregation), scores


# --------------------------------------------------------------------------------------
# Analog and fragment channels, and the ranker-v0 combination.
# --------------------------------------------------------------------------------------
ChannelScores = dict[str, dict[str, float]]  # molecule_id -> {key: score}


def _analog_search(i: int) -> tuple[np.ndarray, np.ndarray]:
    s = _STATE
    mode, prec, peaks = s["raw"][i]
    return s["index"].search(mode, prec, peaks, s["cfg"], s["allowed"])


def analog_hits(queries: pd.DataFrame, index, cfg=None, allowed: np.ndarray | None = None,
                workers: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Top-K analog (library rows, similarities) per query spectrum, in ``queries`` row order."""
    from casmi.channels.analog import AnalogConfig

    cfg = cfg or AnalogConfig()
    raw = [(m, float(p), np.column_stack([mz, it]).astype(np.float32))
           for m, p, mz, it in zip(queries["ionization_mode"], queries["precursor_mz"], queries["ms2_mzs"],
                                   queries["ms2_normalized_intensities"])]
    _STATE.update(raw=raw, index=index, cfg=cfg, allowed=allowed)
    if workers > 1:
        with Pool(workers) as p:
            hits = p.map(_analog_search, range(len(raw)), chunksize=4)
    else:
        hits = [_analog_search(i) for i in range(len(raw))]
    _STATE.clear()
    return hits


def analog_channel(queries: pd.DataFrame, windows: Mapping[str, list[str]], pool: CandidatePool, index,
                   row_key: Mapping[int, str], cfg=None, allowed: np.ndarray | None = None,
                   workers: int = 1, hits: list | None = None) -> ChannelScores:
    """Mass-shifted analog propagation scores (see ``casmi.channels.analog``), mean over spectra.

    ``hits`` (from ``analog_hits`` on the same ``queries``) skips the library search;
    ``cfg.self_min_sim`` limits a candidate's own spectra as its analogs.
    """
    from casmi.channels.analog import AnalogConfig, score_molecule as analog_score

    cfg = cfg or AnalogConfig()
    queries = queries.reset_index(drop=True)
    if hits is None:
        hits = analog_hits(queries, index, cfg, allowed, workers)
    fp_row_of_key = dict(zip(pool.meta["key"], range(len(pool.meta))))
    used = {int(r) for rows, _ in hits for r in rows}
    row_fp = {r: fp_row_of_key[row_key[r]] for r in used if row_key.get(r) in fp_row_of_key}
    fps = pool.fps
    out: ChannelScores = {}
    for mol_id, idx in queries.groupby("molecule_id").indices.items():
        keys = windows.get(mol_id, [])
        if not keys:
            continue
        rows = np.array([fp_row_of_key[k] for k in keys])
        per = analog_score([hits[i] for i in idx], fps[rows], row_fp, fps, (cfg.power,), top_k=cfg.top_k,
                           candidate_rows=rows, self_min_sim=cfg.self_min_sim)[cfg.power]
        out[mol_id] = dict(zip(keys, per.mean(axis=0).tolist()))
    return out


def _fragments(smiles: str):
    from casmi.channels.fragment import fragment_masses

    return fragment_masses(smiles, _STATE["fcfg"])


def fragment_channel(queries: pd.DataFrame, windows: Mapping[str, list[str]], smiles_of: Mapping[str, str],
                     cfg=None, workers: int = 1, cache: dict | None = None) -> ChannelScores:
    """Explained-intensity scores from 1–2 bond-break fragments, mean over spectra."""
    from casmi.channels.fragment import FragmentConfig, explained_intensity, ion_masses

    cfg = cfg or FragmentConfig()
    cache = {} if cache is None else cache
    todo = sorted({k for ks in windows.values() for k in ks} - set(cache))
    _STATE.update(fcfg=cfg)
    if todo:
        if workers > 1:
            with Pool(workers) as p:
                frags = p.map(_fragments, [smiles_of[k] for k in todo], chunksize=50)
        else:
            frags = [_fragments(smiles_of[k]) for k in todo]
        cache.update(zip(todo, frags))
    _STATE.clear()
    out: ChannelScores = {}
    for mol_id, grp in queries.groupby("molecule_id"):
        keys = windows.get(mol_id, [])
        specs = to_spectra(grp, LibraryConfig())
        if not keys or not specs:
            continue
        mat = np.array([[explained_intensity(ion_masses(cache[k], mode, cfg), peaks, cfg.tol_da) for k in keys]
                        for mode, peaks in specs])
        out[mol_id] = dict(zip(keys, mat.mean(axis=0).tolist()))
    return out


def library_channel(queries: pd.DataFrame, windows: Mapping[str, list[str]], library: Path, cfg: PipelineConfig,
                    in_library: pd.Series | None = None, both: bool = False):
    """Library-channel scores (aggregation from ``cfg``) per molecule.

    With ``both``, returns ``(scores, max_scores)``: the configured aggregation and the best match
    over the molecule's spectra (used by the library gate).
    """
    refs = load_references(library, {k for ks in windows.values() for k in ks}, in_library, cfg.library)
    scores = library_scores(queries, windows, refs, cfg)
    agg = {m: dict(zip(s["key"], s[cfg.aggregation].astype(float))) for m, s in scores.items() if len(s)}
    if not both:
        return agg
    return agg, {m: dict(zip(s["key"], s["max"].astype(float))) for m, s in scores.items() if len(s)}


def fingerprint_logits(queries: pd.DataFrame, windows: Mapping[str, list[str]], model) -> dict[str, np.ndarray]:
    """Per-bit logits per molecule (``model.logits`` over its spectra) for molecules with candidates."""
    from casmi import degraded

    logits = {}
    for mol_id, grp in queries.groupby("molecule_id"):
        if windows.get(mol_id):
            z = model.logits(grp)
            if z is not None:
                logits[mol_id] = z
            else:
                degraded.note("molecule without fingerprint logits")
    return logits


def fingerprint_scores(queries: pd.DataFrame, windows: Mapping[str, list[str]], smiles_of: Mapping[str, str],
                       model, workers: int = 1, cache: dict | None = None, how: str = "dot") -> ChannelScores:
    """Fingerprint-channel scores; ``model`` has ``logits(spectra) -> z``, ``layout`` and ``bits``."""
    from casmi.channels.fingerprint import fingerprint_channel

    return fingerprint_channel(fingerprint_logits(queries, windows, model), windows, smiles_of, model.layout,
                               model.bits, workers=workers, cache=cache, how=how)


def pubchem_join(queries: pd.DataFrame, windows: Mapping[str, list[str]], pool: CandidatePool, index,
                 row_key: Mapping[int, str], hits: list, logits: Mapping[str, np.ndarray] | None, fp_model,
                 top: int, allowed: np.ndarray | None = None,
                 workers: int = 1) -> tuple[dict[str, list[str]], set[str]]:
    """Base-pool candidates (train ∪ COCONUT) plus the ``top`` PubChem-only candidates per molecule.

    ``pool`` is the PubChem pool (NP-likeness > 0 already). PubChem-only candidates of a window are
    pre-screened by z(analog) + z(f·z) within that part of the window (z per molecule; only the analog
    term without a model). Returns the joined windows and the joined PubChem keys.
    """
    from casmi.channels.fingerprint import fingerprint_channel

    meta = pool.meta
    base = set(meta.loc[meta["in_train"] | meta["in_coconut"], "key"])
    pub = {m: [k for k in ks if k not in base] for m, ks in windows.items()}
    smiles_of = dict(zip(meta["key"], meta["smiles"]))
    a = analog_channel(queries, pub, pool, index, row_key, allowed=allowed, hits=hits)
    f = (fingerprint_channel(logits, pub, smiles_of, fp_model.layout, fp_model.bits, workers=workers)
         if logits else {})

    def z(v: np.ndarray) -> np.ndarray:
        s = v.std()
        return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)

    out, joined = {}, set()
    for m, ks in windows.items():
        p = pub[m]
        keep: list[str] = []
        if p:
            s = z(np.array([a.get(m, {}).get(k, 0.0) for k in p]))
            if m in f:
                s = s + z(np.array([f[m].get(k, 0.0) for k in p]))
            keep = [p[i] for i in np.argsort(-s, kind="stable")[:top]]
        out[m] = [k for k in ks if k in base] + keep
        joined.update(keep)
    return out, joined


def run_combined_pipeline(queries: pd.DataFrame, pool: CandidatePool, library: Path, index, row_key: Mapping[int, str],
                          ranker: Mapping, cfg: PipelineConfig = PipelineConfig(),
                          in_library: pd.Series | None = None, allowed: np.ndarray | None = None,
                          fp_model=None, tier: Mapping | None = None, alt_adducts: Mapping | None = None,
                          ) -> tuple[dict[str, list[str]], dict[str, ChannelScores]]:
    """Channels needed by ``ranker`` (a ranker-v0 weight dict or a ranker-v1 config), then ranking.

    ``fp_model`` (see ``fingerprint_scores``) is required when the ranker reads the fingerprint channel.
    ``tier`` (``casmi.tier.TierConfig`` fields) turns on the PubChem tier: ``pool`` is then the PubChem
    pool, the ranker sees only its base structures (``in_train`` or ``in_coconut``), and PubChem-only
    candidates may take the last slots (popularity from ``popularity.parquet`` next to the pool).
    ``alt_adducts`` (``casmi.tier.slot_merge`` arguments ``n_slots``, ``gate_k``, ``placement``) lets
    candidates of the alternative-adduct windows (``casmi.adducts``) into gated slots, scored by the
    analog channel on the main list's scale; it needs an analog-only ranker.
    """
    from casmi.ranker import needed_channels, rank_molecule

    cfg_r = ranker if "mode" in ranker else {"mode": "weighted", "weights": dict(ranker)}
    if cfg_r["mode"] in ("learned", "combiner", "rank31"):
        return run_learned_pipeline(queries, pool, library, index, row_key, cfg_r, cfg, fp_model, in_library, allowed)
    need = needed_channels(cfg_r)
    queries = clean_queries(queries)
    windows = candidate_windows(queries, pool, cfg)
    smiles_of = dict(zip(pool.meta["key"], pool.meta["smiles"]))
    channels: dict[str, ChannelScores] = {}
    if need & {"library", "library_max"}:
        channels["library"], channels["library_max"] = library_channel(queries, windows, library, cfg, in_library,
                                                                       both=True)
    hits = None
    if "analog" in need:
        hits = analog_hits(queries.reset_index(drop=True), index, allowed=allowed, workers=cfg.workers)
        channels["analog"] = analog_channel(queries, windows, pool, index, row_key, allowed=allowed, hits=hits)
    if "fragment" in need:
        channels["fragment"] = fragment_channel(queries, windows, smiles_of, workers=cfg.workers)
    if "fingerprint" in need:
        if fp_model is None:
            raise ValueError("the ranker reads the fingerprint channel but no fp_model was given")
        from casmi.channels.fingerprint import fingerprint_channel

        t0 = time.time()
        logits = fingerprint_logits(queries, windows, fp_model)
        t1 = time.time()
        channels["fingerprint"] = fingerprint_channel(logits, windows, smiles_of, fp_model.layout, fp_model.bits,
                                                      workers=cfg.workers)
        print(f"fingerprint channel: logits for {len(queries)} spectra / {len(logits)} molecules in {t1 - t0:.1f}s, "
              f"candidates scored in {time.time() - t1:.1f}s", flush=True)
    preds = {}
    alt = None
    if alt_adducts is not None:
        from casmi.adducts import alt_windows

        if need != {"analog"} or tier is not None:
            raise ValueError("alternative adducts need an analog-only ranker and no PubChem tier")
        alt_win = alt_windows(queries, pool, windows, cfg.ppm, cfg.offset_ppm)
        alt = analog_channel(queries, alt_win, pool, index, row_key, allowed=allowed, hits=hits)
    if tier is None:
        from casmi.tier import slot_merge

        n_used = 0
        for mol_id, keys in windows.items():
            per = {c: channels[c].get(mol_id, {}) for c in channels}
            ranked = rank_molecule(keys, per, cfg_r)
            if alt is not None:
                merged = slot_merge(ranked, per["analog"], alt.get(mol_id, {}), **alt_adducts)
                n_used += merged != ranked[: len(merged)]
                ranked = merged
            preds[mol_id] = [smiles_of[k] for k in ranked]
        if alt is not None:
            print(f"alternative adducts {dict(alt_adducts)}: {len(alt)} molecules with alternative candidates, "
                  f"used for {n_used} of {len(windows)}", flush=True)
        return preds, channels

    from casmi.tier import TierConfig, pub_candidates, tier_merge, union_scores

    tcfg = TierConfig(**tier)
    if cfg_r.get("mode", "weighted") != "weighted":
        raise ValueError("the PubChem tier needs a weighted ranker (its weights give the common scale)")
    meta = pool.meta
    base = set(meta.loc[meta["in_train"] | meta["in_coconut"], "key"])
    pop = pd.read_parquet(pool.path / "popularity.parquet")
    sid = dict(zip(pop["key"], pop["pc_sid"]))
    n_used = 0
    for mol_id, keys in windows.items():
        per = {c: channels[c].get(mol_id, {}) for c in channels}
        main = rank_molecule([k for k in keys if k in base], per, cfg_r)
        common = union_scores(keys, per, cfg_r["weights"])
        pub = pub_candidates([k for k in keys if k not in base], common, sid, tcfg)
        final = tier_merge(main, pub, common, tcfg)
        n_used += final != main[: len(final)]
        preds[mol_id] = [smiles_of[k] for k in final]
    print(f"PubChem tier {tcfg}: used for {n_used} of {len(windows)} molecules", flush=True)
    return preds, channels


def forward_stage(spec: Mapping, models: Mapping | None, queries: pd.DataFrame, frame: pd.DataFrame,
                  score: np.ndarray, smiles_of: Mapping[str, str]) -> np.ndarray:
    """The ranker's ``score`` with its top ``spec["top"]`` re-ordered by forward-model scores (``casmi.forward``).

    ``spec``: ``{"models": [names], "fusion": "rrf", "top": K, "rrf_k": k, "budget_s": seconds}``; ``models``
    maps the names to loaded predictors. Molecules are scored from the weakest library match up, so a spent
    budget leaves the molecules the library already identifies in the ranker's order. The budget is shared:
    each model gets what the ones before it left.
    """
    from casmi.forward import score as fs

    names = list(spec["models"])
    missing = [n for n in names if n not in (models or {})]
    if missing:
        raise ValueError(f"forward models {missing} were not loaded")
    if spec.get("fusion", "rrf") != "rrf":
        raise ValueError(f"unknown forward fusion {spec['fusion']!r}")
    top = int(spec.get("top", fs.TOP_K))
    mol, keys = frame["molecule_id"].to_numpy(), frame["key"].to_numpy()
    heads = fs.top_keys(mol, keys, score, top)
    lib = frame.groupby("molecule_id")["lv_max"].max()
    order = sorted(heads, key=lambda m: (lib.get(m, 0.0), m))
    budget, t0, cols = spec.get("budget_s"), time.time(), []
    for name in names:
        left = None if budget is None else max(float(budget) - (time.time() - t0), 0.0)
        scores, info = fs.forward_scores(models[name], queries, heads, smiles_of, order=order, budget_s=left)
        cols.append(fs.column(mol, keys, scores))
        print(f"forward {name}: {info}", flush=True)
    return fs.rrf(mol, score, cols, int(spec.get("rrf_k", fs.RRF_K)), top)


def run_learned_pipeline(queries: pd.DataFrame, pool: CandidatePool, library: Path, index, row_key: Mapping[int, str],
                         ranker: Mapping, cfg: PipelineConfig = PipelineConfig(), fp_model=None,
                         in_library: pd.Series | None = None, allowed: np.ndarray | None = None,
                         ) -> tuple[dict[str, list[str]], dict[str, ChannelScores]]:
    """Every channel and per-candidate features, then a learned ranker or a combiner.

    ``ranker``: ``{"mode": "learned", "features": [...], "bags": int, "params": {...}, "train": DataFrame}``
    (ranker v2: a classifier fitted on training rows with ``label`` and ``answer``),
    ``{"mode": "combiner", "spec": {...}}`` (``casmi.combiners.combine_rows``) or
    ``{"mode": "rank31", "train": (X, Y, M), ["features": [...], "priors": [...], "seeds": [...],
    "pubchem_join": {"top": N}, "popularity": {"mu": μ}]}`` (``casmi.rank31``: the V33 features and
    ranker, fitted on rows in ``rank_train.npz`` format; with ``pubchem_join`` and the PubChem pool,
    ``pubchem_join`` adds N PubChem-only candidates per molecule and the ranker gets their source as
    ``src_pubchem``; ``popularity`` applies V33's popularity re-order to the ranker's probabilities;
    ``forward`` re-orders the top candidates by forward-model scores, see ``forward_stage``, with the loaded
    predictors in ``ranker["forward_models"]``); popularity is read from ``popularity.parquet`` next to the pool.
    """
    from casmi import rank31
    from casmi.channels.analog import AnalogConfig
    from casmi.channels.fingerprint import fingerprint_channel
    from casmi.combiners import combine_rows
    from casmi.ranker_v2 import candidate_features, fit_bags, predict, rank_all

    if fp_model is None and (ranker["mode"] != "combiner" or ranker["spec"].get("fp", "fingerprint") is not None):
        raise ValueError("the ranker reads the fingerprint channel but no fp_model was given")
    from casmi import degraded

    t0 = time.time()
    queries = clean_queries(queries).reset_index(drop=True)
    windows = candidate_windows(queries, pool, cfg)
    degraded.note("molecule without candidates in its window", sum(1 for v in windows.values() if not v))
    smiles_of = dict(zip(pool.meta["key"], pool.meta["smiles"]))
    # rank31 builds its analog list from more hits per spectrum; the analog channel reads the first top_k
    acfg = AnalogConfig(top_k=rank31.ANALOG_TOP_K) if ranker["mode"] == "rank31" else None
    hits = analog_hits(queries, index, cfg=acfg, allowed=allowed, workers=cfg.workers)
    logits = fingerprint_logits(queries, windows, fp_model) if fp_model is not None else None
    join = ranker.get("pubchem_join") if ranker["mode"] == "rank31" else None
    joined: set[str] = set()
    if join:
        n_before = sum(map(len, windows.values()))
        windows, joined = pubchem_join(queries, windows, pool, index, row_key, hits, logits, fp_model,
                                       int(join["top"]), allowed, cfg.workers)
        print(f"PubChem join: {len(joined):,} PubChem-only candidates added (top {join['top']} per molecule); "
              f"{n_before:,} → {sum(map(len, windows.values())):,} window rows, {time.time() - t0:.0f}s", flush=True)
    print(f"analog hits and fingerprint logits: {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    channels: dict[str, ChannelScores] = {
        "analog": analog_channel(queries, windows, pool, index, row_key, allowed=allowed, hits=hits)}
    if ranker["mode"] != "rank31":  # rank31 computes its own fragment and model features
        channels["fragment"] = fragment_channel(queries, windows, smiles_of, workers=cfg.workers)
        channels["fingerprint"], channels["fingerprint_st"] = {}, {}
        if fp_model is not None:
            for name, how in (("fingerprint", "dot"), ("fingerprint_st", "soft_tanimoto")):
                channels[name] = fingerprint_channel(logits, windows, smiles_of, fp_model.layout, fp_model.bits,
                                                     workers=cfg.workers, how=how)
    channels["library"], channels["library_max"] = library_channel(queries, windows, library, cfg, in_library,
                                                                   both=True)
    print(f"feature channels ({', '.join(channels)}): {time.time() - t1:.0f}s", flush=True)
    if ranker["mode"] == "rank31":
        frame = rank31.frame(queries, windows, smiles_of, hits, row_key, channels["library_max"], logits,
                             fp_model.layout, fp_model.bits, workers=cfg.workers)
        names = list(ranker.get("features", rank31.USED))
        X, Y, M = ranker["train"]
        X = X[:, rank31.columns(names)]
        if join:  # the source as a feature; the training rows hold no PubChem-only candidate (column of 0)
            frame[rank31.SOURCE] = frame["key"].isin(joined).astype(np.float32)
            names = [*names, rank31.SOURCE]
            X = np.column_stack([X, np.zeros(len(X), np.float32)])
        models = rank31.fit(X, Y, M, ranker.get("priors", rank31.PRIORS), ranker.get("seeds", rank31.SEEDS))
        analog = channels["analog"]
        frame["analog"] = [analog.get(m, {}).get(k, 0.0) for m, k in zip(frame["molecule_id"], frame["key"])]
        score = rank31.proba(models, frame[names].to_numpy(np.float32))
        if ranker.get("popularity"):
            pop = rank31.popularity_columns(frame["key"].tolist(), pd.read_parquet(pool.path / "popularity.parquet"))
            score = rank31.popularity_scores(frame["molecule_id"].to_numpy(), score, pop.sum(axis=1),
                                             frame["lv_max"].to_numpy(), float(ranker["popularity"]["mu"]))
            print(f"popularity re-order: μ {ranker['popularity']['mu']}, top {rank31.POP_TOP}, "
                  f"library gate {rank31.LIB_GATE}", flush=True)
        print(f"rank31: {len(frame):,} candidate rows, {len(models)} models fitted on {len(Y):,} rows "
              f"({len(names)} features), {time.time() - t0:.0f}s", flush=True)
        if ranker.get("forward"):
            score = forward_stage(ranker["forward"], ranker.get("forward_models"), queries, frame, score, smiles_of)
            print(f"forward stage done, {time.time() - t0:.0f}s", flush=True)
        ranked = rank_all(frame, score)
    else:
        lib = pd.read_parquet(library, columns=["row", "key"])
        if in_library is not None:
            lib = lib[lib["row"].map(in_library).fillna(False).to_numpy(dtype=bool)]
        pop = pd.read_parquet(pool.path / "popularity.parquet")
        df = candidate_features(queries, windows, channels, hits, row_key, pool.meta, pool.fps, set(lib["key"]), pop)
        if ranker["mode"] == "combiner":
            ranked = rank_all(df, combine_rows(df, ranker["spec"]))
            print(f"combiner {ranker['spec']['kind']}: {len(df):,} candidate rows, {time.time() - t0:.0f}s",
                  flush=True)
        else:
            feats = list(ranker["features"])
            models = fit_bags(ranker["train"], feats, int(ranker.get("bags", 4)), params=ranker.get("params"))
            ranked = rank_all(df, predict(models, df, feats))
            print(f"ranker v2: {len(df):,} candidate rows, {len(models)} bags fitted on {len(ranker['train']):,} "
                  f"rows, {time.time() - t0:.0f}s", flush=True)
    return {m: [smiles_of[k] for k in ranked.get(m, [])] for m in windows}, channels
