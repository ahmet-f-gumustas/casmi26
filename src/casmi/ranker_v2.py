"""Ranker v2: a gradient-boosted classifier over per-candidate channel features.

``candidate_features`` builds one row per (molecule, candidate) from the channel scores, the
analog hits and the pool metadata; validation (``validation/ranker_features.py``) and the Kaggle
notebook use the same function. The model is fitted from simulated validation splits
(``r_seed10``–``14``): 4 bags, each a bootstrap over molecules, of HistGradientBoostingClassifier;
the notebook fits it in place from the shipped training features, so no pickled model has to
match the Kaggle scikit-learn version.

``in_library`` comes from the library the query is searched against (the split's mask in
validation, every training spectrum on Kaggle): a Class 2/3 answer never has training spectra,
which is what the ranker learns to use when no library spectrum matches well.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from casmi.candidates import molecule_masses
from casmi.channels.analog import tanimoto
from casmi.config import TOP_K
from casmi.ranker import rank_normalise

FEATURES = ["analog", "analog_rank", "analog_pct", "fingerprint", "fingerprint_rank", "fingerprint_pct",
            "fingerprint_st", "fragment", "fragment_pct", "library_max", "library_mean", "t_analog1",
            "n_candidates", "mass", "log_sid", "log_pmid", "in_library", "in_coconut"]
POPULARITY = ("log_sid", "log_pmid")
SOURCE = ("in_library", "in_coconut")
FEATURE_SETS = {"all": FEATURES,
                "no popularity": [f for f in FEATURES if f not in POPULARITY],
                "no popularity / source": [f for f in FEATURES if f not in POPULARITY + SOURCE]}
PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
CHANNELS = ("analog", "fingerprint", "fingerprint_st", "fragment", "library", "library_max")


def _within(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Rank (1 = best, ties share the best rank) and rank-normalised value of ``col`` per molecule."""
    g = df.groupby("molecule_id", sort=False)[col]
    return (g.rank(method="min", ascending=False).to_numpy(),
            g.transform(lambda v: rank_normalise(v.to_numpy())).to_numpy())


def candidate_features(queries: pd.DataFrame, windows: Mapping[str, list[str]], channels: Mapping[str, Mapping],
                       hits: Sequence, row_key: Mapping[int, str], meta: pd.DataFrame, fps: np.ndarray,
                       in_library: set[str], popularity: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for every (molecule, candidate) in ``windows``.

    ``queries`` must be in the row order ``hits`` (from ``analog_hits``) was computed in;
    ``channels`` holds the six ``CHANNELS`` score dicts; ``meta`` / ``fps`` are the pool's.
    """
    fp_row = dict(zip(meta["key"], range(len(meta))))
    masses = molecule_masses(queries)
    top1, sim1 = {}, {}
    for m, idx in queries.groupby("molecule_id").indices.items():
        rows = {fp_row.get(row_key.get(int(hits[i][0][0]))) for i in idx if len(hits[i][0])}
        top1[m] = sorted(r for r in rows if r is not None)
        sim1[m] = max((float(hits[i][1][0]) for i in idx if len(hits[i][0])), default=0.0)
    parts = []
    for m, keys in windows.items():
        if not keys:
            continue
        cand = [fp_row[k] for k in keys]
        t_an = tanimoto(fps[cand], fps[top1[m]]).max(axis=1) if top1.get(m) else np.zeros(len(keys))
        ch = {c: channels[c].get(m, {}) for c in CHANNELS}
        parts.append(pd.DataFrame({
            "molecule_id": m, "key": keys,
            "analog": [ch["analog"].get(k, 0.0) for k in keys],
            "fingerprint": [ch["fingerprint"].get(k, 0.0) for k in keys],
            "fingerprint_st": [ch["fingerprint_st"].get(k, 0.0) for k in keys],
            "fragment": [ch["fragment"].get(k, 0.0) for k in keys],
            "library_max": [ch["library_max"].get(k, 0.0) for k in keys],
            "library_mean": [ch["library"].get(k, 0.0) for k in keys],
            "t_analog1": t_an, "sim1": sim1.get(m, 0.0), "n_candidates": len(keys), "mass": float(masses[m]),
            "in_library": [k in in_library for k in keys],
        }))
    df = pd.concat(parts, ignore_index=True)
    for col in ("analog", "fingerprint", "fragment"):
        df[f"{col}_rank"], df[f"{col}_pct"] = _within(df, col)
    df = df.merge(meta[["key", "in_coconut"]], on="key", how="left").merge(popularity, on="key", how="left")
    df["log_sid"] = np.log1p(df.pop("pc_sid").fillna(0))
    df["log_pmid"] = np.log1p(df.pop("pc_pmid").fillna(0))
    df["in_library"] = df["in_library"].astype(float)
    df["in_coconut"] = df["in_coconut"].fillna(False).astype(float)
    return df


def fit_bags(df: pd.DataFrame, feats: Sequence[str], bags: int = 4, seed: int = 0, params: Mapping | None = None,
             group: str = "answer") -> list:
    """``bags`` classifiers, each on a bootstrap of molecules (``group`` column) with its own seed."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    groups = df[group].to_numpy()
    uniq = np.unique(groups)
    x, y = df[list(feats)].to_numpy(np.float32), df["label"].to_numpy()
    models = []
    for b in range(bags):
        rng = np.random.default_rng(seed + b)
        weight = pd.Series(rng.choice(uniq, len(uniq), replace=True)).value_counts()
        w = pd.Series(groups).map(weight).fillna(0).to_numpy()
        keep = w > 0
        m = HistGradientBoostingClassifier(random_state=seed + b, **(params or PARAMS))
        m.fit(x[keep], y[keep], sample_weight=w[keep])
        models.append(m)
    return models


def predict(models: Sequence, df: pd.DataFrame, feats: Sequence[str]) -> np.ndarray:
    x = df[list(feats)].to_numpy(np.float32)
    return np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)


def rank_all(df: pd.DataFrame, score: np.ndarray, k: int = TOP_K) -> dict[str, list[str]]:
    """Top-``k`` keys per molecule by score (ties: analog score, then key)."""
    d = df[["molecule_id", "key", "analog"]].assign(s=score)
    d = d.sort_values(["molecule_id", "s", "analog", "key"], ascending=[True, False, False, True])
    return d.groupby("molecule_id", sort=False)["key"].agg(lambda s: s.head(k).tolist()).to_dict()
