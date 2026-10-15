"""Analog + fingerprint (+ fragment) combiners over per-candidate feature rows.

Every function takes the rows of ``casmi.ranker_v2.candidate_features`` (one per molecule ×
candidate, ``molecule_id`` grouped) and returns one score per row; higher ranks first, ties are
broken by the analog score, then the key (``casmi.ranker_v2.rank_all``).

- ``topk_rerank``: the analog top-K is re-ranked by α·z(analog) + (1 − α)·z(fp), z-scores taken
  over the whole window; candidates outside the top-K keep their analog order behind it.
- ``product``: softmax(analog / T1) × softmax(fp / T2). The softmax normalisers are constant within
  a molecule, so this ranks by analog / T1 + fp / T2: only T2 / T1 matters.
- ``mixed``: analog / T1 + w·z(fp), the product with T2 set to the fingerprint spread of the window,
  so the same configuration fits fingerprint models of any scale.
- ``Logistic``: logistic regression on z-scores, log ranks, the best analog spectral similarity,
  the candidate's Tanimoto to the top analog and the window size, with the molecule-level terms
  entering as interactions (alone they cannot reorder a window). Coefficients are plain floats,
  so the notebook needs no pickled model.
- ``insert_top1``: Class 1 protection that changes one slot: the window's best library match, when
  its similarity is ≥ τ, goes first; the other candidates keep their order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TOP = 1e6


def zscore(df: pd.DataFrame, col: str) -> np.ndarray:
    """``col`` standardised within each molecule's window (0 where the window has no spread)."""
    g = df.groupby("molecule_id", sort=False)[col]
    sd = g.transform("std", ddof=0).to_numpy()
    z = (df[col].to_numpy() - g.transform("mean").to_numpy()) / np.where(sd > 0, sd, 1.0)
    return np.where(sd > 0, z, 0.0)


def analog_rank(df: pd.DataFrame) -> np.ndarray:
    """1 = best analog score in the window (ties: key)."""
    order = df.assign(_i=np.arange(len(df))).sort_values(["molecule_id", "analog", "key"],
                                                          ascending=[True, False, True])
    rank = np.empty(len(df))
    rank[order["_i"].to_numpy()] = order.groupby("molecule_id", sort=False).cumcount().to_numpy() + 1
    return rank


def topk_rerank(df: pd.DataFrame, k: int, alpha: float, fp: str = "fingerprint",
                extra: Mapping[str, float] | None = None) -> np.ndarray:
    """Analog top-``k`` re-ranked by α·z(analog) + (1 − α)·z(``fp``) (+ Σ w·z(col) for ``extra``)."""
    s = alpha * zscore(df, "analog") + (1 - alpha) * zscore(df, fp)
    for col, w in (extra or {}).items():
        s = s + w * zscore(df, col)
    r = analog_rank(df)
    return np.where(r <= k, TOP + s, -r)


def product(df: pd.DataFrame, t1: float, t2: float, fp: str = "fingerprint",
            extra: Mapping[str, float] | None = None) -> np.ndarray:
    """Rank score of softmax(analog / t1) × softmax(fp / t2) (× softmax(col / T) for ``extra`` col → T)."""
    s = df["analog"].to_numpy() / t1 + df[fp].to_numpy() / t2
    for col, t in (extra or {}).items():
        s = s + df[col].to_numpy() / t
    return s


def mixed(df: pd.DataFrame, t1: float, w: float, fp: str = "fingerprint",
          extra: Mapping[str, float] | None = None) -> np.ndarray:
    """analog / t1 + w·z(``fp``) (+ Σ w·z(col) for ``extra``): the product with the fingerprint factor's
    temperature set to its spread in the window, so any fingerprint model's scale fits."""
    s = df["analog"].to_numpy() / t1 + w * zscore(df, fp)
    for col, wc in (extra or {}).items():
        s = s + wc * zscore(df, col)
    return s


LOGIT_BASE = ("analog_z", "fp_z", "analog_logrank", "fp_logrank", "t_analog1",
              "analog_z*sim1", "fp_z*sim1", "analog_z*logn", "fp_z*logn")
LOGIT_FRAGMENT = ("fragment_z", "fragment_logrank")


def logit_features(df: pd.DataFrame, fp: str = "fingerprint", fragment: bool = False) -> pd.DataFrame:
    """Design matrix of ``Logistic`` (columns ``LOGIT_BASE`` [+ ``LOGIT_FRAGMENT``])."""
    az, fz = zscore(df, "analog"), zscore(df, fp)
    sim1, logn = df["sim1"].to_numpy(), np.log(df["n_candidates"].to_numpy())
    x = pd.DataFrame({"analog_z": az, "fp_z": fz, "analog_logrank": np.log(df["analog_rank"].to_numpy()),
                      "fp_logrank": np.log(df[f"{fp}_rank"].to_numpy() if f"{fp}_rank" in df else
                                           df.groupby("molecule_id")[fp].rank(method="min", ascending=False)),
                      "t_analog1": df["t_analog1"].to_numpy(), "analog_z*sim1": az * sim1, "fp_z*sim1": fz * sim1,
                      "analog_z*logn": az * logn, "fp_z*logn": fz * logn}, index=df.index)
    if fragment:
        x["fragment_z"] = zscore(df, "fragment")
        x["fragment_logrank"] = np.log(df["fragment_rank"].to_numpy())
    return x


@dataclass
class Logistic:
    """Standardised logistic regression with its coefficients as plain floats."""

    columns: list[str]
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    fp: str = "fingerprint"

    @property
    def fragment(self) -> bool:
        return "fragment_z" in self.columns

    def fit(self, df: pd.DataFrame, c: float = 1.0) -> Logistic:
        from sklearn.linear_model import LogisticRegression

        x = logit_features(df, self.fp, self.fragment)[self.columns].to_numpy(np.float64)
        self.mean, sd = x.mean(axis=0), x.std(axis=0)
        self.scale = np.where(sd > 0, sd, 1.0)
        m = LogisticRegression(C=c, max_iter=2000).fit((x - self.mean) / self.scale, df["label"].to_numpy())
        self.mean, self.scale = self.mean.tolist(), self.scale.tolist()
        self.coef, self.intercept = m.coef_[0].tolist(), float(m.intercept_[0])
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        x = logit_features(df, self.fp, self.fragment)[self.columns].to_numpy(np.float64)
        return ((x - np.asarray(self.mean)) / np.asarray(self.scale)) @ np.asarray(self.coef) + self.intercept

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "mean": list(self.mean), "scale": list(self.scale),
                "coef": list(self.coef), "intercept": self.intercept, "fp": self.fp}


def insert_top1(df: pd.DataFrame, score: np.ndarray, tau: float) -> np.ndarray:
    """``score`` with each window's best library match (if ≥ ``tau``) moved to the first slot."""
    lib = df["library_max"].to_numpy()
    best = df.assign(_l=lib, _i=np.arange(len(df))).sort_values(["molecule_id", "_l", "key"],
                                                               ascending=[True, False, True])
    first = best.groupby("molecule_id", sort=False).head(1)
    rows = first.loc[first["_l"] >= tau, "_i"].to_numpy()
    out = np.asarray(score, dtype=float).copy()
    out[rows] = np.inf
    return out


def combine_rows(df: pd.DataFrame, spec: Mapping) -> np.ndarray:
    """Score rows by a combiner spec: ``{"kind": "topk" | "product" | "logistic" | "rank_average", ...}``,
    optionally with ``"insert_tau"`` (``insert_top1``)."""
    kind = spec["kind"]
    fp = spec.get("fp", "fingerprint")
    if kind == "topk":
        s = topk_rerank(df, int(spec["k"]), float(spec["alpha"]), fp, spec.get("extra"))
    elif kind == "product":
        s = product(df, float(spec["t1"]), float(spec["t2"]), fp, spec.get("extra"))
    elif kind == "mixed":
        s = mixed(df, float(spec["t1"]), float(spec["w"]), fp, spec.get("extra"))
    elif kind == "logistic":
        s = Logistic(**spec["model"]).score(df)
    elif kind == "rank_average":
        s = df["analog_pct"].to_numpy() + df[f"{fp}_pct"].to_numpy()
    elif kind == "analog":
        s = df["analog"].to_numpy()
    else:
        raise ValueError(f"unknown combiner {kind!r}")
    return insert_top1(df, s, float(spec["insert_tau"])) if spec.get("insert_tau") is not None else s

