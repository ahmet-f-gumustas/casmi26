"""Combiners of the analog and fingerprint channels on split v4 (items 3–5 of the v4 plan).

Feature rows come from ``validation/ranker_features.py --out <dir>`` (one folder per fingerprint
model): validation on v4_seed0–2, training rows (logistic combiner, HGB ranker) from r4_seed10–14.

- baselines: analog only, fingerprint only, fragment only, rank average (analog + fingerprint);
- (a) analog top-K re-rank by α·z(analog) + (1 − α)·z(fp), K ∈ {25, 50, 100}, α ∈ {0.5, 0.7, 0.85};
- (b) softmax(analog / T1) × softmax(fp / T2) (ranks by analog / T1 + fp / T2);
- (c) logistic combiner (``casmi.combiners.Logistic``) fitted on the r4 rows;
- the best of (a)–(c) by NP C2, with the fragment channel added;
- Class 1 protection by rank-1 insertion (``insert_top1``, τ ∈ {0.6, 0.7, 0.8}) on the best combiner:
  the chosen τ loses ≤ ``--c1-loss`` C1 against analog only and no measurable C2;
- ranker v3: bagged HGB without popularity / source features (fragment and fingerprint features in),
  fitted on the r4 rows; kept over the logistic combiner only with ≥ 0.015 more NP C2.
- every NP example is a validation molecule in every split, so the learned models (logistic, HGB) are
  fitted per validation split without the rows whose answer is one of that split's answers. The library
  similarity features are 0 exactly for candidates without library spectra, so they leak the source;
  the main variant drops them, a reference variant keeps them.

MRR@25 for C1 / C2 / NP C2 (total alongside), mean ± std over seeds; Δ vs rank average and vs analog
only with the noise labels (C1 ±0.04, C2 / NP C2 ±0.02). Results: ``notes/EXP_combiners_<split>.md``,
``outputs/combiners_<split>.json`` (the chosen configuration per model).

    python validation/eval_combiners.py --models e5=outputs/ranker_features_e5 e6=outputs/ranker_features_e6
    python validation/eval_combiners.py --models e5=outputs/ranker_features_e5 --val-prefix v3_seed --train-prefix r3_seed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.combiners import LOGIT_BASE, LOGIT_FRAGMENT, Logistic, combine_rows, insert_top1
from casmi.config import REPO_ROOT, TOP_K, WORK_DIR
from casmi.ranker_v2 import FEATURE_SETS, PARAMS, fit_bags, predict
from casmi.report import md_table
from casmi.split import NP_EXAMPLES

NOISE = {"C1": 0.04, "C2": 0.02, "NP C2": 0.02, "total": 0.015}
# library_max / library_mean are 0 exactly when a candidate has no library spectra, so they carry the
# source too (the v2 shortcut); the main ranker v3 drops them, the "+ library" variant is a reference.
HGB_LIBRARY = [*FEATURE_SETS["no popularity / source"], "sim1"]
HGB_FEATURES = [f for f in HGB_LIBRARY if f not in ("library_max", "library_mean")]
TAUS = (0.6, 0.7, 0.8)


def load(folder: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(folder / f"{split}.parquet")
    val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
    df["answer"] = df["molecule_id"].map(dict(zip(val["molecule_id"], val["inchikey14"])))
    return df, val


def class_mrr(df: pd.DataFrame, score: np.ndarray, val: pd.DataFrame) -> dict:
    """MRR@25 per class over every validation molecule (ties: analog score, then key)."""
    d = df[["molecule_id", "key", "label", "analog"]].assign(s=score)
    d = d.sort_values(["molecule_id", "s", "analog", "key"], ascending=[True, False, False, True])
    d["pos"] = d.groupby("molecule_id").cumcount() + 1
    hit = d[d["label"] & (d["pos"] <= TOP_K)].set_index("molecule_id")["pos"]
    rr = val["molecule_id"].map(1.0 / hit).fillna(0.0)
    c2 = val["class"] == 2
    return {"C1": float(rr[val["class"] == 1].mean()), "C2": float(rr[c2].mean()),
            "NP C2": float(rr[c2 & (val["source"] == NP_EXAMPLES)].mean()), "total": float(rr.mean())}


def grid() -> dict[str, dict]:
    out = {"analog only": {"kind": "analog"}, "rank average (analog + FP)": {"kind": "rank_average"}}
    for k in (25, 50, 100):
        for a in (0.5, 0.7, 0.85):
            out[f"(a) top-{k} re-rank α={a}"] = {"kind": "topk", "k": k, "alpha": a}
    for t1 in (0.01, 0.03, 0.1):
        for t2 in (50.0, 150.0, 500.0):
            out[f"(b) product T1={t1} T2={t2:g}"] = {"kind": "product", "t1": t1, "t2": t2}
    for t1 in (0.01, 0.03, 0.1):
        for w in (0.5, 1.0, 2.0, 3.0):
            out[f"(b') scale-free product T1={t1} w={w}"] = {"kind": "mixed", "t1": t1, "w": w}
    return out


def pick(names, mean_of) -> str:
    """Best NP C2; among the runs within its noise, the best C1."""
    top = max(mean_of(n, "NP C2") for n in names)
    close = [n for n in names if top - mean_of(n, "NP C2") <= NOISE["NP C2"]]
    return max(close, key=lambda n: (mean_of(n, "C1"), mean_of(n, "NP C2")))


def evaluate(folder: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    vals = {s: load(folder, s) for s in args.val}
    train = pd.concat([load(folder, s)[0] for s in args.train], ignore_index=True)
    # Every NP example is a validation molecule in every split, so the r4 rows hold the v4 answers too:
    # each validation split is scored by models fitted without its answers. The full fit goes to the notebook.
    held = {s: train[~train["answer"].isin(set(val["inchikey14"]))] for s, (_, val) in vals.items()}
    specs = grid()
    fitted: dict[str, dict] = {}  # run name -> {split: Logistic} for the learned combiners
    rows = []

    def run(name: str, score_fn) -> None:
        for split, (df, val) in vals.items():
            rows.append({"run": name, "split": split, **class_mrr(df, score_fn(df, split), val)})

    def add_logistic(name: str, cols) -> None:
        fitted[name] = {s: Logistic(list(cols)).fit(tr) for s, tr in held.items()}
        specs[name] = {"kind": "logistic", "model": Logistic(list(cols)).fit(train).to_dict()}
        run(name, lambda df, s: fitted[name][s].score(df))

    def score(name: str, df: pd.DataFrame, split: str) -> np.ndarray:
        if name in fitted:
            return fitted[name][split].score(df)
        return combine_rows(df, specs[name])

    for name in list(specs):
        run(name, lambda df, s, name=name: score(name, df, s))
    for col in ("fingerprint", "fragment"):
        run(f"{col} only", lambda df, s, col=col: df[col].to_numpy())
    add_logistic("(c) logistic", LOGIT_BASE)

    def mean_of(name: str, col: str) -> float:
        return float(np.mean([r[col] for r in rows if r["run"] == name]))

    cands = [n for n in specs if n.startswith(("(a)", "(b)", "(c)"))]
    best = pick(cands, mean_of)
    best_spec = dict(specs[best])
    frag = []
    if best_spec["kind"] == "logistic":
        add_logistic(f"{best} + fragment", [*LOGIT_BASE, *LOGIT_FRAGMENT])
        frag.append(f"{best} + fragment")
    else:
        extras = {"topk": [(f"{g} fragment", g) for g in (0.1, 0.2, 0.3)],
                  "mixed": [(f"{g} fragment", g) for g in (0.25, 0.5, 1.0)],
                  "product": [(f"fragment T3={t3}", t3) for t3 in (0.05, 0.2)]}[best_spec["kind"]]
        for label, w in extras:  # top-K / mixed: + w·z(fragment); product: × softmax(fragment / T3)
            name = f"{best} + {label}"
            specs[name] = {**best_spec, "extra": {"fragment": w}}
            run(name, lambda df, s, name=name: score(name, df, s))
            frag.append(name)
    best_frag = pick([best, *frag], mean_of)
    for base in dict.fromkeys([best, best_frag]):
        for tau in TAUS:
            name = f"{base} + rank-1 insert τ={tau}"
            specs[name] = {**specs[base], "insert_tau": tau}
            run(name, lambda df, s, base=base, tau=tau: insert_top1(df, score(base, df, s), tau))

    for name, feats in (("ranker v3 (HGB, no popularity / source / library)", HGB_FEATURES),
                        ("reference: ranker v3 + library features", HGB_LIBRARY)):
        models = {s: fit_bags(tr, feats, bags=args.bags, params=PARAMS) for s, tr in held.items()}
        run(name, lambda df, s, models=models, feats=feats: predict(models[s], df, feats))
        if feats is HGB_FEATURES:
            for tau in TAUS:
                run(f"ranker v3 + rank-1 insert τ={tau}",
                    lambda df, s, models=models, tau=tau: insert_top1(df, predict(models[s], df, HGB_FEATURES), tau))

    res = pd.DataFrame(rows)
    choice = choose(res, specs, best_frag, args)
    return res, choice


def choose(res: pd.DataFrame, specs: dict, best_frag: str, args: argparse.Namespace) -> dict:
    m = res.groupby("run")[["C1", "C2", "NP C2", "total"]].mean()
    a = m.loc["analog only"]
    ok = []
    for tau in TAUS:
        name = f"{best_frag} + rank-1 insert τ={tau}"
        if a["C1"] - m.loc[name, "C1"] <= args.c1_loss and m.loc[best_frag, "C2"] - m.loc[name, "C2"] <= NOISE["C2"]:
            ok.append(name)
    tau_pick = max(ok, key=lambda n: m.loc[n, "NP C2"]) if ok else None
    combiner = tau_pick or best_frag
    hgb_np = m.loc["ranker v3 (HGB, no popularity / source / library)", "NP C2"]
    use_hgb = hgb_np - m.loc["(c) logistic", "NP C2"] >= 0.015
    return {"combiner": combiner, "spec": specs[combiner], "tau_ok": ok, "hgb_np_c2": round(float(hgb_np), 4),
            "use_hgb": bool(use_hgb), "np_c2": round(float(m.loc[combiner, "NP C2"]), 4),
            "c1": round(float(m.loc[combiner, "C1"]), 4), "c2": round(float(m.loc[combiner, "C2"]), 4)}


def table(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["C1", "C2", "NP C2", "total"]
    agg = res.groupby("run", sort=False)[cols].agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        out[c] = agg[(c, "mean")].map("{:.3f}".format) + " ± " + agg[(c, "std")].map("{:.3f}".format)
    for base, tag in (("rank average (analog + FP)", "rank avg"), ("analog only", "analog")):
        for c in ("C1", "C2", "NP C2"):
            d = agg[(c, "mean")] - agg.loc[base, (c, "mean")]
            out[f"Δ {c} vs {tag}"] = [f"{x:+.3f}" + ("" if abs(x) > NOISE[c] else " (n.m.)") for x in d]
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["e5=outputs/ranker_features_e5", "e6=outputs/ranker_features_e6"],
                    help="name=feature folder (relative to the repo)")
    ap.add_argument("--c1-loss", type=float, default=0.01, help="largest C1 loss vs analog only for the rank-1 τ")
    ap.add_argument("--bags", type=int, default=4)
    ap.add_argument("--val-prefix", default="v4_seed")
    ap.add_argument("--train-prefix", default="r4_seed")
    args = ap.parse_args()
    args.val = [f"{args.val_prefix}{s}" for s in range(3)]
    args.train = [f"{args.train_prefix}{s}" for s in range(10, 15)]
    tag = args.val_prefix.removesuffix("_seed")
    t0 = time.time()
    sections, choices = [], {}
    for item in args.models:
        name, folder = item.split("=", 1)
        res, choice = evaluate(REPO_ROOT / folder, args)
        res.assign(model=name).to_parquet(WORK_DIR / f"combiners_{tag}_{name}.parquet", index=False)
        choices[name] = choice
        sections += [f"## FP v2 {name} ({folder})", md_table(table(res)),
                     f"- choice: **{choice['combiner']}** (C1 {choice['c1']:.3f}, C2 {choice['c2']:.3f}, NP C2 "
                     f"{choice['np_c2']:.3f}); τ meeting C1 loss ≤ {args.c1_loss} and no measurable C2 loss: "
                     f"{choice['tau_ok'] or 'none'}; ranker v3 NP C2 {choice['hgb_np_c2']:.3f} → "
                     f"{'ranker v3' if choice['use_hgb'] else 'logistic / combiner kept'}"]
        print(sections[-2], sections[-1], sep="\n\n", flush=True)
    (WORK_DIR / f"combiners_{tag}.json").write_text(json.dumps(choices, indent=2) + "\n")
    report = "\n\n".join([
        f"# Combiners on split {tag} ({', '.join(args.val)}; training rows {', '.join(args.train)})",
        "- MRR@25 mean ± std over seeds; n.m. = not measurable (C1 ±0.04, C2 / NP C2 ±0.02); z-scores within the "
        "window; (b) ranks by analog / T1 + fp / T2",
        *sections, f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP_combiners_{tag}.md").write_text(report)


if __name__ == "__main__":
    main()
