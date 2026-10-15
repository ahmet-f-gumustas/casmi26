"""Recall probe 2b: alternative-adduct candidates in gated slots (analog only).

Main list: the analog channel over the labelled adduct's window. Alternative-adduct candidates
(``casmi.adducts``: positive [M+H]+ / [M+Na]+ / [M+NH4]+ / [M+H-H2O]+, negative [M-H]- / [M+HCOO]- /
[M+Cl]-, windows minus the main one) are scored by the same analog hits and enter up to
``n_slots`` = 5 slots when they beat the main list's ``gate_k``-th score (``casmi.tier.slot_merge``;
``tail`` = last slots, ``merit`` = merged by score).

- Placeholder test (slot loss, as on Kaggle): its molecules are enveda-180 training spectra with
  correct labels, so any change is a loss. Answers come from the training rows they match;
  library: C2-like (every spectrum of the answer removed) and C1-like (its timsTOF spectra removed).
- Validation (v2 seeds): the same loss per class (H0, labels correct), and a mislabel simulation (H1):
  each single-adduct Class 1/2 molecule is relabelled with another adduct of its polarity, so its
  main window misses the answer; ``recovered`` is the MRR the slots then win back.
Expected Δ total at a mislabelled share b of all test molecules ≈ (1 − b)·Δ(H0) + b·recovered.

    python validation/eval_alt_adducts.py [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from casmi.adducts import ALT_ADDUCTS, alt_windows, same_ion
from casmi.candidates import CandidatePool
from casmi.channels.analog import AnalogIndex, library_row_keys
from casmi.config import DATA_DIR, REPO_ROOT, TOP_K, WORK_DIR
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val
from casmi.pipeline import QUERY_COLUMNS, PipelineConfig, analog_channel, analog_hits, candidate_windows, clean_queries
from casmi.ranker import rank_molecule
from casmi.report import md_table
from casmi.tier import slot_merge

TIMS = ("enveda-180", "enveda-np-examples")
ANALOG = {"mode": "weighted", "weights": {"analog": 1.0}}
SHARES = (0.02, 0.05, 0.10, 0.20)


def grid() -> list[dict]:
    return [{"n_slots": 5, "gate_k": g, "placement": p}
            for p, g in itertools.product(("tail", "merit"), (1, 2, 3, 5, 10, 25))]


def rr_of(ranked: list[str], answer: str) -> float:
    return 1.0 / (ranked.index(answer) + 1) if answer in ranked[:TOP_K] else 0.0


class Scenario:
    """Main and alternative-adduct analog scores for a set of molecules."""

    def __init__(self, queries: pd.DataFrame, pool: CandidatePool, index: AnalogIndex, row_key: dict,
                 allowed: np.ndarray, hits: list, workers: int) -> None:
        cfg = PipelineConfig()
        self.windows = candidate_windows(queries, pool, cfg)
        self.alt_win = alt_windows(queries, pool, self.windows, cfg.ppm, cfg.offset_ppm)
        self.main = analog_channel(queries, self.windows, pool, index, row_key, allowed=allowed, hits=hits)
        self.alt = analog_channel(queries, self.alt_win, pool, index, row_key, allowed=allowed, hits=hits)

    def ranked(self, mol_id: str, cfg: dict | None) -> list[str]:
        per = {"analog": self.main.get(mol_id, {})}
        main = rank_molecule(self.windows.get(mol_id, []), per, ANALOG)
        return main if cfg is None else slot_merge(main, per["analog"], self.alt.get(mol_id, {}), **cfg)


def evaluate(sc: Scenario, answers: dict[str, str], cfgs: list[dict]) -> pd.DataFrame:
    """Per molecule: reciprocal rank without the slots and under each config, and whether the slots fired."""
    rows = []
    for m, a in answers.items():
        base = sc.ranked(m, None)
        row = {"molecule_id": m, "base": rr_of(base, a), "has_alt": bool(sc.alt_win.get(m))}
        for i, cfg in enumerate(cfgs):
            r = sc.ranked(m, cfg)
            row[f"rr{i}"], row[f"used{i}"] = rr_of(r, a), r != base[: len(r)]
        rows.append(row)
    return pd.DataFrame(rows)


def mislabel(queries: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, set[str]]:
    """Queries of single-adduct molecules relabelled with another adduct of their polarity (H1)."""
    out, done = queries.copy(), set()
    for m, g in queries.groupby("molecule_id"):
        labels = g["adduct"].unique()
        if any(not same_ion(labels[0], a) for a in labels[1:]):
            continue
        options = [a for a in ALT_ADDUCTS.get(str(g["ionization_mode"].iloc[0]), ()) if not same_ion(a, labels[0])]
        if options:
            out.loc[g.index, "adduct"] = options[int(rng.integers(len(options)))]
            done.add(m)
    return out, done


def summarise(df: pd.DataFrame, cfgs: list[dict], name: str, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(cfgs):
        row = {"scenario": name, **cfg, "fires": df[f"used{i}"].mean()}
        for g, mask in groups.items():
            row[f"Δ {g}"] = df.loc[mask, f"rr{i}"].mean() - df.loc[mask, "base"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def placeholder(index: AnalogIndex, row_key: dict, pool: CandidatePool, cfgs: list[dict], workers: int) -> list:
    test = pd.read_parquet(DATA_DIR / "test.parquet", columns=QUERY_COLUMNS + ["base_peak_intensity"])
    lib = pd.read_parquet(CLEAN_LIBRARY, columns=["row", "key", "ingest_lib", "precursor_mz", "adduct",
                                                  "base_peak_intensity"])
    j = test.merge(lib[lib["ingest_lib"] == "enveda-180"], on=["precursor_mz", "adduct", "base_peak_intensity"])
    answers = j.groupby("molecule_id")["key"].agg(lambda s: s.mode().iloc[0]).to_dict()
    queries = clean_queries(test.drop(columns=["base_peak_intensity"])).reset_index(drop=True)
    out = []
    for name, drop in (("C2-like", lib["key"].isin(set(answers.values()))),
                       ("C1-like", lib["key"].isin(set(answers.values())) & lib["ingest_lib"].isin(TIMS))):
        allowed = np.zeros(int(lib["row"].max()) + 1, dtype=bool)
        allowed[lib.loc[~drop, "row"].to_numpy()] = True
        hits = analog_hits(queries, index, allowed=allowed, workers=workers)
        sc = Scenario(queries, pool, index, row_key, allowed, hits, workers)
        df = evaluate(sc, answers, cfgs)
        out.append((name, df, summarise(df, cfgs, f"placeholder {name}", {"MRR": np.ones(len(df), bool)}),
                    sum(bool(v) for v in sc.alt_win.values()) / max(1, len(sc.windows))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="v2_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    cfgs = grid()
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    ph = placeholder(index, row_key, CandidatePool.load(), cfgs, args.workers)
    print(f"placeholder done ({time.time() - t0:.0f}s)", flush=True)

    h0, h1 = [], []
    for seed in args.seeds:
        split = f"{args.prefix}{seed}"
        data = load_val(WORK_DIR / "splits" / split)
        queries = clean_queries(data.queries).reset_index(drop=True)
        allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
        allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
        hits = analog_hits(queries, index, allowed=allowed, workers=args.workers)
        answers = dict(zip(data.val["molecule_id"], data.val["inchikey14"]))
        info = data.val.set_index("molecule_id")
        df = evaluate(Scenario(queries, data.pool, index, row_key, allowed, hits, args.workers), answers, cfgs)
        h0.append(df.join(info[["class", "source"]], on="molecule_id").assign(split=split))
        wrong, relabelled = mislabel(queries, np.random.default_rng(seed))
        keep = [m for m in relabelled if info.loc[m, "class"] in (1, 2)]
        df1 = evaluate(Scenario(wrong, data.pool, index, row_key, allowed, hits, args.workers),
                       {m: answers[m] for m in keep}, cfgs)
        h1.append(df1.join(info[["class", "source"]], on="molecule_id").assign(split=split))
        print(f"{split} done ({time.time() - t0:.0f}s)", flush=True)
    h0, h1 = pd.concat(h0, ignore_index=True), pd.concat(h1, ignore_index=True)
    h0.to_parquet(WORK_DIR / "alt_adducts_h0.parquet", index=False)
    h1.to_parquet(WORK_DIR / "alt_adducts_h1.parquet", index=False)
    is_np = (h0["source"] == "enveda-np-examples").to_numpy()
    cls = h0["class"].to_numpy()
    res = summarise(h0, cfgs, "val H0", {"C1": cls == 1, "C2": cls == 2, "C2 NP": (cls == 2) & is_np,
                                          "total": np.ones(len(h0), bool)})
    rec = summarise(h1, cfgs, "val H1", {"recovered": np.ones(len(h1), bool),
                                          "recovered NP": (h1["source"] == "enveda-np-examples").to_numpy()})
    res["recovered (H1)"] = rec["Δ recovered"].to_numpy()
    res["recovered NP (H1)"] = rec["Δ recovered NP"].to_numpy()
    res["fires (H1)"] = rec["fires"].to_numpy()
    for name, _, s, _ in ph:
        res[f"slot loss placeholder {name}"] = s["Δ MRR"].to_numpy()
    for b in SHARES:
        res[f"Δ total @ b={b:.0%}"] = (1 - b) * res["Δ total"] + b * res["recovered (H1)"]
    ok = res[(res["Δ total"].abs() <= 0.005) & (res["slot loss placeholder C2-like"].abs() <= 0.005)
             & (res["slot loss placeholder C1-like"].abs() <= 0.005)]
    best = (ok if len(ok) else res).sort_values("recovered (H1)", ascending=False).iloc[0]
    choice = {"n_slots": int(best["n_slots"]), "gate_k": int(best["gate_k"]), "placement": str(best["placement"])}
    (WORK_DIR / "alt_adducts_v1.json").write_text(json.dumps({"alt_adducts": choice}, indent=2) + "\n")
    base_mis = h1["base"].mean()
    report = "\n\n".join([
        f"# Recall probe 2b: alternative adducts ({args.prefix} {args.seeds} + placeholder test)",
        f"- val: {len(h0)} molecule-seeds; molecules with alternative candidates (single adduct type): "
        f"{h0['has_alt'].mean():.0%}. Placeholder: "
        + ", ".join(f"{n} {len(d)} molecules, alternative windows for {share:.0%}" for n, d, _, share in ph),
        f"- H1 simulation: {len(h1)} relabelled Class 1/2 molecule-seeds; answer still ranked without the slots: "
        f"MRR {base_mis:.3f} (the windows overlap rarely)",
        "### Grid (Δ vs the analog-only list; slot loss = Δ MRR on the placeholder test)",
        md_table(res.drop(columns=["scenario"]).sort_values("recovered (H1)", ascending=False)),
        f"- choice (|Δ| ≤ 0.005 on val total and both placeholder scenarios, best H1 recovery): **{choice}** "
        "→ outputs/alt_adducts_v1.json",
        "- Expected LB change if a share b of all test molecules carried a wrong adduct label: Δ total @ b. "
        "The train data says such labels are rare in Enveda's own libraries (notes/ADDUCTS.md: enveda-180 0 / "
        "1.15M rows, enveda-np-examples 1 / 1184 rows with |ppm| > 20 for the labelled adduct).",
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / "EXP_alt_adducts.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
