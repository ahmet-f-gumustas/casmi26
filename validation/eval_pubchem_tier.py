"""PubChem tier on split v2: cost on ordinary validation, gain on simulated Class 2b.

Pool: the PubChem pool (base pool ∪ filtered PubChem rows). Per seed, analog and fingerprint
scores over the union window. The main list ranks the base-pool candidates with the rank average
over the base window — the submission without the tier. PubChem-only candidates score
union-window rank average × popularity prior (``casmi.tier``).

- ordinary: every validation answer is a base-pool structure, so this measures the tier's cost.
- C2b: Class 2 molecules whose answer is among the filtered PubChem rows are made PubChem-only:
  the answer leaves the main list and joins the PubChem candidates. Without the tier they score 0.
Grid: N ∈ {3, 5, 8}, gate_k ∈ {1, 3, 5, 10, 25 − N}, beta ∈ {0, 0.5, 1}. Expected Class 2 MRR at a
test C2b share b is (1 − b)·ordinary + b·C2b; the break-even share is cost / (cost + C2b MRR).
Bias: simulated C2b answers are training structures, more often well documented in PubChem than a
typical PubChem-only isomer, so the popularity prior looks better here than it may on the test.

    python validation/eval_pubchem_tier.py --model outputs/fp_models/rank_full [--prefix v2_seed] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.channels.analog import AnalogIndex, library_row_keys
from casmi.channels.fingerprint import fingerprint_channel
from casmi.config import EXTERNAL_DIR, REPO_ROOT, TOP_K, WORK_DIR
from casmi.fpmodel import CasmiBundle
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val
from casmi.pipeline import PipelineConfig, analog_channel, candidate_windows, clean_queries, fingerprint_logits
from casmi.ranker import rank_molecule
from casmi.report import md_table
from casmi.tier import TierConfig, prior, tier_merge, union_scores

PUBCHEM_POOL = EXTERNAL_DIR / "pool_pubchem"
WEIGHTS = {"analog": 1.0, "fingerprint": 1.0}
SHARES = (0.02, 0.05, 0.10, 0.20)


def grid() -> list[TierConfig]:
    out = []
    for n, beta in itertools.product((3, 5, 8), (0.0, 0.5, 1.0)):
        for gate_k in sorted({1, 3, 5, 10, TOP_K - n}):
            out.append(TierConfig(n_slots=n, gate_k=gate_k, beta=beta))
    return out


def rr_of(ranked: list[str], answer: str) -> float:
    return 1.0 / (ranked.index(answer) + 1) if answer in ranked[:TOP_K] else 0.0


def molecule_cases(split: str, model: CasmiBundle, index: AnalogIndex, row_key: dict, sid: dict,
                   workers: int, weights: dict[str, float]) -> list[dict]:
    """Per molecule: the ingredients of the ordinary and (Class 2, filtered-in) C2b scenarios."""
    data = load_val(WORK_DIR / "splits" / split, pool_dir=PUBCHEM_POOL)
    meta = data.pool.meta
    base = set(meta.loc[meta["in_train"] | meta["in_coconut"], "key"])
    filtered = set(meta.loc[meta["in_pubchem"], "key"])
    queries = clean_queries(data.queries).reset_index(drop=True)
    windows = candidate_windows(queries, data.pool, PipelineConfig())
    allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    smiles_of = dict(zip(meta["key"], meta["smiles"]))
    per_all = {"analog": analog_channel(queries, windows, data.pool, index, row_key, allowed=allowed, workers=workers),
               "fingerprint": fingerprint_channel(fingerprint_logits(queries, windows, model), windows, smiles_of,
                                                  model.layout, workers=workers)}
    cases = []
    for m, answer, cls, src in zip(data.val["molecule_id"], data.val["inchikey14"], data.val["class"],
                                   data.val["source"]):
        keys = windows.get(m, [])
        per = {c: per_all[c].get(m, {}) for c in per_all}
        common = union_scores(keys, per, weights)
        main_keys = [k for k in keys if k in base]
        pub_keys = [k for k in keys if k not in base]
        case = {"split": split, "molecule_id": m, "class": cls, "source": src, "answer": answer,
                "main": rank_molecule(main_keys, per, {"mode": "weighted", "weights": weights}, k=TOP_K),
                "pub": pub_keys, "common": common,
                "c2b": None}
        if cls == 2 and answer in filtered and answer in main_keys:
            case["c2b"] = rank_molecule([k for k in main_keys if k != answer], per,
                                        {"mode": "weighted", "weights": weights}, k=TOP_K)
        cases.append(case)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prefix", default="v2_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--weights", default="analog=1,fingerprint=1", help="channel weights of the main list")
    ap.add_argument("--device", default=None, help="fingerprint model device (default: cuda if available)")
    args = ap.parse_args()
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
    tag = "_".join(k for k, v in weights.items() if v)
    t0 = time.time()
    model = CasmiBundle.load(sorted(args.model.glob("fp_casmi*.pt")), device=args.device)
    index = AnalogIndex.load(ANALOG_INDEX)
    row_key = library_row_keys(CLEAN_LIBRARY).to_dict()
    pop = pd.read_parquet(PUBCHEM_POOL / "popularity.parquet")
    sid = dict(zip(pop["key"], pop["pc_sid"]))
    cases = []
    for seed in args.seeds:
        cases += molecule_cases(f"{args.prefix}{seed}", model, index, row_key, sid, args.workers, weights)
        print(f"{args.prefix}{seed} done ({time.time() - t0:.0f}s)", flush=True)

    base_rr = np.array([rr_of(c["main"], c["answer"]) for c in cases])
    cls = np.array([c["class"] for c in cases])
    is_np = np.array([c["source"] == "enveda-np-examples" for c in cases])
    c2b_idx = [i for i, c in enumerate(cases) if c["c2b"] is not None]
    rows = []
    for cfg in grid():
        ordinary, c2b, used = [], [], []
        for c in cases:
            pub = {k: c["common"][k] * prior(sid.get(k, 0), cfg) for k in c["pub"]}
            final = tier_merge(c["main"], pub, c["common"], cfg)
            used.append(final != c["main"][:TOP_K])
            ordinary.append(rr_of(final, c["answer"]))
        for i in c2b_idx:
            c = cases[i]
            pub = {k: c["common"][k] * prior(sid.get(k, 0), cfg) for k in c["pub"] + [c["answer"]]}
            c2b.append(rr_of(tier_merge(c["c2b"], pub, c["common"], cfg), c["answer"]))
        ordinary, c2b = np.array(ordinary), np.array(c2b)
        c2_ord, c2_base = ordinary[cls == 2].mean(), base_rr[cls == 2].mean()
        m_b = c2b.mean() if len(c2b) else 0.0
        cost = c2_base - c2_ord
        row = {"N": cfg.n_slots, "gate_k": cfg.gate_k, "beta": cfg.beta,
               "Δ total (ordinary)": ordinary.mean() - base_rr.mean(), "Δ C2 (ordinary)": -cost,
               "Δ C2 NP (ordinary)": ordinary[(cls == 2) & is_np].mean() - base_rr[(cls == 2) & is_np].mean(),
               "C2b MRR": m_b,
               "C2b MRR NP": c2b[[cases[i]["source"] == "enveda-np-examples" for i in c2b_idx]].mean()
               if len(c2b) else np.nan,
               "tier used (ordinary)": float(np.mean(used))}
        for b in SHARES:
            row[f"ΔC2 @ b={b:.0%}"] = (1 - b) * c2_ord + b * m_b - (1 - b) * c2_base
        row["break-even b"] = cost / (cost + m_b) if cost > 0 and (cost + m_b) > 0 else 0.0
        rows.append(row)
    res = pd.DataFrame(rows)
    ok = res[res["Δ total (ordinary)"].abs() <= 0.005]
    best = (ok if len(ok) else res).sort_values("ΔC2 @ b=5%", ascending=False).iloc[0]
    choice = {"n_slots": int(best["N"]), "gate_k": int(best["gate_k"]), "beta": float(best["beta"]),
              "sid_cap": TierConfig().sid_cap}
    name = "tier_v1.json" if weights == WEIGHTS else f"tier_v1_{tag}.json"
    (WORK_DIR / name).write_text(json.dumps({"tier": choice, "weights": weights}, indent=2) + "\n")
    report = "\n\n".join([
        f"# PubChem tier ({args.prefix} {args.seeds}; main list weights {weights})",
        f"- {len(cases)} molecule-seeds; C2b cases (C2 answers among the filtered PubChem rows): {len(c2b_idx)} "
        f"of {int((cls == 2).sum())} C2 (NP {int(sum(cases[i]['source'] == 'enveda-np-examples' for i in c2b_idx))})",
        f"- no tier: total {base_rr.mean():.3f}, C2 {base_rr[cls == 2].mean():.3f}, "
        f"C2 NP {base_rr[(cls == 2) & is_np].mean():.3f}; C2b MRR without the tier = 0",
        "### Grid (sorted by expected ΔC2 at a 5 % C2b share)",
        md_table(res.sort_values("ΔC2 @ b=5%", ascending=False).head(20)),
        f"- choice (|Δ total ordinary| ≤ 0.005, best ΔC2 @ 5 %): **{choice}** → outputs/{name}",
        "- caveat: simulated C2b answers are training structures; the popularity prior favours them more than it "
        "would a typical PubChem-only test answer",
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP_pubchem_tier_{args.prefix}_{tag}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
