"""PubChem pool extension: recall, candidate counts, analog Class 2 trade-off and Class 2b.

1. Membership: is a validation structure in the *full* PubChem index? The index rows within
   ±2 mDa of its mass are compared by InChIKey14 against the structure's own InChIKey14 and its
   scorer key (a row drawn as another tautomer can be missed, ~2 %); cached per key.
2. Recall and candidates per mass window, old pool vs PubChem pool (C3 answers removed).
3. Analog channel C2 MRR with each pool: more isomers per window is the cost of PubChem.
4. Class 2b: C2 molecules whose structure is in PubChem, simulated as PubChem-only. With the
   old pool their answer is unreachable; with the PubChem pool it is reachable only if the
   NP filter kept it. Ranking has no source prior. The break-even C2b share is reported:
   above it the PubChem pool wins on Class 2.

    python validation/eval_pubchem.py [--prefix v2_seed] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from casmi.candidates import POOL_DIR, CandidatePool
from casmi.config import EXTERNAL_DIR, REPO_ROOT, WORK_DIR
from casmi.harness import class_mrr, compute_channels, load_val, reciprocal_ranks
from casmi.pipeline import PipelineConfig, candidate_windows, clean_queries
from casmi.pubchem import PubChemIndex
from casmi.report import md_table

RDLogger.DisableLog("rdApp.*")
PUBCHEM_POOL = EXTERNAL_DIR / "pool_pubchem"
RAW = EXTERNAL_DIR / "pubchem" / "raw"
MEMBERSHIP = WORK_DIR / "pubchem_membership.parquet"
_IDX: dict = {}


def _ik14(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    key = Chem.MolToInchiKey(mol) if mol is not None else ""
    return key[:14] or None


def _members(item: tuple[str, float, str]) -> tuple[str, bool, int]:
    key, mass, smiles = item
    index = _IDX["index"]
    window = index.window(mass, 0.002)
    targets = {key, _ik14(smiles)} - {None}
    found = any(_ik14(index.smiles(p)) in targets for p in window)
    return key, found, len(window)


def membership(items: dict[str, tuple[float, str]], workers: int) -> pd.DataFrame:
    """``in_pubchem_full`` and window size per scorer key; ``items`` maps key → (mass, SMILES)."""
    cached = pd.read_parquet(MEMBERSHIP) if MEMBERSHIP.exists() else pd.DataFrame(columns=["key", "in_pubchem_full", "n_window"])
    done = set(cached["key"])
    todo = [(k, m, s) for k, (m, s) in items.items() if k not in done]
    if todo:
        _IDX["index"] = PubChemIndex(RAW)
        with Pool(workers) as pool:
            res = pool.map(_members, todo, chunksize=1)
        _IDX.clear()
        new = pd.DataFrame(res, columns=["key", "in_pubchem_full", "n_window"])
        cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
        cached.to_parquet(MEMBERSHIP, index=False)
    return cached.set_index("key")


def window_stats(data, pool: CandidatePool) -> pd.DataFrame:
    windows = candidate_windows(clean_queries(data.queries), pool, PipelineConfig())
    return pd.DataFrame([{"molecule_id": m, "class": c, "n_candidates": len(windows.get(m, [])),
                          "hit": key in set(windows.get(m, []))}
                         for m, c, key in zip(data.val["molecule_id"], data.val["class"], data.val["inchikey14"])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="v2_seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()

    new_meta = pd.read_parquet(PUBCHEM_POOL / "meta.parquet", columns=["key", "exact_mass", "in_pubchem", "in_train",
                                                                       "in_coconut"])
    filtered_keys = set(new_meta.loc[new_meta["in_pubchem"], "key"])
    lib_np = pd.read_parquet(WORK_DIR / "train_clean.parquet",
                             columns=["ingest_lib", "key", "exact_mass", "normalized_smiles"])
    np_proxy = lib_np[lib_np["ingest_lib"] == "enveda-np-examples"].drop_duplicates("key")

    win_rows, mrr_rows, c2b_rows = [], [], []
    all_keys: dict[str, tuple[float, str]] = {k: (m, s) for k, m, s in zip(
        np_proxy["key"], np_proxy["exact_mass"], np_proxy["normalized_smiles"])}
    for seed in args.seeds:
        split = f"{args.prefix}{seed}"
        for name, pool_dir in (("old pool", POOL_DIR), ("PubChem pool", PUBCHEM_POOL)):
            data = load_val(WORK_DIR / "splits" / split, pool_dir=pool_dir)
            all_keys.update({k: (m, s) for k, m, s in zip(data.val["inchikey14"], data.val["exact_mass"],
                                                           data.val["smiles"])})
            win_rows.append(window_stats(data, data.pool).assign(seed=seed, pool=name))
            windows, channels = compute_channels(data, channels=("analog",), workers=args.workers)
            smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
            from casmi.harness import predict

            preds = predict(windows, channels, {"analog": 1.0}, smiles_of, data.val["molecule_id"])
            mrr_rows.append({**class_mrr(preds, data.val, f"analog [{name}]"), "seed": seed, "pool": name})
            rr = reciprocal_ranks(windows, channels, {"analog": 1.0}, data.val)
            c2 = data.val[data.val["class"] == 2]
            c2b_rows.append(c2[["molecule_id", "inchikey14", "source"]].assign(
                seed=seed, pool=name, rr=c2["molecule_id"].map(rr).to_numpy()))
            print(f"{split} / {name} done ({time.time() - t0:.0f}s)", flush=True)

    member = membership(all_keys, args.workers)
    wins = pd.concat(win_rows)
    win_tab = (wins[wins["class"] != 3].groupby(["pool", "class"])
               .agg(n=("hit", "size"), recall=("hit", "mean"), median_candidates=("n_candidates", "median"),
                    p90_candidates=("n_candidates", lambda s: s.quantile(0.9)))
               .reset_index())
    mrr = pd.DataFrame(mrr_rows)
    mrr_tab = mrr.groupby("pool")[["C1", "C2", "total"]].agg(["mean", "std"])
    mrr_tab.columns = [f"{a}_{b}" for a, b in mrr_tab.columns]

    c2 = pd.concat(c2b_rows)
    c2["in_pubchem_full"] = c2["inchikey14"].map(member["in_pubchem_full"]).fillna(False).astype(bool)
    c2["kept_by_filter"] = c2["inchikey14"].isin(filtered_keys)
    # C2b: PubChem-only simulation — reachable only through the filtered PubChem rows.
    c2b = c2[c2["in_pubchem_full"] & (c2["pool"] == "PubChem pool")].copy()
    c2b["rr_c2b"] = np.where(c2b["kept_by_filter"], c2b["rr"], 0.0)
    m_old = c2.loc[c2["pool"] == "old pool", "rr"].mean()
    m_new_a = c2.loc[c2["pool"] == "PubChem pool", "rr"].mean()
    m_new_b = c2b["rr_c2b"].mean() if len(c2b) else 0.0
    breakeven = (m_old - m_new_a) / (m_old - m_new_a + m_new_b) if (m_old - m_new_a + m_new_b) > 0 else np.nan
    scen = pd.DataFrame([{"C2b share": b, "old pool C2": (1 - b) * m_old, "PubChem pool C2": (1 - b) * m_new_a + b * m_new_b}
                         for b in (0.0, 0.25, 0.5, 0.75)])
    by_src = []
    for src in sorted(c2["source"].unique()):
        o = c2.loc[(c2["pool"] == "old pool") & (c2["source"] == src), "rr"].mean()
        a = c2.loc[(c2["pool"] == "PubChem pool") & (c2["source"] == src), "rr"].mean()
        sub = c2b[c2b["source"] == src]
        b_ = sub["rr_c2b"].mean() if len(sub) else 0.0
        be = (o - a) / (o - a + b_) if (o - a + b_) > 0 else np.nan
        by_src.append({"source": src, "old pool C2": o, "PubChem pool C2 (in pool)": a, "C2b filter recall":
                       sub["kept_by_filter"].mean() if len(sub) else np.nan, "C2b MRR (PubChem pool)": b_,
                       "break-even C2b share": be})

    npm = np_proxy.assign(in_pubchem_full=np_proxy["key"].map(member["in_pubchem_full"]).fillna(False).astype(bool),
                          kept=np_proxy["key"].isin(filtered_keys))
    report = "\n\n".join([
        f"# PubChem pool extension ({args.prefix} {args.seeds})",
        f"- PubChem pool: {len(new_meta):,} structures ({int(new_meta['in_pubchem'].sum()):,} from the filtered "
        f"PubChem rows, {int((new_meta['in_pubchem'] & ~new_meta['in_train'] & ~new_meta['in_coconut']).sum()):,} new)",
        "### Recall and candidates per window (C1/C2; C3 answers removed)", md_table(win_tab),
        "### Analog channel MRR@25, mean ± std over seeds", md_table(mrr_tab.reset_index()),
        "### Class 2b (PubChem-only simulation)",
        f"- val C2 molecule-seeds whose structure is in the full PubChem index: {int(c2.loc[c2['pool'] == 'old pool', 'in_pubchem_full'].sum())}"
        f"/{int((c2['pool'] == 'old pool').sum())}; kept by the NP filter: {int(c2b['kept_by_filter'].sum())}/{len(c2b)} "
        f"(filter recall {c2b['kept_by_filter'].mean():.1%})",
        f"- NP proxy (250 enveda-np-examples, all in train by construction, so none is PubChem-only): in full "
        f"PubChem {npm['in_pubchem_full'].mean():.1%}, kept by the filter {npm['kept'].mean():.1%} "
        f"(of those in PubChem: {npm.loc[npm['in_pubchem_full'], 'kept'].mean():.1%})",
        f"- C2 MRR: old pool {m_old:.3f}; PubChem pool, in-pool C2 {m_new_a:.3f}; PubChem pool, C2b {m_new_b:.3f} "
        f"(old pool C2b = 0)",
        f"- **break-even C2b share: {breakeven:.1%}** — if more of the hidden C2 answers than this are PubChem-only, "
        "the PubChem pool scores higher on Class 2",
        md_table(scen),
        "### By source (enveda-np-examples is the test proxy)", md_table(pd.DataFrame(by_src)),
        f"- runtime {time.time() - t0:.0f}s",
    ]) + "\n"
    (REPO_ROOT / "notes" / f"EXP011_pubchem_{args.prefix}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
