"""Per-spectrum rank31 scores combined by ranker confidence (plan item 7a), on the v3 validation splits (reference).

Every query spectrum becomes its own query (``<molecule>#<i>``) over its molecule's candidate window, and gets the
31 rank31 features from that spectrum alone (analogs, library match, fingerprint logits, MetFrag-lite). The
rank31 ranker (our r3 rows, refitted per split without its answers, as in ``eval_rank31.py``) scores them; the
per-spectrum probabilities are combined per molecule with weights from each spectrum's confidence:

- ``mean``: uniform; ``max``: best spectrum per candidate;
- ``conf``: weight = the spectrum's top probability; ``margin``: top minus second probability;
- ``soft-T``: weight = exp(top probability / T).

Each combination alone and blended with the molecule-level probability (α · molecule + (1 − α) · combined), next
to the oracle that picks each molecule's best spectrum (an upper bound, not a method). Frames are cached in
``outputs/rank31_features/<split>_spectra.parquet``. Report: ``notes/EXP_spectrum_weight.md``.

    python validation/eval_spectrum_weight.py --model outputs/fp_models/np_v2_e5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi import rank31
from casmi.channels.analog import AnalogConfig, AnalogIndex, library_row_keys
from casmi.config import REPO_ROOT, TOP_K, WORK_DIR
from casmi.fpmodel import CasmiBundle
from casmi.harness import ANALOG_INDEX, CLEAN_LIBRARY, load_val
from casmi.pipeline import PipelineConfig, analog_hits, candidate_windows, clean_queries, fingerprint_logits, \
    library_channel
from casmi.report import md_table
from casmi.split import NP_EXAMPLES
from eval_forward import NOISE, our_rows
from eval_rank31 import FEATS, class_mrr

ALPHAS = (0.0, 0.25, 0.5, 0.75)
TEMPS = (0.1, 0.3)


def spectrum_frame(split: str, model: CasmiBundle, index: AnalogIndex, row_key: dict, caches: dict,
                   workers: int) -> pd.DataFrame:
    data = load_val(WORK_DIR / "splits" / split)
    queries = clean_queries(data.queries).reset_index(drop=True)
    cfg = PipelineConfig(workers=workers)
    windows = candidate_windows(queries, data.pool, cfg)
    q = queries.assign(parent=queries["molecule_id"])
    q["molecule_id"] = q["parent"] + "#" + q.groupby("parent").cumcount().astype(str)
    win = {s: windows.get(p, []) for s, p in zip(q["molecule_id"], q["parent"])}
    allowed = np.zeros(int(data.in_library.index.max()) + 1, dtype=bool)
    allowed[data.in_library.index.to_numpy()] = data.in_library.to_numpy()
    smiles_of = dict(zip(data.pool.meta["key"], data.pool.meta["smiles"]))
    hits = analog_hits(q, index, cfg=AnalogConfig(top_k=rank31.ANALOG_TOP_K), allowed=allowed, workers=workers)
    _, lib_max = library_channel(q, win, CLEAN_LIBRARY, cfg, data.in_library, both=True)
    logits = fingerprint_logits(q, win, model)
    df = rank31.frame(q, win, smiles_of, hits, row_key, lib_max, logits, model.layout, model.bits,
                      workers=workers, caches=caches)
    df["spectrum"] = df.pop("molecule_id")
    df["molecule_id"] = df["spectrum"].str.rsplit("#", n=1).str[0]
    val = data.val.set_index("molecule_id")
    df["label"] = df["key"].to_numpy() == df["molecule_id"].map(val["inchikey14"]).to_numpy()
    return df


def combine(sp: pd.DataFrame, how: str) -> pd.Series:
    """Per (molecule_id, key) combination of the per-spectrum probabilities ``p``."""
    g = sp.groupby("spectrum")["p"]
    top = g.transform("max")
    if how == "mean":
        w = pd.Series(1.0, index=sp.index)
    elif how == "max":
        return sp.groupby(["molecule_id", "key"])["p"].max()
    elif how == "conf":
        w = top
    elif how == "margin":
        second = g.transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else 0.0)
        w = top - second + 1e-6
    else:
        w = np.exp(top / float(how.split("-")[1]))
    d = sp.assign(wp=w * sp["p"], w=w)
    s = d.groupby(["molecule_id", "key"])[["wp", "w"]].sum()
    return s["wp"] / s["w"]


def oracle(sp: pd.DataFrame, val: pd.DataFrame) -> dict:
    """MRR@25 per class when each molecule uses its best single spectrum."""
    d = sp.sort_values(["spectrum", "p", "key"], ascending=[True, False, True])
    d["pos"] = d.groupby("spectrum").cumcount() + 1
    rr = d[d["label"] & (d["pos"] <= TOP_K)].assign(rr=lambda x: 1.0 / x["pos"]).groupby("molecule_id")["rr"].max()
    rr = val["molecule_id"].map(rr).fillna(0.0)
    c2 = val["class"] == 2
    return {"C1": rr[val["class"] == 1].mean(), "C2": rr[c2].mean(),
            "NP C2": rr[c2 & (val["source"] == NP_EXAMPLES)].mean()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ours-frames", nargs="+", default=[f"r3_seed{s}" for s in range(10, 15)])
    ap.add_argument("--val", nargs="+", default=["v3_seed0", "v3_seed1", "v3_seed2"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    ours = our_rows(args.ours_frames)
    model = index = row_key = None
    caches: dict = {}
    res = []
    for split in args.val:
        path = FEATS / f"{split}_spectra.parquet"
        if path.exists():
            sp = pd.read_parquet(path)
        else:
            if model is None:
                model = CasmiBundle.load(sorted(args.model.glob("fp_casmi*.pt")))
                index, row_key = AnalogIndex.load(ANALOG_INDEX), library_row_keys(CLEAN_LIBRARY).to_dict()
            sp = spectrum_frame(split, model, index, row_key, caches, args.workers)
            sp.to_parquet(path, index=False)
        df = pd.read_parquet(FEATS / f"{split}.parquet")
        val = pd.read_parquet(WORK_DIR / "splits" / split / "val_ids.parquet")
        tr = ours[~ours["answer"].isin(set(val["inchikey14"]))]
        models = rank31.fit(tr[rank31.USED].to_numpy(np.float32), tr["label"].to_numpy(float),
                            (tr["class"] == 2).to_numpy(float))
        p_mol = rank31.proba(models, df[rank31.USED].to_numpy(np.float32))
        sp["p"] = rank31.proba(models, sp[rank31.USED].to_numpy(np.float32))
        mol = pd.Series(p_mol, index=pd.MultiIndex.from_arrays([df["molecule_id"], df["key"]]))
        runs = {"molecule-level ranker": p_mol}
        for how in ["mean", "max", "conf", "margin", *[f"soft-{t}" for t in TEMPS]]:
            comb = combine(sp, how).reindex(mol.index).fillna(0.0).to_numpy()
            for a in ALPHAS:
                runs[f"{how}, α={a}"] = a * p_mol + (1 - a) * comb
        for run, s in runs.items():
            res.append({"run": run, "split": split, **class_mrr(df, s, val)})
        res.append({"run": "oracle: best single spectrum", "split": split, **oracle(sp, val)})
        n_spec = sp.groupby("molecule_id")["spectrum"].nunique()
        print(f"{split}: {sp['spectrum'].nunique():,} spectra over {len(n_spec)} molecules "
              f"({(n_spec > 1).mean():.0%} with several), {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(res)
    agg = res.groupby("run", sort=False)[["C1", "C2", "NP C2"]].mean()
    tab = agg.reset_index()
    for c in ("C1", "C2", "NP C2"):
        d = tab[c].to_numpy() - agg.loc["molecule-level ranker", c]
        tab[f"Δ {c}"] = [f"{x:+.3f}" + ("" if abs(x) > NOISE else " (n.m.)") for x in d]
    report = "\n\n".join([
        "# Per-spectrum rank31 scores combined by ranker confidence (v3, reference)",
        f"- ranker: rank31 on our rows {', '.join(args.ours_frames)}, refitted per split without its answers; the "
        f"same models score molecule-level and per-spectrum features; FP model {args.model}; n.m. = within ±{NOISE}",
        md_table(tab.round(3)), f"- runtime {time.time() - t0:.0f}s"]) + "\n"
    (REPO_ROOT / "notes" / "EXP_spectrum_weight.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
