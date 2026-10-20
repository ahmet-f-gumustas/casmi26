"""Build the Kaggle submission notebook with ``src/casmi`` embedded (base64 tar.gz).

The notebook installs RDKit and ms_entropy from attached wheel datasets, unpacks the
embedded package, finds its inputs anywhere under the input root (Kaggle's mount
layout has changed before), runs ``casmi.pipeline`` on ``test.parquet`` and writes
``submission.csv`` with ``casmi.submit``. Runtime and peak memory are printed.

    python scripts/make_kaggle_notebook.py                 # → kaggle/submission/
    python scripts/make_kaggle_notebook.py --local-check   # also run it locally on the placeholder test
    python scripts/make_kaggle_notebook.py --mode combined [--ranker outputs/ranker_v1.json] [--pool pubchem]
    python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/ranker_fp_ap.json --fp ap   # GPU
    python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/ranker_fp.json --fp casmi    # our model
    python scripts/make_kaggle_notebook.py --mode combined --ranker <json> --fp casmi --pool pubchem --tier outputs/tier_v1.json
    python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/ranker_v2.json --fp casmi   # ranker v2
    python scripts/make_kaggle_notebook.py --mode combined --alt-adducts outputs/alt_adducts_v1.json   # probe 2b
    python scripts/make_kaggle_notebook.py --mode combined --ranker outputs/combiner_e5.json --fp casmi   # combiner

``--local-check`` builds a fake ``/kaggle/input`` from symlinks, runs the notebook's
code cells as a script, validates the CSV and scores it against the placeholder
test's known answers (its molecules are enveda-180 training spectra).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

from casmi.config import COMPETITION, DATA_DIR, REPO_ROOT, WORK_DIR
from casmi.candidates import POOL_DIR

KERNEL_ID = "red6161/casmi26-submission"
DATASETS = ["red6161/casmi26-library", "red6161/casmi26-wheels", "metric/rdkit-2026-3-3-wheel"]
# local folders mirror what each dataset holds (the PubChem pool is checked from its upload staging, fps.npz)
POOLS = {"base": ("red6161/casmi26-pool", POOL_DIR),
         "pubchem": ("red6161/casmi26-pool-pubchem", WORK_DIR / "kaggle" / "datasets" / "casmi26-pool-pubchem")}
OUT = REPO_ROOT / "kaggle" / "submission"
# Fingerprint models, attached as published (not re-uploaded): name -> (datasets, local folders).
# "ap": prvsiyan's FPNet checkpoints (CC BY-NC-SA 4.0) and the fp_bits.npy column order; see NOTICE.
# "casmi": our SpecFormer (scripts/train_fp.py), uploaded as red6161/casmi26-fp-model; "final_same" /
# "final_big": the same model trained on all data, uploaded as red6161/casmi26-fp-final-{same,big}.
FP_MODELS = {"ap": (["prvsiyan/casmi26-fp-models-v2", "prvsiyan/coconut-casmi26-candidates"],
                    {"casmi26-fp-models-v2": REPO_ROOT / "external" / "prvsiyan" / "fp_models_v2",
                     "coconut-casmi26-candidates": REPO_ROOT / "external" / "prvsiyan" / "coconut_fp_bits"}),
             "casmi": (["red6161/casmi26-fp-model"], {"casmi26-fp-model": WORK_DIR / "fp_models" / "submit"}),
             # final models on all data (plan item E), one dataset per recipe
             "final_same": (["red6161/casmi26-fp-final-same"],
                            {"casmi26-fp-final-same": WORK_DIR / "fp_models" / "final_same"}),
             "final_big": (["red6161/casmi26-fp-final-big"],
                           {"casmi26-fp-final-big": WORK_DIR / "fp_models" / "final_big"})}

SETUP = r'''
import glob, os, subprocess, sys, time, resource
T0 = time.time()
INPUT = os.environ.get("CASMI_KAGGLE_INPUT", "/kaggle/input")
WORK = os.environ.get("CASMI_KAGGLE_WORK", "/kaggle/working")

def find(pattern):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {INPUT}")
    return hits[0]

def ensure(module, requirement):
    try:
        __import__(module)
    except ImportError:
        wheel_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(INPUT, "**", "*.whl"), recursive=True)})
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "-q", requirement]
        cmd += [f"--find-links={d}" for d in wheel_dirs]
        subprocess.run(cmd, check=True)

ensure("rdkit", "rdkit==2026.3.3")
ensure("ms_entropy", "ms_entropy==1.5.3")
import rdkit
print("python", sys.version.split()[0], "| rdkit", rdkit.__version__)
'''

UNPACK = r'''
import base64, io, tarfile
CASMI_SRC = os.path.join(WORK, "casmi_src")
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CASMI_TGZ_B64)), mode="r:gz") as tar:
    tar.extractall(CASMI_SRC, filter="data")
sys.path.insert(0, CASMI_SRC)
import casmi
print("casmi", casmi.__version__, "unpacked to", CASMI_SRC)
'''

RUN = r'''
import pathlib
import pandas as pd
from casmi.candidates import CandidatePool
from casmi.pipeline import PipelineConfig, run_combined_pipeline, run_library_pipeline
from casmi.submit import make_submission, write_submission

test = pd.read_parquet(find("test.parquet"))
sample = pd.read_csv(find("sample_submission.csv"))
pool = CandidatePool.load(pathlib.Path(os.path.dirname(find("masses.npy"))))
library = find("train_clean.parquet")
cfg = PipelineConfig()
print(f"test: {len(test)} spectra / {test['molecule_id'].nunique()} molecules; pool {len(pool):,}; "
      f"library {library}; workers {cfg.workers}; mode {CASMI_MODE}", flush=True)

if CASMI_MODE == "library":
    preds, _ = run_library_pipeline(test, pool, library, cfg)
else:
    from casmi.channels.analog import AnalogIndex
    index = AnalogIndex.load(pathlib.Path(os.path.dirname(find("index.json"))))
    row_key = pd.read_parquet(library, columns=["row", "key"]).set_index("row")["key"].to_dict()
    print(f"analog index loaded, {time.time() - T0:.0f}s; ranker {CASMI_RANKER}", flush=True)
    fp_model = None
    if CASMI_FP == "ap":
        import numpy as np
        from casmi.fpmodel import APBundle
        ckpts = sorted(glob.glob(os.path.join(INPUT, "**", "fp_*.pt"), recursive=True))
        fp_model = APBundle.load(ckpts, bits=np.load(find("fp_bits.npy"), allow_pickle=False))
        print(f"fingerprint models {[os.path.basename(c) for c in ckpts]} on {fp_model.device}, "
              f"{time.time() - T0:.0f}s", flush=True)
    elif CASMI_FP == "casmi":
        from casmi.fpmodel import CasmiBundle
        ckpts = sorted(glob.glob(os.path.join(INPUT, "**", "fp_casmi*.pt"), recursive=True))
        fp_model = CasmiBundle.load(ckpts)
        print(f"fingerprint model {[os.path.basename(c) for c in ckpts]} on {fp_model.device}, "
              f"{time.time() - T0:.0f}s", flush=True)
    ranker = dict(CASMI_RANKER)
    if ranker.get("mode") == "learned":
        ranker["train"] = pd.read_parquet(find("ranker_train.parquet"))
    if ranker.get("mode") == "rank31":
        import numpy as np
        rows = np.load(find(ranker["rows"]), allow_pickle=False)
        ranker["train"] = (rows["X"], rows["Y"], rows["M"])
        print(f"rank31 training rows {ranker['rows']} ({ranker['dataset']}): {len(rows['Y']):,} rows", flush=True)
    preds, _ = run_combined_pipeline(test, pool, library, index, row_key, ranker, cfg, fp_model=fp_model,
                                     tier=CASMI_TIER, alt_adducts=CASMI_ALT)
sub = make_submission(preds, sample["molecule_id"].tolist())
path = write_submission(sub, pathlib.Path(WORK) / "submission.csv")
ranked = sum(1 for v in preds.values() if v)
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"wrote {path}: {len(sub)} rows, {ranked} molecules with library hits")
print(f"CASMI_RUNTIME_S={time.time() - T0:.0f} CASMI_PEAK_RSS_GB={peak_gb:.2f}")
from casmi import degraded
print(degraded.report(), flush=True)
'''


def package_b64() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(REPO_ROOT / "src" / "casmi", arcname="casmi",
                filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    return base64.b64encode(buf.getvalue()).decode()


def code_cells(mode: str, ranker: dict, fp: str | None = None, tier: dict | None = None,
               alt: dict | None = None) -> list[str]:
    kind = fp if fp in (None, "ap") else "casmi"  # every model of ours is a SpecFormer checkpoint
    config = (f"CASMI_MODE = {mode!r}\nCASMI_RANKER = {ranker!r}\nCASMI_FP = {kind!r}\nCASMI_TIER = {tier!r}\n"
              f"CASMI_ALT = {alt!r}\n")
    return [SETUP.strip(), f"CASMI_TGZ_B64 = {package_b64()!r}\n" + UNPACK.strip(), config + RUN.strip()]


def build(out: Path, mode: str = "library", ranker: dict | None = None, pool: str = "base",
          fp: str | None = None, tier: dict | None = None, alt: dict | None = None) -> Path:
    ranker = ranker or {}
    if tier is not None and pool != "pubchem":
        raise ValueError("the PubChem tier needs --pool pubchem")
    title = ("library-channel" if mode == "library"
             else f"combined ({ranker}, {pool} pool, fp {fp}, tier {tier}, alternative adducts {alt})")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": f"# CASMI 2026 — {title} submission\n\n"
              "Generated by `scripts/make_kaggle_notebook.py`; the `casmi` package is embedded below."}]
    cells += [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c}
              for c in code_cells(mode, ranker, fp, tier, alt)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
    out.mkdir(parents=True, exist_ok=True)
    code_file = "casmi26-submission.ipynb"
    (out / code_file).write_text(json.dumps(nb, indent=1))
    meta = {"id": KERNEL_ID, "title": "casmi26 submission", "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": fp is not None, "enable_tpu": False,
            "enable_internet": False, "competition_sources": [COMPETITION],
            "dataset_sources": [POOLS[pool][0], *DATASETS] + (["red6161/casmi26-analog-index"] if mode != "library" else [])
            + (FP_MODELS[fp][0] if fp else []) + (["red6161/casmi26-ranker-train"] if ranker.get("mode") == "learned"
                                                   else [])
            + ([ranker["dataset"]] if ranker.get("mode") == "rank31" else []),
            "kernel_sources": [], "model_sources": [], "keywords": [],
            "docker_image_pinning_type": "original"}  # rerun on the image the version was saved with
    if fp:
        meta["machine_shape"] = "NvidiaTeslaT4"  # recent PyTorch builds no longer run on the P100
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out / code_file


def placeholder_truth() -> dict[str, str]:
    """Answers for the visible placeholder test: its rows are enveda-180 training spectra."""
    cols = ["precursor_mz", "adduct", "base_peak_intensity"]
    test = pd.read_parquet(DATA_DIR / "test.parquet", columns=["molecule_id", *cols])
    train = pd.read_parquet(DATA_DIR / "train.parquet", columns=["ingest_lib", "normalized_smiles", *cols])
    train = train[train["ingest_lib"] == "enveda-180"]
    for df in (test, train):
        df["k"] = df["precursor_mz"].round(6).astype(str) + "|" + df["adduct"] + "|" + \
            df["base_peak_intensity"].round(1).astype(str)
    m = test.merge(train.drop_duplicates("k")[["k", "normalized_smiles"]], on="k")
    return m.groupby("molecule_id")["normalized_smiles"].agg(lambda s: s.mode().iloc[0]).to_dict()


def local_check(notebook: Path, pool: str = "base", fp: str | None = None, ranker: dict | None = None) -> None:
    root = WORK_DIR / "kaggle" / "fake_input"
    work = WORK_DIR / "kaggle" / "fake_working"
    links = {
        root / COMPETITION / "test.parquet": DATA_DIR / "test.parquet",
        root / COMPETITION / "sample_submission.csv": DATA_DIR / "sample_submission.csv",
        root / "casmi26-pool": POOLS[pool][1],
        root / "casmi26-library" / "train_clean.parquet": WORK_DIR / "train_clean.parquet",
        root / "casmi26-analog-index": WORK_DIR / "analog_index",
    }
    if fp:
        links.update({root / name: src for name, src in FP_MODELS[fp][1].items()})
    links[root / "casmi26-ranker-train"] = WORK_DIR / "ranker_train"
    if ranker and ranker.get("mode") == "rank31":  # the training rows' dataset, from its local copy
        links[root / ranker["dataset"].split("/")[1]] = REPO_ROOT / ranker["local"]
    for stale in root.iterdir() if root.exists() else []:  # links from an earlier check with other datasets
        if stale.is_symlink() and stale not in links:
            stale.unlink()
    for dst, src in links.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    work.mkdir(parents=True, exist_ok=True)
    nb = json.loads(notebook.read_text())
    script = work / "notebook_as_script.py"
    script.write_text("\n\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code"))
    env = {**os.environ, "CASMI_KAGGLE_INPUT": str(root), "CASMI_KAGGLE_WORK": str(work)}
    t0 = time.time()
    # Fresh interpreter, isolated from the repo checkout: it must import the embedded package.
    subprocess.run([sys.executable, "-I", str(script)], cwd=work, env=env, check=True)
    print(f"local notebook run: {time.time() - t0:.0f}s")

    from casmi.metric import evaluate
    from casmi.submit import validate_submission

    sub = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    validate_submission(sub, sample["molecule_id"].tolist())
    assert sub["molecule_id"].tolist() == sample["molecule_id"].tolist(), "row order differs from sample"
    truth = placeholder_truth()
    res = evaluate(dict(zip(sub["molecule_id"], sub["smiles"])), truth)
    print(f"placeholder test: {len(truth)} molecules with known answers, MRR@25 = {res['mrr@25']:.3f} "
          "(answers are in the library here, so this is a plumbing check, not an estimate)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--local-check", action="store_true")
    ap.add_argument("--mode", choices=["library", "combined"], default="library")
    ap.add_argument("--ranker", type=Path, default=WORK_DIR / "ranker_v1.json",
                    help="ranker config json: {'ranker': {...}} (v1) or {'weights': {...}} (v0); combined mode")
    ap.add_argument("--pool", choices=sorted(POOLS), default="base")
    ap.add_argument("--fp", choices=sorted(FP_MODELS), default=None, help="fingerprint models to attach (GPU)")
    ap.add_argument("--tier", type=Path, default=None, help="PubChem tier config json {'tier': {...}} (--pool pubchem)")
    ap.add_argument("--alt-adducts", type=Path, default=None,
                    help="alternative-adduct slots json {'alt_adducts': {...}} (analog-only ranker)")
    args = ap.parse_args()
    ranker = {}
    if args.mode == "combined":
        spec = json.loads(args.ranker.read_text())
        if "features" in spec:  # ranker v2 (validation/eval_ranker_v2.py): fitted in the notebook
            ranker = {"mode": "learned", "features": spec["features"], "bags": spec["bags"], "params": spec["params"]}
        elif "combiner" in spec:  # validation/eval_combiners.py: {"combiner": {"kind": ..., ...}}
            ranker = {"mode": "combiner", "spec": spec["combiner"]}
        elif "rank31" in spec:  # {"rank31": {"rows": <npz name>, "dataset": <owner/slug>, "local": <folder>}}
            ranker = {"mode": "rank31", **spec["rank31"]}
        else:
            ranker = spec.get("ranker") or {"mode": "weighted", "weights": spec["weights"]}
    tier = json.loads(args.tier.read_text())["tier"] if args.tier else None
    alt = json.loads(args.alt_adducts.read_text())["alt_adducts"] if args.alt_adducts else None
    notebook = build(args.out, args.mode, ranker, args.pool, args.fp, tier, alt)
    print(f"wrote {notebook} ({notebook.stat().st_size / 1e3:.0f} kB)")
    if args.local_check:
        local_check(notebook, args.pool, args.fp, ranker)


if __name__ == "__main__":
    main()
