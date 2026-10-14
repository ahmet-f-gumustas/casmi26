"""Stage and upload the private Kaggle datasets the submission notebook reads.

- ``red6161/casmi26-pool``: external/pool (masses.npy, fps.npy, meta.parquet)
- ``red6161/casmi26-pool-pubchem``: external/pool_pubchem, fingerprints as compressed ``fps.npz``
- ``red6161/casmi26-library``: outputs/train_clean.parquet
- ``red6161/casmi26-analog-index``: outputs/analog_index, flattened (``<mode>__<file>``)
- ``red6161/casmi26-wheels``: ms_entropy wheels for CPython 3.10–3.13 (RDKit comes from
  the public ``metric/rdkit-2026-3-3-wheel`` dataset)

Files are hard-linked into ``outputs/kaggle/datasets/<name>/`` (no copies), then
``kaggle datasets create`` (first time) or ``kaggle datasets version`` (``--update``).

    python scripts/kaggle_datasets.py pool library index wheels [--update -m "message"]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from casmi.candidates import POOL_DIR
from casmi.config import WORK_DIR

OWNER = "red6161"
STAGE = WORK_DIR / "kaggle" / "datasets"
DATASETS = {
    "pool": ("casmi26-pool", [POOL_DIR / "masses.npy", POOL_DIR / "fps.npy", POOL_DIR / "meta.parquet"],
             "Candidate pool for CASMI 2026: COCONUT (Oct 2026, CC0) and training structures, deduped by the "
             "metric's tautomer-canonical InChIKey14, sorted by neutral mass, with Morgan r2/2048 fingerprints."),
    "pool_pubchem": ("casmi26-pool-pubchem", [POOL_DIR.with_name("pool_pubchem") / n for n in ("masses.npy", "fps.npy",
                                                                                              "meta.parquet")],
                     "Candidate pool extended with NP-like PubChem structures (CC0 mass index by prvsiyan, filtered by "
                     "RDKit NP-likeness / COCONUT scaffolds), deduped by scorer key, mass-sorted; fingerprints "
                     "compressed in fps.npz."),
    "library": ("casmi26-library", [WORK_DIR / "train_clean.parquet"],
                "Competition training spectra cleaned like the test: peaks above precursor + 2 Da removed, base "
                "peak < 1000 dropped, precursor checked against the labelled structure (|ppm| <= 20)."),
    "index": ("casmi26-analog-index", [], "Flash entropy search index (ms_entropy 1.5.3) of the cleaned library, one "
              "per ionization mode, flattened as <mode>__<file>; library row / precursor maps as .npy."),
    "wheels": ("casmi26-wheels", sorted((STAGE / "casmi26-wheels").glob("ms_entropy-*.whl")),
               "Offline wheels: ms_entropy 1.5.3 for CPython 3.10-3.13 (manylinux x86_64)."),
}


def index_files(index_dir: Path = WORK_DIR / "analog_index") -> list[tuple[Path, str]]:
    """(source, flattened name) pairs: Kaggle datasets cannot hold sub-folders."""
    out = []
    for p in sorted(index_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(index_dir)
            out.append((p, "__".join(rel.parts)))
    return out


def stage(name: str) -> Path:
    slug, files, description = DATASETS[name]
    folder = STAGE / slug
    folder.mkdir(parents=True, exist_ok=True)
    pairs = index_files() if name == "index" else [(f, f.name) for f in files]
    if not pairs:
        raise SystemExit(f"{name}: no files to upload")
    for src, dst_name in pairs:
        if name == "pool_pubchem" and src.name == "fps.npy":  # ~1 GB raw; compressed copy keeps the dataset small
            dst = folder / "fps.npz"
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                np.savez_compressed(dst, fps=np.load(src))
            continue
        dst = folder / dst_name
        if dst.resolve() == src.resolve():
            continue
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    meta = {"title": slug, "id": f"{OWNER}/{slug}", "licenses": [{"name": "other"}], "description": description}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(DATASETS))
    ap.add_argument("--update", action="store_true", help="push a new version instead of creating")
    ap.add_argument("-m", "--message", default="update")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    kaggle = str(Path(sys.executable).with_name("kaggle"))
    for name in args.names:
        folder = stage(name)
        size = sum(p.stat().st_size for p in folder.iterdir() if p.name != "dataset-metadata.json")
        cmd = ([kaggle, "datasets", "version", "-p", str(folder), "-m", args.message, "-t"] if args.update
               else [kaggle, "datasets", "create", "-p", str(folder), "-t"])
        print(f"{OWNER}/{DATASETS[name][0]}: {size / 1e6:.0f} MB → {' '.join(cmd[1:3])}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
