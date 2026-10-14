"""Rebuild prvsiyan's scored V33 notebook as an unchanged private Kaggle kernel.

The public notebook ``prvsiyan/analog-propagation-casmi-2026-baseline`` (Apache 2.0)
now runs in a "presentation" mode that writes no submission. Its scored V33 cell
(public LB 0.344, script version 356401066) is embedded byte-for-byte, with sha256
pins, as ``ENVEDA_ORIGINAL_CELL_B64``. This script decodes that cell and its original
kernel metadata (T4 GPU, pinned image, seven input datasets) and writes a kernel
folder ready for ``kaggle kernels push``. Nothing in the cell is modified; only the
kernel id/title/privacy change.

    kaggle kernels pull prvsiyan/analog-propagation-casmi-2026-baseline -p <dir> -m
    python scripts/fork_prvsiyan.py <dir>/analog-propagation-casmi-2026-baseline.ipynb
    kaggle kernels push -p outputs/kaggle/prvsiyan_v33_fork
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

from casmi.config import WORK_DIR

KERNEL_ID = "red6161/casmi26-prvsiyan-v33-fork"
TITLE = "casmi26 prvsiyan v33 fork"


def embedded_constants(notebook: Path) -> dict[str, str]:
    cells = json.loads(notebook.read_text())["cells"]
    code = next("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    tree = ast.parse(code)  # parsed, never executed
    return {n.targets[0].id: n.value.value for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path, help="pulled prvsiyan notebook (.ipynb)")
    ap.add_argument("--out", type=Path, default=WORK_DIR / "kaggle" / "prvsiyan_v33_fork")
    args = ap.parse_args()

    c = embedded_constants(args.notebook)
    cell_raw = base64.b64decode(c["ENVEDA_ORIGINAL_CELL_B64"])
    meta_raw = base64.b64decode(c["ENVEDA_ORIGINAL_METADATA_B64"])
    cell = json.loads(cell_raw)
    body = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if hashlib.sha256(body.encode()).hexdigest() != c["ENVEDA_ORIGINAL_BODY_SHA256"]:
        raise SystemExit("embedded V33 body does not match its sha256 pin")
    if hashlib.sha256(meta_raw).hexdigest() != c["ENVEDA_ORIGINAL_METADATA_SHA256"]:
        raise SystemExit("embedded V33 metadata does not match its sha256 pin")
    meta = json.loads(meta_raw)

    args.out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [cell], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                       "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
    code_file = "prvsiyan-v33-fork.ipynb"
    (args.out / code_file).write_text(json.dumps(nb))
    fork_meta = {**{k: meta[k] for k in ("language", "kernel_type", "enable_gpu", "enable_tpu", "enable_internet",
                                         "dataset_sources", "kernel_sources", "competition_sources",
                                         "model_sources", "docker_image", "machine_shape")},
                 "id": KERNEL_ID, "title": TITLE, "code_file": code_file, "is_private": True, "keywords": []}
    (args.out / "kernel-metadata.json").write_text(json.dumps(fork_meta, indent=2) + "\n")
    print(f"wrote {args.out} (V33 body sha256 {c['ENVEDA_ORIGINAL_BODY_SHA256'][:12]}…, "
          f"GPU={meta['enable_gpu']}, {len(meta['dataset_sources'])} datasets)")


if __name__ == "__main__":
    main()
