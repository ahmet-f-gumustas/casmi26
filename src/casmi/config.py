"""Paths and global constants, resolved for Kaggle vs. local runs."""

from __future__ import annotations

import os
from pathlib import Path

COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
TOP_K = 25
RDKIT_VERSION = "2026.03.3"

REPO_ROOT = Path(__file__).resolve().parents[2]
ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    DATA_DIR = Path("/kaggle/input") / COMPETITION
    EXTERNAL_DIR = Path("/kaggle/input")
    WORK_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path(os.environ.get("CASMI_DATA", REPO_ROOT / "data"))
    EXTERNAL_DIR = Path(os.environ.get("CASMI_EXTERNAL", REPO_ROOT / "external"))
    WORK_DIR = Path(os.environ.get("CASMI_WORK", REPO_ROOT / "outputs"))
