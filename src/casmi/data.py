"""Column-wise access to the competition parquet files.

``train.parquet`` is ~3 GB, almost all of it in the peak arrays, so callers
should always request only the columns they need.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from casmi.config import DATA_DIR

META_COLUMNS = (
    "ingest_lib",
    "normalized_smiles",
    "inchikey14",
    "molecular_formula",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "adduct_orig",
    "precursor_mz",
    "precursor_error_ppm",
    "num_peaks",
    "base_peak_intensity",
    "collision_energy_ev",
    "collision_energy_orig_units",
)


def read_table(name: str, columns: Sequence[str] | None = None) -> pa.Table:
    """Read ``{name}.parquet`` from ``DATA_DIR``, restricted to ``columns``."""
    return pq.read_table(DATA_DIR / f"{name}.parquet", columns=list(columns) if columns else None)


def read_frame(name: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Like ``read_table`` but returns a DataFrame."""
    return read_table(name, columns).to_pandas()


def mean_ce(v: object, default: float = float("nan")) -> float:
    """Mean of a list-valued ``collision_energy_ev`` cell; ``default`` when it is missing or empty."""
    if v is None:
        return default
    a = np.atleast_1d(np.asarray(v, dtype=np.float64))
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else default
