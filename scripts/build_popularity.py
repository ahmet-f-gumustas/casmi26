"""PubChem popularity per pool structure → ``<pool>/popularity.parquet``.

Source: ``prvsiyan/pubchem-popularity-counts`` (CC0 1.0; NCBI PubChem CID-SID / CID-PMID,
2 Oct 2026): per InChIKey first block, the number of distinct PubChem substance records
(``sid``) and PubMed records (``pmid``). The scorer key of a pool structure is that block, so
the lookup is exact; a key PubChem does not have gets 0.

    python scripts/build_popularity.py [--src external/prvsiyan/pubchem_popularity] [--pools pool pool_pubchem]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.config import EXTERNAL_DIR


def lookup(keys: pd.Series, ik14: np.ndarray, sid: np.ndarray, pmid: np.ndarray) -> pd.DataFrame:
    q = keys.to_numpy().astype("S14")
    pos = np.minimum(np.searchsorted(ik14, q), len(ik14) - 1)
    hit = ik14[pos] == q
    return pd.DataFrame({"key": keys.to_numpy(), "pc_sid": np.where(hit, sid[pos], 0).astype(np.int64),
                         "pc_pmid": np.where(hit, pmid[pos], 0).astype(np.int64)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=EXTERNAL_DIR / "prvsiyan" / "pubchem_popularity")
    ap.add_argument("--pools", nargs="+", default=["pool", "pool_pubchem"])
    args = ap.parse_args()
    ik14 = np.load(args.src / "ap2pop_ik14.npy", mmap_mode="r", allow_pickle=False)
    sid = np.load(args.src / "ap2pop_ik14_sid.npy", mmap_mode="r", allow_pickle=False)
    pmid = np.load(args.src / "ap2pop_ik14_pmid.npy", mmap_mode="r", allow_pickle=False)
    for name in args.pools:
        meta = pd.read_parquet(EXTERNAL_DIR / name / "meta.parquet")
        pop = lookup(meta["key"], ik14, sid, pmid)
        pop.to_parquet(EXTERNAL_DIR / name / "popularity.parquet", index=False)
        found = pop["pc_sid"] > 0
        by = {c: f"{found[meta[c].to_numpy(bool)].mean():.1%}" for c in ("in_train", "in_coconut", "in_pubchem")
              if c in meta}
        print(f"{name}: {len(pop):,} keys, in PubChem {found.mean():.1%} {by}; median sid {pop['pc_sid'].median():.0f}, "
              f"with PubMed {(pop['pc_pmid'] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
