"""Do the library and analog channels still see the neighbours a split v4 / v5 removed?

Per split: the removed neighbour structures (``neighbour_keys.parquet``) that are still library rows of
the split (the mask the library and analog channels search), and the feature-table rows whose
candidate is a removed neighbour with ``library_max`` > 0 or ``in_library`` set. Both must be 0; the
fragment channel is rule based and has nothing to leak.

    python validation/check_neighbour_leak.py --features outputs/ranker_features_e5 --splits v5_seed0 r5_seed10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from casmi.channels.analog import library_row_keys
from casmi.config import REPO_ROOT, WORK_DIR
from casmi.harness import CLEAN_LIBRARY, load_val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=Path("outputs/ranker_features_e5"))
    ap.add_argument("--splits", nargs="+",
                    default=[f"v5_seed{s}" for s in range(3)] + [f"r5_seed{s}" for s in range(10, 15)])
    args = ap.parse_args()
    row_key = library_row_keys(CLEAN_LIBRARY)
    bad = 0
    for split in args.splits:
        folder = WORK_DIR / "splits" / split
        removed = set(pd.read_parquet(folder / "neighbour_keys.parquet")["key"])
        data = load_val(folder)
        library = set(row_key.reindex(data.in_library.index[data.in_library.to_numpy()]).dropna())
        feats = pd.read_parquet(REPO_ROOT / args.features / f"{split}.parquet",
                                columns=["key", "library_max", "in_library"])
        rows = feats[feats["key"].isin(removed)]
        in_mask = len(removed & library)
        lib_max, flag = int((rows["library_max"] > 0).sum()), int(rows["in_library"].sum())
        bad += in_mask + lib_max + flag
        print(f"{split}: {len(removed):,} removed neighbours, {in_mask} in the library / analog mask; "
              f"{len(rows):,} window rows, library_max > 0: {lib_max}, in_library: {flag}", flush=True)
    print("no leak" if bad == 0 else f"LEAK: {bad} rows / structures")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
