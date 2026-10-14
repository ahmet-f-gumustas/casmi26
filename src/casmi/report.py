"""Small helpers for writing markdown reports from scripts."""

from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    """Minimal markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def share_table(s: pd.Series, name: str, top: int | None = None) -> pd.DataFrame:
    """Value counts of ``s`` (nulls included) with their share, as a two-column frame."""
    vc = s.value_counts(dropna=False)
    if top is not None and len(vc) > top:
        vc = pd.concat([vc.head(top), pd.Series({"(other)": vc.iloc[top:].sum()})])
    out = vc.rename_axis(name).reset_index(name="n")
    out[name] = out[name].astype(str)
    out["share"] = out["n"] / len(s)
    return out
