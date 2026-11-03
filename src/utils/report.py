"""Small helpers for writing markdown tables without the optional `tabulate` dependency."""
from __future__ import annotations

import pandas as pd


def to_md(df: pd.DataFrame | pd.Series, index: bool = True, floatfmt: str = "{:.4f}") -> str:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(str(c) for c in col if str(c)) for col in df.columns]
    if index:
        idx_name = df.index.name if isinstance(df.index.name, str) else ""
        df = df.reset_index()
        if "index" in df.columns and not idx_name:
            df = df.rename(columns={"index": ""})

    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)
