"""Channel coverage and decade distribution stats."""
from __future__ import annotations

import pandas as pd


def build_channel_stats(df_base: pd.DataFrame) -> pd.DataFrame:
    """Per-channel coverage stats: date_min, date_max, n_obs, share_pct.

    Vectorized via groupby. Returns an empty DataFrame if required columns
    are missing or no valid dates exist.
    """
    if "_channel" not in df_base.columns or "_date" not in df_base.columns:
        return pd.DataFrame()

    total = len(df_base)
    valid = df_base[df_base["_date"].notna()]
    if valid.empty:
        return pd.DataFrame()

    stats = (
        valid.groupby("_channel", as_index=False)
        .agg(date_min=("_date", "min"), date_max=("_date", "max"), n_obs=("_date", "count"))
    )
    stats["share_pct"] = (stats["n_obs"] / total * 100).round(2)
    return stats.sort_values("n_obs", ascending=False).reset_index(drop=True)


def build_decade_distribution(df_base: pd.DataFrame) -> pd.DataFrame:
    """Long-format channel distribution per decade.

    Columns: _channel, decade (int), decade_label (str), n, pct (% within decade).
    For each decade, the sum of pct across all channels equals 100.
    """
    if "_channel" not in df_base.columns or "_date" not in df_base.columns:
        return pd.DataFrame()
    df = df_base[df_base["_date"].notna()].copy()
    df["decade"] = (df["_date"].dt.year // 10) * 10
    df["decade_label"] = df["decade"].astype(str) + "–" + (df["decade"] + 9).astype(str)
    cross = (
        df.groupby(["_channel", "decade_label", "decade"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    decade_totals = cross.groupby("decade")["n"].transform("sum")
    cross["pct"] = (cross["n"] / decade_totals * 100).round(1)
    return cross.sort_values(["_channel", "decade"]).reset_index(drop=True)
