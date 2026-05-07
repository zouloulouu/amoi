"""Tests for ina_core.coverage — channel stats and decade distribution."""
from __future__ import annotations

import pandas as pd

from ina_core import build_channel_stats, build_decade_distribution


def _make_df(rows):
    return pd.DataFrame(rows)


def test_channel_stats_returns_one_row_per_channel():
    df = _make_df([
        {"_channel": "TF1", "_date": pd.Timestamp("2020-01-01")},
        {"_channel": "TF1", "_date": pd.Timestamp("2021-06-15")},
        {"_channel": "BFMTV", "_date": pd.Timestamp("2022-03-10")},
    ])
    out = build_channel_stats(df)
    assert set(out["_channel"]) == {"TF1", "BFMTV"}
    assert len(out) == 2


def test_channel_stats_dates_are_min_and_max():
    df = _make_df([
        {"_channel": "TF1", "_date": pd.Timestamp("2020-01-01")},
        {"_channel": "TF1", "_date": pd.Timestamp("2021-06-15")},
    ])
    out = build_channel_stats(df)
    row = out[out["_channel"] == "TF1"].iloc[0]
    assert row["date_min"] == pd.Timestamp("2020-01-01")
    assert row["date_max"] == pd.Timestamp("2021-06-15")
    assert row["n_obs"] == 2


def test_channel_stats_share_pct_sums_to_100():
    df = _make_df([
        {"_channel": "TF1", "_date": pd.Timestamp("2020-01-01")},
        {"_channel": "TF1", "_date": pd.Timestamp("2020-02-01")},
        {"_channel": "BFMTV", "_date": pd.Timestamp("2020-03-01")},
        {"_channel": "BFMTV", "_date": pd.Timestamp("2020-04-01")},
    ])
    out = build_channel_stats(df)
    assert abs(out["share_pct"].sum() - 100.0) < 0.01


def test_channel_stats_returns_empty_when_columns_missing():
    df = pd.DataFrame({"foo": [1, 2]})
    assert build_channel_stats(df).empty


def test_channel_stats_returns_empty_when_no_valid_dates():
    df = _make_df([
        {"_channel": "TF1", "_date": pd.NaT},
    ])
    assert build_channel_stats(df).empty


def test_decade_distribution_buckets_correctly():
    df = _make_df([
        {"_channel": "TF1", "_date": pd.Timestamp("2005-06-01")},
        {"_channel": "TF1", "_date": pd.Timestamp("2015-03-15")},
        {"_channel": "TF1", "_date": pd.Timestamp("2018-12-20")},
    ])
    out = build_decade_distribution(df)
    decades = set(out["decade"])
    assert decades == {2000, 2010}


def test_decade_distribution_pct_sums_to_100_per_decade():
    """For each decade, the sum of pct across channels must be 100%."""
    df = _make_df([
        {"_channel": "TF1", "_date": pd.Timestamp("2005-06-01")},
        {"_channel": "TF1", "_date": pd.Timestamp("2005-09-01")},
        {"_channel": "BFMTV", "_date": pd.Timestamp("2005-12-01")},
        {"_channel": "BFMTV", "_date": pd.Timestamp("2020-01-01")},
        {"_channel": "TF1", "_date": pd.Timestamp("2020-06-15")},
    ])
    out = build_decade_distribution(df)
    for decade, group in out.groupby("decade"):
        assert abs(group["pct"].sum() - 100.0) < 0.1, f"{decade} pct sum != 100"


def test_decade_distribution_channel_share_of_decade():
    """For 2000s: TF1 has 2 obs, BFMTV has 1 obs → TF1 = 66.7%, BFMTV = 33.3%."""
    df = _make_df([
        {"_channel": "TF1", "_date": pd.Timestamp("2005-06-01")},
        {"_channel": "TF1", "_date": pd.Timestamp("2005-09-01")},
        {"_channel": "BFMTV", "_date": pd.Timestamp("2005-12-01")},
    ])
    out = build_decade_distribution(df)
    tf1_2000s = out[(out["_channel"] == "TF1") & (out["decade"] == 2000)].iloc[0]
    bfm_2000s = out[(out["_channel"] == "BFMTV") & (out["decade"] == 2000)].iloc[0]
    assert abs(tf1_2000s["pct"] - 66.7) < 0.1
    assert abs(bfm_2000s["pct"] - 33.3) < 0.1


def test_decade_distribution_returns_empty_when_columns_missing():
    df = pd.DataFrame({"foo": [1, 2]})
    assert build_decade_distribution(df).empty
