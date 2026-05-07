"""Tests for ina_core.aggregation — temporal aggregation of tagged dataframes."""
from __future__ import annotations

import pandas as pd

from ina_core import (
    aggregate_by_period,
    build_descriptive_table,
    build_top_channels,
    periodize,
    tag_dataframe,
)


def _build_tagged(titles, dates, channels=None):
    n = len(titles)
    df = pd.DataFrame({
        "_date": pd.to_datetime(dates),
        "_channel": channels if channels else ["BFMTV"] * n,
        "title": titles,
        "_title_norm": titles,
        "source_file": ["test"] * n,
    })
    return tag_dataframe(df, title_col="title",
                         concept_norm=["inflation"],
                         up_norm=["hausse"], down_norm=["baisse"],
                         title_norm_col="_title_norm")


def test_periodize_monthly():
    series = pd.to_datetime(["2020-01-15", "2020-01-30", "2020-02-10"])
    out = periodize(pd.Series(series), "Mensuelle")
    assert out.iloc[0] == pd.Timestamp("2020-01-01")
    assert out.iloc[1] == pd.Timestamp("2020-01-01")
    assert out.iloc[2] == pd.Timestamp("2020-02-01")


def test_periodize_quarterly():
    series = pd.to_datetime(["2020-01-15", "2020-04-15", "2020-07-15"])
    out = periodize(pd.Series(series), "Trimestrielle")
    assert out.iloc[0] == pd.Timestamp("2020-01-01")  # Q1
    assert out.iloc[1] == pd.Timestamp("2020-04-01")  # Q2
    assert out.iloc[2] == pd.Timestamp("2020-07-01")  # Q3


def test_periodize_annual():
    series = pd.to_datetime(["2020-06-15", "2021-03-10"])
    out = periodize(pd.Series(series), "Annuelle")
    assert out.iloc[0] == pd.Timestamp("2020-01-01")
    assert out.iloc[1] == pd.Timestamp("2021-01-01")


def test_periodize_unknown_frequency_defaults_monthly():
    series = pd.to_datetime(["2020-01-15"])
    out = periodize(pd.Series(series), "Hebdomadaire")  # not supported
    assert out.iloc[0] == pd.Timestamp("2020-01-01")


def test_aggregate_basic_counts():
    df = _build_tagged(
        titles=["inflation en hausse", "actualite generale", "inflation en baisse"],
        dates=["2020-01-15", "2020-01-20", "2020-02-10"],
    )
    stats = aggregate_by_period(df, "Mensuelle")
    assert len(stats) == 2  # January + February
    jan = stats[stats["period_start"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert jan["total_titles"] == 2
    assert jan["matched_titles"] == 1
    assert jan["up_titles"] == 1
    assert jan["down_titles"] == 0


def test_aggregate_frequency_is_matched_over_total():
    df = _build_tagged(
        titles=["inflation A", "actualite B", "inflation C", "actualite D"],
        dates=["2020-01-01"] * 4,
    )
    stats = aggregate_by_period(df, "Mensuelle")
    assert stats["frequency"].iloc[0] == 0.5


def test_aggregate_net_signal_is_up_minus_down():
    df = _build_tagged(
        titles=["inflation en hausse", "inflation en hausse", "inflation en baisse"],
        dates=["2020-01-01"] * 3,
    )
    stats = aggregate_by_period(df, "Mensuelle")
    assert stats["net_signal"].iloc[0] == 1  # 2 up - 1 down


def test_aggregate_handles_zero_strict_matched_without_dividing_by_zero():
    """direction_share_up uses pd.NA when no strict matches — no ZeroDivisionError."""
    df = _build_tagged(
        titles=["actualite generale"],
        dates=["2020-01-01"],
    )
    stats = aggregate_by_period(df, "Mensuelle")
    # No match → strict_matched_titles == 0, share should be NA
    assert pd.isna(stats["direction_share_up"].iloc[0])


def test_descriptive_table_returns_twelve_indicators():
    df = _build_tagged(
        titles=["inflation en hausse", "inflation en baisse"],
        dates=["2020-01-01", "2020-02-01"],
    )
    stats = aggregate_by_period(df, "Mensuelle")
    desc = build_descriptive_table(stats, df)
    assert len(desc) == 12
    assert set(desc.columns) == {"indicateur", "valeur"}


def test_descriptive_table_empty_when_no_data():
    desc = build_descriptive_table(pd.DataFrame(), pd.DataFrame())
    assert desc.empty


def test_top_channels_caps_at_ten():
    titles = [f"inflation {i}" for i in range(15)]
    channels = [f"chaine_{i}" for i in range(15)]
    df = _build_tagged(titles=titles, dates=["2020-01-01"] * 15, channels=channels)
    top = build_top_channels(df)
    assert len(top) <= 10


def test_top_channels_sorted_by_matched_titles_desc():
    df = _build_tagged(
        titles=["inflation A", "actualite B", "inflation C", "inflation D"],
        dates=["2020-01-01"] * 4,
        channels=["TF1", "BFMTV", "TF1", "TF1"],
    )
    top = build_top_channels(df)
    # TF1 has 3 matches (A, C, D), BFMTV has 0
    assert top.iloc[0]["_channel"] == "TF1"
    assert top.iloc[0]["matched_titles"] == 3
