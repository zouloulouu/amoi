"""Tests for ina_core.tagging — keyword preparation and dataframe tagging."""
from __future__ import annotations

import pandas as pd

from ina_core import (
    DIRECTION_AMBIGUOUS,
    DIRECTION_DOWN,
    DIRECTION_FLAT,
    DIRECTION_UP,
    count_occurrences,
    prepare_keywords,
    tag_dataframe,
)


def _make_df(titles, channels=None, dates=None):
    n = len(titles)
    return pd.DataFrame({
        "_date": pd.to_datetime(dates if dates else [f"2020-01-{i+1:02d}" for i in range(n)]),
        "_channel": channels if channels else ["BFMTV"] * n,
        "title": titles,
        "_title_norm": titles,  # already normalized fixtures
        "source_file": ["test"] * n,
    })


def test_prepare_keywords_normalizes_and_dedupes():
    out = prepare_keywords(["Inflation", "INFLATION", "  inflation  ", "ipc"])
    assert out == ["inflation", "ipc"]


def test_prepare_keywords_filters_empty():
    assert prepare_keywords(["", "  ", None]) == []


def test_count_occurrences_short_alpha_keyword_uses_word_boundary():
    """'ipc' (3 chars, alphabetic) must NOT match 'epicea'."""
    assert count_occurrences("epicea fournit du bois", ["ipc"]) == 0
    assert count_occurrences("ipc en hausse", ["ipc"]) == 1


def test_count_occurrences_long_keyword_uses_substring():
    """Multi-word keywords use plain substring matching."""
    assert count_occurrences("hausse des prix de l energie", ["hausse des prix"]) == 1


def test_count_occurrences_empty_text_returns_zero():
    assert count_occurrences("", ["inflation"]) == 0


def test_count_occurrences_multiple_hits():
    assert count_occurrences("inflation, l inflation reste forte", ["inflation"]) == 2


def test_tag_dataframe_does_not_mutate_input():
    df = _make_df(["inflation en hausse"])
    original_columns = list(df.columns)
    tag_dataframe(df, title_col="title",
                  concept_norm=["inflation"], up_norm=["hausse"], down_norm=[],
                  title_norm_col="_title_norm")
    assert list(df.columns) == original_columns, "tag_dataframe mutated input!"


def test_tag_dataframe_direction_up():
    df = _make_df(["inflation en hausse"])
    out = tag_dataframe(df, title_col="title",
                        concept_norm=["inflation"], up_norm=["hausse"], down_norm=["baisse"],
                        title_norm_col="_title_norm")
    assert out.loc[0, "is_match"] == 1
    assert out.loc[0, "direction"] == DIRECTION_UP


def test_tag_dataframe_direction_down():
    df = _make_df(["inflation en baisse"])
    out = tag_dataframe(df, title_col="title",
                        concept_norm=["inflation"], up_norm=["hausse"], down_norm=["baisse"],
                        title_norm_col="_title_norm")
    assert out.loc[0, "direction"] == DIRECTION_DOWN


def test_tag_dataframe_direction_ambiguous():
    df = _make_df(["inflation en hausse mais baisse possible"])
    out = tag_dataframe(df, title_col="title",
                        concept_norm=["inflation"], up_norm=["hausse"], down_norm=["baisse"],
                        title_norm_col="_title_norm")
    assert out.loc[0, "direction"] == DIRECTION_AMBIGUOUS


def test_tag_dataframe_direction_flat_when_matched_but_no_signal():
    df = _make_df(["inflation reste stable"])
    out = tag_dataframe(df, title_col="title",
                        concept_norm=["inflation"], up_norm=["hausse"], down_norm=["baisse"],
                        title_norm_col="_title_norm")
    assert out.loc[0, "is_match"] == 1
    assert out.loc[0, "direction"] == DIRECTION_FLAT


def test_tag_dataframe_direction_flat_when_not_matched():
    df = _make_df(["actualite economique generale"])
    out = tag_dataframe(df, title_col="title",
                        concept_norm=["inflation"], up_norm=["hausse"], down_norm=["baisse"],
                        title_norm_col="_title_norm")
    assert out.loc[0, "is_match"] == 0
    assert out.loc[0, "direction"] == DIRECTION_FLAT


def test_tag_dataframe_falls_back_to_normalize_when_no_norm_col():
    """Without title_norm_col, tag_dataframe normalizes the title_col on the fly."""
    df = pd.DataFrame({
        "_date": pd.to_datetime(["2020-01-01"]),
        "title": ["Inflation à la HAUSSE"],
    })
    out = tag_dataframe(df, title_col="title",
                        concept_norm=["inflation"], up_norm=["hausse"], down_norm=[])
    assert out.loc[0, "is_match"] == 1
    assert out.loc[0, "direction"] == DIRECTION_UP
