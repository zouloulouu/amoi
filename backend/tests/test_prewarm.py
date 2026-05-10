"""Tests for the background prewarming helper."""
from __future__ import annotations

import time

import pandas as pd

from ina_core.cache import TaggingCache
from ina_core.prewarm import prewarm_themes_async, prewarm_themes_blocking


def _make_corpus():
    return pd.DataFrame({
        "title": [
            "inflation en hausse",
            "chomage en baisse",
            "actualite generale",
            "inflation et hausse forte",
        ],
        "_title_norm": [
            "inflation en hausse",
            "chomage en baisse",
            "actualite generale",
            "inflation et hausse forte",
        ],
        "_date": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "_channel": ["BFMTV"] * 4,
    })


def test_prewarm_blocking_populates_cache():
    cache = TaggingCache(maxsize=8)
    df = _make_corpus()
    dictionaries = {
        "inflation": {"concept": ["inflation"], "up": ["hausse"], "down": []},
        "chomage": {"concept": ["chomage"], "up": [], "down": ["baisse"]},
    }

    assert len(cache) == 0
    prewarm_themes_blocking(cache, df, dictionaries)
    assert len(cache) == 2


def test_prewarm_skips_themes_without_concept():
    cache = TaggingCache(maxsize=8)
    df = _make_corpus()
    dictionaries = {
        "inflation": {"concept": ["inflation"], "up": [], "down": []},
        "empty_theme": {"concept": [], "up": [], "down": []},
    }
    prewarm_themes_blocking(cache, df, dictionaries)
    assert len(cache) == 1


def test_prewarm_async_returns_thread_and_completes():
    cache = TaggingCache(maxsize=8)
    df = _make_corpus()
    dictionaries = {"inflation": {"concept": ["inflation"], "up": [], "down": []}}

    thread = prewarm_themes_async(cache, df, dictionaries)
    assert thread.daemon is True
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(cache) == 1


def test_prewarm_handles_empty_corpus():
    cache = TaggingCache(maxsize=8)
    prewarm_themes_blocking(cache, pd.DataFrame(), {"inflation": {"concept": ["x"]}})
    assert len(cache) == 0


def test_prewarm_handles_empty_dict():
    cache = TaggingCache(maxsize=8)
    prewarm_themes_blocking(cache, _make_corpus(), {})
    assert len(cache) == 0
