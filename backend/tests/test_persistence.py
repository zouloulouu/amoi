"""Tests for the on-disk persistence layer (df_base + tagged dataframes)."""
from __future__ import annotations

import pandas as pd

from ina_core.cache import TaggingCache
from ina_core.store.persistence import LocalDiskPersistence


def _make_df(n: int = 4, channel: str = "BFMTV") -> pd.DataFrame:
    return pd.DataFrame({
        "source_file": [f"f{i}.parquet" for i in range(n)],
        "title": [f"inflation hausse {i}" for i in range(n)],
        "_title_norm": [f"inflation hausse {i}" for i in range(n)],
        "_date": pd.date_range("2024-01-01", periods=n, freq="MS"),
        "_channel": [channel] * n,
    })


# ─── df_base round-trip ──────────────────────────────────────────────────────


def test_corpus_returns_none_when_no_cache(tmp_path):
    p = LocalDiskPersistence(tmp_path / "cache")
    assert p.read_corpus("any_token") is None


def test_corpus_round_trip(tmp_path):
    p = LocalDiskPersistence(tmp_path / "cache")
    df = _make_df(n=10)
    p.write_corpus(df, snapshot_token="v_2024_01")
    loaded = p.read_corpus("v_2024_01")
    assert loaded is not None
    assert len(loaded) == 10
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))


def test_corpus_token_mismatch_invalidates(tmp_path):
    p = LocalDiskPersistence(tmp_path / "cache")
    p.write_corpus(_make_df(n=3), snapshot_token="v_old")
    assert p.read_corpus("v_new") is None  # different token → cache miss


# ─── tagged round-trip ──────────────────────────────────────────────────────


def test_tagged_returns_none_when_no_cache(tmp_path):
    p = LocalDiskPersistence(tmp_path / "cache")
    assert p.read_tagged("inflation", ("ipc",), (), ()) is None


def test_tagged_round_trip(tmp_path):
    p = LocalDiskPersistence(tmp_path / "cache")
    df = _make_df(n=5)
    p.write_tagged(df, "inflation", ("ipc",), ("hausse",), ("baisse",))
    loaded = p.read_tagged("inflation", ("ipc",), ("hausse",), ("baisse",))
    assert loaded is not None
    assert len(loaded) == 5


def test_tagged_different_signatures_dont_collide(tmp_path):
    """Two themes with different keywords store under different files."""
    p = LocalDiskPersistence(tmp_path / "cache")
    df_a = _make_df(n=3, channel="A")
    df_b = _make_df(n=7, channel="B")
    p.write_tagged(df_a, "inflation", ("ipc",), (), ())
    p.write_tagged(df_b, "inflation", ("ipc", "prix"), (), ())  # different concept
    assert len(p.read_tagged("inflation", ("ipc",), (), ())) == 3
    assert len(p.read_tagged("inflation", ("ipc", "prix"), (), ())) == 7


def test_invalidate_theme_drops_all_signatures(tmp_path):
    p = LocalDiskPersistence(tmp_path / "cache")
    p.write_tagged(_make_df(), "inflation", ("a",), (), ())
    p.write_tagged(_make_df(), "inflation", ("b",), (), ())
    p.write_tagged(_make_df(), "chomage", ("c",), (), ())
    removed = p.invalidate_theme("inflation")
    assert removed == 2
    assert p.read_tagged("inflation", ("a",), (), ()) is None
    assert p.read_tagged("inflation", ("b",), (), ()) is None
    # Other theme untouched
    assert p.read_tagged("chomage", ("c",), (), ()) is not None


# ─── TaggingCache hits the disk on memory miss ──────────────────────────────


def test_tagging_cache_uses_disk_when_memory_miss(tmp_path):
    """Cold in-memory cache hydrates from disk without recomputing."""
    p = LocalDiskPersistence(tmp_path / "cache")
    cache = TaggingCache(maxsize=2, persistence=p)

    fake_tagged = _make_df(n=8)
    p.write_tagged(fake_tagged, "inflation", ("ipc",), (), ())

    compute_calls = {"n": 0}

    def _never_called():
        compute_calls["n"] += 1
        raise AssertionError("compute_fn should not run when disk cache is warm")

    out = cache.get_or_compute(
        "inflation", ("ipc",), (), (), _never_called
    )
    assert len(out) == 8
    assert compute_calls["n"] == 0
    # Now in-memory layer is also warm
    assert len(cache) == 1


def test_tagging_cache_writes_to_disk_on_compute(tmp_path):
    """Computed entries are persisted to disk for next process."""
    p = LocalDiskPersistence(tmp_path / "cache")
    cache = TaggingCache(maxsize=2, persistence=p)

    df = _make_df(n=5)
    cache.get_or_compute("chomage", ("a",), (), (), lambda: df)

    on_disk = p.read_tagged("chomage", ("a",), (), ())
    assert on_disk is not None
    assert len(on_disk) == 5


def test_tagging_cache_invalidate_clears_disk(tmp_path):
    p = LocalDiskPersistence(tmp_path / "cache")
    cache = TaggingCache(maxsize=2, persistence=p)
    cache.get_or_compute("chomage", ("a",), (), (), lambda: _make_df(n=3))
    assert p.read_tagged("chomage", ("a",), (), ()) is not None

    cache.invalidate("chomage")
    assert p.read_tagged("chomage", ("a",), (), ()) is None
