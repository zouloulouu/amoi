"""Tests for DataStore (local snapshot + HF fallback)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ina_core.store.config import Settings
from ina_core.store.data_store import DataStore


def _make_clean_parquet(path: Path, n: int = 3, channel: str = "BFMTV"):
    df = pd.DataFrame({
        "source_file": [path.name] * n,
        "title": [f"Inflation hausse {i}" for i in range(n)],
        "_title_norm": [f"inflation hausse {i}" for i in range(n)],
        "_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "_channel": [channel] * n,
    })
    df.to_parquet(path)


def _make_settings(project_root: Path, hf_token=None) -> Settings:
    return Settings(
        hf_repo_id="fake/repo",
        hf_parquet_files=("a.parquet", "b.parquet"),
        hf_token=hf_token,
        project_root=project_root,
    )


# ─── Local snapshot ─────────────────────────────────────────────────────────


def test_load_local_returns_empty_when_no_current_pointer(tmp_path):
    settings = _make_settings(tmp_path)
    store = DataStore(settings)
    df, issues = store.load(prefer="local")
    assert df.empty
    assert any("Snapshot local introuvable" in i for i in issues)


def test_load_local_reads_snapshot_pointed_by_current(tmp_path):
    snapshot = tmp_path / "data" / "clean" / "v_test"
    snapshot.mkdir(parents=True)
    _make_clean_parquet(snapshot / "a.parquet", n=3, channel="BFMTV")
    _make_clean_parquet(snapshot / "b.parquet", n=2, channel="TF1")
    (tmp_path / "data" / "clean" / "CURRENT").write_text("v_test")

    store = DataStore(_make_settings(tmp_path))
    df, issues = store.load(prefer="local")
    assert len(df) == 5
    assert set(df["_channel"]) == {"BFMTV", "TF1"}


def test_load_local_reports_missing_files(tmp_path):
    snapshot = tmp_path / "data" / "clean" / "v_test"
    snapshot.mkdir(parents=True)
    _make_clean_parquet(snapshot / "a.parquet", n=2)
    # b.parquet is intentionally missing
    (tmp_path / "data" / "clean" / "CURRENT").write_text("v_test")

    store = DataStore(_make_settings(tmp_path))
    df, issues = store.load(prefer="local")
    assert len(df) == 2
    assert any("b.parquet" in i for i in issues)


# ─── HF fallback ────────────────────────────────────────────────────────────


def test_load_auto_falls_back_to_hf_when_local_empty(tmp_path):
    settings = _make_settings(tmp_path, hf_token="hf_xxx")
    store = DataStore(settings)

    # Create an in-memory parquet that requests.get will return.
    df_remote = pd.DataFrame({
        "source_file": ["x"],
        "title": ["Inflation"],
        "_title_norm": ["inflation"],
        "_date": pd.to_datetime(["2024-01-01"]),
        "_channel": ["BFMTV"],
    })
    import io as _io
    buf = _io.BytesIO()
    df_remote.to_parquet(buf)
    parquet_bytes = buf.getvalue()

    class _FakeResponse:
        status_code = 200
        content = parquet_bytes
        def raise_for_status(self):
            pass

    with patch("requests.get", return_value=_FakeResponse()):
        df, issues = store.load(prefer="auto")

    assert len(df) == 2  # 1 row × 2 files
    assert any("bascule sur HuggingFace" in i for i in issues)


def test_load_hf_retries_on_failure(tmp_path):
    settings = _make_settings(tmp_path, hf_token=None)
    store = DataStore(settings)

    call_count = {"n": 0}

    def _flaky_get(*args, **kwargs):
        call_count["n"] += 1
        raise ConnectionError("unreachable")

    with patch("requests.get", side_effect=_flaky_get):
        df, issues = store.load(prefer="hf")

    assert df.empty
    # 2 files × 2 attempts each = 4 calls
    assert call_count["n"] == 4
    assert any("HF impossible" in i for i in issues)


def test_load_hf_warns_when_no_token(tmp_path):
    settings = _make_settings(tmp_path, hf_token=None)
    store = DataStore(settings)
    with patch("requests.get", side_effect=ConnectionError("x")):
        df, issues = store.load(prefer="hf")
    assert any("HF_TOKEN absent" in i for i in issues)


# ─── Metadata ───────────────────────────────────────────────────────────────


def test_metadata_summarizes_dataframe():
    df = pd.DataFrame({
        "_date": pd.to_datetime(["2020-01-01", "2021-06-15"]),
        "_channel": ["BFMTV", "TF1"],
    })
    store = DataStore(_make_settings(Path(".")))
    md = store.metadata(df)
    assert md["n_total"] == 2
    assert md["channels"] == ["BFMTV", "TF1"]
    assert md["date_min"] == "2020-01-01"
    assert md["date_max"] == "2021-06-15"


def test_metadata_handles_empty_dataframe():
    store = DataStore(_make_settings(Path(".")))
    md = store.metadata(pd.DataFrame())
    assert md["n_total"] == 0
    assert md["channels"] == []
    assert md["date_min"] is None
