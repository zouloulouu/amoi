"""Tests for the dictionary repositories."""
from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from ina_core.store.dict_repo import (
    CompositeDictRepository,
    HuggingFaceRepository,
    LocalJsonRepository,
)


# ─── LocalJsonRepository ────────────────────────────────────────────────────


def test_local_repo_load_returns_empty_when_file_missing(tmp_path):
    repo = LocalJsonRepository(tmp_path / "missing.json")
    assert repo.load() == {}


def test_local_repo_load_returns_empty_for_empty_file(tmp_path):
    p = tmp_path / "dict.json"
    p.write_text("", encoding="utf-8")
    assert LocalJsonRepository(p).load() == {}


def test_local_repo_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "dict.json"
    repo = LocalJsonRepository(p)
    payload = {"inflation": {"concept": ["ipc"], "context": [], "up": [], "down": []}}
    repo.save(payload)
    assert repo.load() == payload


def test_local_repo_save_normalizes_payload(tmp_path):
    """Save passes through normalize_dictionaries_payload (drops invalid keys)."""
    p = tmp_path / "dict.json"
    repo = LocalJsonRepository(p)
    raw = {
        "valid": {"concept": ["a"]},
        "": {"concept": ["dropped"]},
    }
    repo.save(raw)
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert "valid" in on_disk
    assert "" not in on_disk


def test_local_repo_save_is_atomic(tmp_path):
    """No .tmp file should remain after a successful save."""
    p = tmp_path / "dict.json"
    repo = LocalJsonRepository(p)
    repo.save({"theme": {"concept": ["x"]}})
    assert p.exists()
    assert not (tmp_path / "dict.json.tmp").exists()


# ─── HuggingFaceRepository ──────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_hf_repo_load_returns_empty_when_no_token_and_404():
    repo = HuggingFaceRepository(repo_id="fake/repo", token=None)
    with patch("requests.get", return_value=_FakeResponse(404, {})):
        assert repo.load() == {}


def test_hf_repo_load_parses_200(tmp_path):
    repo = HuggingFaceRepository(repo_id="fake/repo", token="hf_xxx")
    payload = {"inflation": {"concept": ["ipc"]}}
    with patch("requests.get", return_value=_FakeResponse(200, payload)):
        out = repo.load()
    assert "inflation" in out
    # Normalization fills in the 4 buckets:
    assert set(out["inflation"].keys()) == {"concept", "context", "up", "down"}


def test_hf_repo_load_swallows_network_errors():
    repo = HuggingFaceRepository(repo_id="fake/repo", token="hf_xxx")
    with patch("requests.get", side_effect=ConnectionError("unreachable")):
        assert repo.load() == {}


def test_hf_repo_save_is_noop_without_token():
    """No token → save returns silently (no HfApi call)."""
    repo = HuggingFaceRepository(repo_id="fake/repo", token=None)
    # If this tried to import huggingface_hub or call HfApi, it would fail
    repo.save({"theme": {"concept": ["x"]}})


# ─── CompositeDictRepository ────────────────────────────────────────────────


def test_composite_load_uses_local_when_present(tmp_path):
    local_path = tmp_path / "dict.json"
    local = LocalJsonRepository(local_path)
    local.save({"inflation": {"concept": ["ipc"]}})

    class _ExplodingMirror:
        def load(self):
            raise AssertionError("Mirror.load should NOT be called when local is non-empty")

    repo = CompositeDictRepository(primary=local, mirror=_ExplodingMirror())
    assert "inflation" in repo.load()


def test_composite_load_falls_back_to_mirror_when_local_empty(tmp_path):
    local = LocalJsonRepository(tmp_path / "missing.json")

    class _StubMirror:
        def load(self):
            return {"chomage": {"concept": ["unedic"], "context": [], "up": [], "down": []}}

    repo = CompositeDictRepository(primary=local, mirror=_StubMirror())
    assert "chomage" in repo.load()


def test_composite_save_writes_local_synchronously(tmp_path):
    local = LocalJsonRepository(tmp_path / "dict.json")
    repo = CompositeDictRepository(primary=local, mirror=None)
    repo.save({"theme": {"concept": ["x"]}})
    # Reading immediately after save must see the data.
    assert "theme" in local.load()


def test_composite_save_mirror_runs_in_background(tmp_path):
    local = LocalJsonRepository(tmp_path / "dict.json")

    mirror_calls = []

    class _SlowMirror:
        def save(self, dictionaries):
            time.sleep(0.05)
            mirror_calls.append(dictionaries)

    repo = CompositeDictRepository(primary=local, mirror=_SlowMirror())

    # Save should return immediately even though mirror is "slow".
    t0 = time.perf_counter()
    repo.save({"theme": {"concept": ["x"]}})
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.04, f"save was blocking: {elapsed:.3f}s"

    # Wait for the background thread to complete.
    deadline = time.time() + 1.0
    while time.time() < deadline and not mirror_calls:
        time.sleep(0.01)
    assert mirror_calls, "mirror.save was never called"


def test_composite_save_swallows_mirror_failures(tmp_path):
    local = LocalJsonRepository(tmp_path / "dict.json")

    class _FailingMirror:
        def save(self, dictionaries):
            raise RuntimeError("HF down")

    repo = CompositeDictRepository(primary=local, mirror=_FailingMirror())
    # Must not raise — local save succeeded, mirror is best-effort.
    repo.save({"theme": {"concept": ["x"]}})
    # Local must still have the data.
    assert "theme" in local.load()
