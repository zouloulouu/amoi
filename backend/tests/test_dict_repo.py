"""Tests for the dictionary repositories."""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import pytest

from ina_core.store.dict_repo import (
    CompositeDictRepository,
    HuggingFaceRepository,
    LocalJsonRepository,
    ThemeAlreadyExists,
    ThemeNotFound,
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


# ─── LocalJsonRepository — fine-grained atomic operations ──────────────────


def test_create_theme_inserts(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    repo.create_theme("inflation", {"concept": ["ipc"]})
    assert "inflation" in repo.load()


def test_create_theme_raises_when_exists(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    repo.create_theme("inflation", {"concept": ["ipc"]})
    with pytest.raises(ThemeAlreadyExists):
        repo.create_theme("inflation", {"concept": ["x"]})


def test_update_theme_replaces(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    repo.create_theme("inflation", {"concept": ["ipc"]})
    repo.update_theme("inflation", {"concept": ["new"]})
    assert repo.load()["inflation"]["concept"] == ["new"]


def test_update_theme_raises_when_absent(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    with pytest.raises(ThemeNotFound):
        repo.update_theme("ghost", {"concept": ["x"]})


def test_upsert_theme_inserts_or_replaces(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    repo.upsert_theme("a", {"concept": ["x"]})  # insert
    repo.upsert_theme("a", {"concept": ["y"]})  # replace
    assert repo.load()["a"]["concept"] == ["y"]


def test_delete_theme_removes(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    repo.create_theme("a", {"concept": ["x"]})
    repo.delete_theme("a")
    assert "a" not in repo.load()


def test_delete_theme_raises_when_absent(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    with pytest.raises(ThemeNotFound):
        repo.delete_theme("ghost")


def test_rename_theme(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    repo.create_theme("old", {"concept": ["x"]})
    repo.rename_theme("old", "new")
    state = repo.load()
    assert "old" not in state and "new" in state


def test_rename_theme_raises_when_target_exists(tmp_path):
    repo = LocalJsonRepository(tmp_path / "dict.json")
    repo.create_theme("a", {"concept": ["x"]})
    repo.create_theme("b", {"concept": ["y"]})
    with pytest.raises(ThemeAlreadyExists):
        repo.rename_theme("a", "b")


def test_concurrent_creates_dont_lose_data(tmp_path):
    """Two threads creating different themes must both succeed.

    This is the core test: simulates Alice and Bob creating themes at the
    same time. Without atomic read-modify-write, one would overwrite the
    other (lost update). With it, both end up on disk.
    """
    repo = LocalJsonRepository(tmp_path / "dict.json")
    barrier = threading.Barrier(2)

    def _create(name):
        barrier.wait()  # both threads enter at the same time
        repo.create_theme(name, {"concept": [name]})

    t1 = threading.Thread(target=_create, args=("alice_theme",))
    t2 = threading.Thread(target=_create, args=("bob_theme",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    final = repo.load()
    assert "alice_theme" in final
    assert "bob_theme" in final


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
