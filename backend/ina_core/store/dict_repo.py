"""Dictionary persistence: local JSON + optional HuggingFace mirror."""
from __future__ import annotations

import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Protocol

from ina_core.dictionaries import normalize_dictionaries_payload

logger = logging.getLogger("ina_core.store.dict")


class DictRepository(Protocol):
    def load(self) -> dict: ...
    def save(self, dictionaries: dict) -> None: ...


class ThemeAlreadyExists(Exception):
    """Raised by create_theme when the theme name is already present on disk."""


class ThemeNotFound(KeyError):
    """Raised when an operation targets a non-existing theme."""


class LocalJsonRepository:
    """Atomic local JSON file storage for the dictionaries.

    The fine-grained operations (`create_theme`, `update_theme`, `upsert_theme`,
    `delete_theme`, `rename_theme`) all do **read → modify → write under the
    same lock**, which prevents lost updates between two concurrent editors
    in the same process. Cross-process safety would require a file lock
    (e.g. via the `filelock` package) — out of scope here.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()

    # ─── Internal helpers (lock NOT acquired) ──────────────────────────────

    def _read_unlocked(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            text = self._path.read_text(encoding="utf-8").strip()
            if not text:
                return {}
            return normalize_dictionaries_payload(json.loads(text))
        except Exception as exc:
            logger.exception("Local dict load failed: %s", exc)
            return {}

    def _write_unlocked(self, dictionaries: dict) -> None:
        normalized = normalize_dictionaries_payload(dictionaries)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    # ─── Coarse API (compat) ───────────────────────────────────────────────

    def load(self) -> dict:
        with self._lock:
            return self._read_unlocked()

    def save(self, dictionaries: dict) -> None:
        """Replace the entire file. Use only for intentional bulk overwrites
        (e.g. reset-to-defaults). For single-theme changes prefer the
        fine-grained methods below to avoid lost updates.
        """
        with self._lock:
            self._write_unlocked(dictionaries)

    # ─── Fine-grained atomic operations ────────────────────────────────────

    def create_theme(self, name: str, theme: dict) -> dict:
        """Insert a new theme. Raises ThemeAlreadyExists if name is taken."""
        with self._lock:
            current = self._read_unlocked()
            if name in current:
                raise ThemeAlreadyExists(name)
            current[name] = theme
            self._write_unlocked(current)
            return current

    def update_theme(self, name: str, theme: dict) -> dict:
        """Replace an existing theme. Raises ThemeNotFound if absent."""
        with self._lock:
            current = self._read_unlocked()
            if name not in current:
                raise ThemeNotFound(name)
            current[name] = theme
            self._write_unlocked(current)
            return current

    def upsert_theme(self, name: str, theme: dict) -> dict:
        """Insert or replace, no error either way."""
        with self._lock:
            current = self._read_unlocked()
            current[name] = theme
            self._write_unlocked(current)
            return current

    def delete_theme(self, name: str) -> dict:
        """Remove a theme. Raises ThemeNotFound if absent."""
        with self._lock:
            current = self._read_unlocked()
            if name not in current:
                raise ThemeNotFound(name)
            del current[name]
            self._write_unlocked(current)
            return current

    def rename_theme(self, old: str, new: str) -> dict:
        """Rename a theme. Raises ThemeNotFound or ThemeAlreadyExists."""
        with self._lock:
            current = self._read_unlocked()
            if old not in current:
                raise ThemeNotFound(old)
            if new in current and new != old:
                raise ThemeAlreadyExists(new)
            current[new] = current.pop(old)
            self._write_unlocked(current)
            return current


class HuggingFaceRepository:
    """Read/write dictionaries.json on a HuggingFace dataset repo."""

    def __init__(self, repo_id: str, token: Optional[str], filename: str = "dictionaries.json"):
        self._repo_id = repo_id
        self._token = token
        self._filename = filename

    def load(self) -> dict:
        import requests

        url = f"https://huggingface.co/datasets/{self._repo_id}/resolve/main/{self._filename}"
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return normalize_dictionaries_payload(r.json())
        except Exception as exc:
            logger.warning("HF dict load failed: %s", exc)
        return {}

    def save(self, dictionaries: dict) -> None:
        if not self._token:
            return
        from huggingface_hub import HfApi

        normalized = normalize_dictionaries_payload(dictionaries)
        content = json.dumps(normalized, ensure_ascii=False, indent=2).encode("utf-8")
        try:
            HfApi().upload_file(
                path_or_fileobj=io.BytesIO(content),
                path_in_repo=self._filename,
                repo_id=self._repo_id,
                repo_type="dataset",
                token=self._token,
                commit_message="Update dictionaries",
            )
        except Exception as exc:
            logger.warning("HF dict save failed: %s", exc)
            raise


class CompositeDictRepository:
    """Local JSON as source of truth + optional async HF mirror.

    Reads come from local first; if empty, fall back to HF (bootstrap case).
    All write methods (save + the fine-grained ones) are synchronous on the
    local primary and mirrored to HF in a background daemon thread.
    """

    def __init__(
        self,
        primary: LocalJsonRepository,
        mirror: Optional[HuggingFaceRepository] = None,
    ):
        self._primary = primary
        self._mirror = mirror

    def load(self) -> dict:
        local = self._primary.load()
        if local:
            return local
        if self._mirror is not None:
            return self._mirror.load()
        return {}

    def save(self, dictionaries: dict) -> None:
        self._primary.save(dictionaries)
        self._mirror_async(dictionaries)

    def create_theme(self, name: str, theme: dict) -> dict:
        new_state = self._primary.create_theme(name, theme)
        self._mirror_async(new_state)
        return new_state

    def update_theme(self, name: str, theme: dict) -> dict:
        new_state = self._primary.update_theme(name, theme)
        self._mirror_async(new_state)
        return new_state

    def upsert_theme(self, name: str, theme: dict) -> dict:
        new_state = self._primary.upsert_theme(name, theme)
        self._mirror_async(new_state)
        return new_state

    def delete_theme(self, name: str) -> dict:
        new_state = self._primary.delete_theme(name)
        self._mirror_async(new_state)
        return new_state

    def rename_theme(self, old: str, new: str) -> dict:
        new_state = self._primary.rename_theme(old, new)
        self._mirror_async(new_state)
        return new_state

    def _mirror_async(self, state: dict) -> None:
        if self._mirror is None:
            return
        threading.Thread(
            target=self._safe_mirror_save,
            args=(state,),
            daemon=True,
        ).start()

    def _safe_mirror_save(self, dictionaries: dict) -> None:
        try:
            self._mirror.save(dictionaries)
        except Exception:
            logger.exception("Mirror HF dict save failed (non-blocking)")
