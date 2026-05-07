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


class LocalJsonRepository:
    """Atomic local JSON file storage for the dictionaries.

    Uses tmp+os.replace for atomic writes. A threading.Lock protects
    against concurrent writes within a single process.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> dict:
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

    def save(self, dictionaries: dict) -> None:
        normalized = normalize_dictionaries_payload(dictionaries)
        tmp = self._path.with_name(self._path.name + ".tmp")
        with self._lock:
            tmp.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)


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
    Writes go synchronously to local and are mirrored to HF in a background
    daemon thread — failures are logged but do not block the caller.
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
        # Source of truth — must be synchronous and atomic.
        self._primary.save(dictionaries)
        # Best-effort mirror.
        if self._mirror is not None:
            threading.Thread(
                target=self._safe_mirror_save,
                args=(dictionaries,),
                daemon=True,
            ).start()

    def _safe_mirror_save(self, dictionaries: dict) -> None:
        try:
            self._mirror.save(dictionaries)
        except Exception:
            logger.exception("Mirror HF dict save failed (non-blocking)")
