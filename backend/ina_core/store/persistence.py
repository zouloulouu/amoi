"""On-disk persistence for cached corpus + tagged dataframes.

Two-layer cache to make boot near-instant:
- df_base.feather       cached deserialized corpus (rebuilt when snapshot version changes)
- tagged/{theme}_{hash}.parquet   cached per-theme tagged dataframes (invalidated when dictionary changes)

Both layers are best-effort: any I/O failure logs a warning and falls back
to recomputing from the source. The cache directory is meant to be
gitignored and may be safely deleted at any time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pyarrow.feather as feather

logger = logging.getLogger("ina_core.store.persistence")

CORPUS_FILENAME = "df_base.feather"
CORPUS_META_FILENAME = "df_base.meta.json"
TAGGED_DIR = "tagged"


def _theme_signature(
    theme: str,
    concept: Tuple[str, ...],
    up: Tuple[str, ...],
    down: Tuple[str, ...],
) -> str:
    """Stable short hash of the dictionary content for cache invalidation."""
    payload = json.dumps(
        {"theme": theme, "concept": list(concept), "up": list(up), "down": list(down)},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


class LocalDiskPersistence:
    """Stores df_base and tagged dataframes on the local filesystem.

    Thread-safe by design (each operation is a single OS call), but does NOT
    coordinate concurrent writes across processes — last-writer-wins on the
    binary file. Acceptable for our use case (rare writes, monotonic content).
    """

    def __init__(self, cache_dir: Path):
        self._dir = Path(cache_dir)
        self._tagged_dir = self._dir / TAGGED_DIR

    # ─── df_base ──────────────────────────────────────────────────────────

    def read_corpus(self, snapshot_token: str) -> Optional[pd.DataFrame]:
        """Return cached df_base if its snapshot_token matches, else None."""
        meta_path = self._dir / CORPUS_META_FILENAME
        data_path = self._dir / CORPUS_FILENAME
        if not meta_path.exists() or not data_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("snapshot_token") != snapshot_token:
                return None
            t0 = time.perf_counter()
            df = feather.read_feather(data_path)
            logger.info(
                "Corpus cache hit (%d rows in %.0f ms)",
                len(df), (time.perf_counter() - t0) * 1000,
            )
            return df
        except Exception as exc:
            logger.warning("Corpus cache read failed: %s", exc)
            return None

    def write_corpus(self, df: pd.DataFrame, snapshot_token: str) -> None:
        """Persist df_base + a tiny metadata file pinning the snapshot token."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp_data = self._dir / (CORPUS_FILENAME + ".tmp")
            tmp_meta = self._dir / (CORPUS_META_FILENAME + ".tmp")
            feather.write_feather(df, tmp_data, compression="lz4")
            tmp_meta.write_text(
                json.dumps({"snapshot_token": snapshot_token, "n_rows": int(len(df))}),
                encoding="utf-8",
            )
            tmp_data.replace(self._dir / CORPUS_FILENAME)
            tmp_meta.replace(self._dir / CORPUS_META_FILENAME)
            logger.info("Corpus cache written (%d rows)", len(df))
        except Exception as exc:
            logger.warning("Corpus cache write failed: %s", exc)

    # ─── df_tagged per theme ──────────────────────────────────────────────

    def _tagged_path(
        self,
        theme: str,
        concept: Tuple[str, ...],
        up: Tuple[str, ...],
        down: Tuple[str, ...],
    ) -> Path:
        sig = _theme_signature(theme, concept, up, down)
        # Slugify the theme name minimally — the hash guarantees uniqueness anyway.
        safe_theme = "".join(c if c.isalnum() or c in "_-" else "_" for c in theme)
        return self._tagged_dir / f"{safe_theme}_{sig}.parquet"

    def read_tagged(
        self,
        theme: str,
        concept: Tuple[str, ...],
        up: Tuple[str, ...],
        down: Tuple[str, ...],
    ) -> Optional[pd.DataFrame]:
        path = self._tagged_path(theme, concept, up, down)
        if not path.exists():
            return None
        try:
            t0 = time.perf_counter()
            df = pd.read_parquet(path)
            logger.info(
                "Tagged cache hit for %r (%d rows in %.0f ms)",
                theme, len(df), (time.perf_counter() - t0) * 1000,
            )
            return df
        except Exception as exc:
            logger.warning("Tagged cache read failed for %r: %s", theme, exc)
            return None

    def write_tagged(
        self,
        df: pd.DataFrame,
        theme: str,
        concept: Tuple[str, ...],
        up: Tuple[str, ...],
        down: Tuple[str, ...],
    ) -> None:
        try:
            self._tagged_dir.mkdir(parents=True, exist_ok=True)
            path = self._tagged_path(theme, concept, up, down)
            tmp = path.with_suffix(path.suffix + ".tmp")
            df.to_parquet(tmp, compression="snappy")
            tmp.replace(path)
            logger.info("Tagged cache written for %r (%d rows)", theme, len(df))
        except Exception as exc:
            logger.warning("Tagged cache write failed for %r: %s", theme, exc)

    def invalidate_theme(self, theme: str) -> int:
        """Remove all cached tagged files for the given theme. Returns count removed."""
        if not self._tagged_dir.exists():
            return 0
        safe_theme = "".join(c if c.isalnum() or c in "_-" else "_" for c in theme)
        prefix = f"{safe_theme}_"
        removed = 0
        for path in self._tagged_dir.glob(f"{prefix}*.parquet"):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", path, exc)
        return removed
