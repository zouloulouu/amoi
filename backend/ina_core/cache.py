"""Process-wide tagging cache shared across requests, with optional disk persistence."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

import pandas as pd

CacheKey = Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]


class TaggingCache:
    """Two-level LRU cache keyed by (theme, concept_terms, up_terms, down_terms).

    Level 1: in-memory OrderedDict (LRU, maxsize entries).
    Level 2 (optional): on-disk via a `LocalDiskPersistence` instance.

    On a miss, we look on disk first; if found, we hydrate the in-memory
    layer. Otherwise we compute, then write to BOTH layers. Invalidating
    a theme drops both layers.

    Thread-safe. The compute step happens outside the lock so concurrent
    requests for OTHER themes are not blocked by a long tagging operation.
    """

    def __init__(self, maxsize: int = 4, persistence: Optional[Any] = None):
        self._cache: "OrderedDict[CacheKey, pd.DataFrame]" = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        # Optional: a LocalDiskPersistence-like object exposing
        # read_tagged / write_tagged / invalidate_theme. Typed as Any to
        # avoid an import cycle with ina_core.store.persistence.
        self._persistence = persistence

    def _put_in_memory(self, key: CacheKey, df: pd.DataFrame) -> None:
        """Insert into the in-memory layer with LRU eviction. Caller holds the lock."""
        self._cache[key] = df
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def get_or_compute(
        self,
        theme: str,
        concept: Tuple[str, ...],
        up: Tuple[str, ...],
        down: Tuple[str, ...],
        compute_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        key: CacheKey = (theme, concept, up, down)

        # Level 1: in-memory hit
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        # Level 2: disk hit
        if self._persistence is not None:
            disk_df = self._persistence.read_tagged(theme, concept, up, down)
            if disk_df is not None and not disk_df.empty:
                with self._lock:
                    self._put_in_memory(key, disk_df)
                return disk_df

        # Miss on both levels: compute outside the lock.
        df = compute_fn()

        # Write to both layers.
        with self._lock:
            self._put_in_memory(key, df)
        if self._persistence is not None:
            self._persistence.write_tagged(df, theme, concept, up, down)

        return df

    def invalidate(self, theme: Optional[str] = None) -> int:
        """Drop cached entries for one theme (or all when theme is None).

        Drops BOTH the in-memory layer and the on-disk layer (when present).
        Returns the number of in-memory entries dropped.
        """
        with self._lock:
            if theme is None:
                count = len(self._cache)
                self._cache.clear()
            else:
                keys_to_drop = [k for k in self._cache if k[0] == theme]
                for k in keys_to_drop:
                    del self._cache[k]
                count = len(keys_to_drop)

        if self._persistence is not None and theme is not None:
            try:
                self._persistence.invalidate_theme(theme)
            except Exception:
                pass

        return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
