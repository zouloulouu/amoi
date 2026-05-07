"""Process-wide tagging cache shared across requests."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import pandas as pd

CacheKey = Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]


class TaggingCache:
    """LRU cache keyed by (theme, concept_terms, up_terms, down_terms).

    Thread-safe. Entries are evicted when the maxsize is exceeded.
    Invalidating a theme drops every entry whose first key element matches.
    """

    def __init__(self, maxsize: int = 4):
        self._cache: "OrderedDict[CacheKey, pd.DataFrame]" = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get_or_compute(
        self,
        theme: str,
        concept: Tuple[str, ...],
        up: Tuple[str, ...],
        down: Tuple[str, ...],
        compute_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        key: CacheKey = (theme, concept, up, down)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        # Compute outside the lock so concurrent requests for OTHER themes
        # are not blocked by a long tagging operation.
        df = compute_fn()

        with self._lock:
            self._cache[key] = df
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
            return df

    def invalidate(self, theme: Optional[str] = None) -> int:
        """Drop cached entries for one theme (or all when theme is None).
        Returns the number of dropped entries.
        """
        with self._lock:
            if theme is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            keys_to_drop = [k for k in self._cache if k[0] == theme]
            for k in keys_to_drop:
                del self._cache[k]
            return len(keys_to_drop)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
