"""Background prewarming of the tagging cache for all known themes.

Goal: make the first /analysis call for every theme cheap (in-memory hit
or disk hit), even on a cold process. Runs in a daemon thread so the
caller (FastAPI lifespan or Streamlit boot) is not blocked.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable

import pandas as pd

from ina_core.cache import TaggingCache
from ina_core.tagging import prepare_keywords, tag_dataframe

logger = logging.getLogger("ina_core.prewarm")


def _warm_one_theme(
    cache: TaggingCache,
    df_base: pd.DataFrame,
    theme_name: str,
    theme_payload: dict,
    title_col: str = "title",
    title_norm_col: str = "_title_norm",
) -> None:
    concept_norm = prepare_keywords(theme_payload.get("concept", []))
    up_norm = prepare_keywords(theme_payload.get("up", []))
    down_norm = prepare_keywords(theme_payload.get("down", []))
    if not concept_norm:
        # No concept keywords → nothing to tag for this theme.
        return

    cache.get_or_compute(
        theme=theme_name,
        concept=tuple(concept_norm),
        up=tuple(up_norm),
        down=tuple(down_norm),
        compute_fn=lambda: tag_dataframe(
            df_base,
            title_col=title_col,
            concept_norm=concept_norm,
            up_norm=up_norm,
            down_norm=down_norm,
            title_norm_col=title_norm_col,
        ),
    )


def prewarm_themes_blocking(
    cache: TaggingCache,
    df_base: pd.DataFrame,
    dictionaries: dict,
    title_col: str = "title",
    title_norm_col: str = "_title_norm",
) -> None:
    """Tag every theme present in `dictionaries` and populate the cache.
    Synchronous variant — caller blocks until done.
    """
    if df_base.empty or not dictionaries:
        return
    t0 = time.perf_counter()
    for name, payload in dictionaries.items():
        try:
            _warm_one_theme(cache, df_base, name, payload, title_col, title_norm_col)
        except Exception as exc:
            logger.warning("Prewarm failed for theme %r: %s", name, exc)
    logger.info(
        "Prewarmed %d theme(s) in %.1f s",
        len(dictionaries), time.perf_counter() - t0,
    )


def prewarm_themes_async(
    cache: TaggingCache,
    df_base: pd.DataFrame,
    dictionaries: dict,
    title_col: str = "title",
    title_norm_col: str = "_title_norm",
) -> threading.Thread:
    """Same as `prewarm_themes_blocking` but launched in a daemon thread.

    Returns the Thread so the caller can monitor or .join() if desired.
    The thread is `daemon=True` so it never blocks process shutdown.
    """
    t = threading.Thread(
        target=prewarm_themes_blocking,
        args=(cache, df_base, dictionaries, title_col, title_norm_col),
        daemon=True,
        name="ina-prewarm",
    )
    t.start()
    return t
