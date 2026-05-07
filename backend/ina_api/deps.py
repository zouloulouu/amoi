"""Dependency injection helpers: read app state from request.app.state."""
from __future__ import annotations

from typing import List

import pandas as pd
from fastapi import HTTPException, Request, status

from ina_core.store import CompositeDictRepository, DataStore, Settings
from ina_api.cache import TaggingCache


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_data_store(request: Request) -> DataStore:
    return request.app.state.data_store


def get_dict_repo(request: Request) -> CompositeDictRepository:
    return request.app.state.dict_repo


def get_cache(request: Request) -> TaggingCache:
    return request.app.state.cache


def get_df_base(request: Request) -> pd.DataFrame:
    df = getattr(request.app.state, "df_base", None)
    if df is None or df.empty:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Corpus not loaded — check /health for details",
        )
    return df


def get_load_issues(request: Request) -> List[str]:
    return list(getattr(request.app.state, "load_issues", []))


def get_load_source(request: Request) -> str:
    return getattr(request.app.state, "load_source", "unknown")
