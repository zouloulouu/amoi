"""FastAPI application entry point."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ina_core.store import (
    DEFAULT_HF_PARQUET_FILES,
    CompositeDictRepository,
    DataStore,
    HuggingFaceRepository,
    LocalDiskPersistence,
    LocalJsonRepository,
    Settings,
)
from ina_core.cache import TaggingCache
from ina_core.prewarm import prewarm_themes_async
from ina_api.routers import analysis, channels, export, health, metadata, themes


logger = logging.getLogger("ina_api")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer env var with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default=%d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("Invalid %s=%r, using minimum=%d", name, raw, minimum)
        return minimum
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_load_source(issues: list[str]) -> str:
    """Detect whether the corpus came from local snapshot or HF, based on issues."""
    if any("Snapshot local introuvable" in i for i in issues):
        return "hf"
    if any("bascule sur HuggingFace" in i for i in issues):
        return "hf"
    return "local"


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(os.environ.get("INA_PROJECT_ROOT", Path.cwd()))
    settings = Settings(
        hf_repo_id=os.environ.get("INA_HF_REPO_ID", "zouloulouu/data_ina_clean"),
        hf_parquet_files=DEFAULT_HF_PARQUET_FILES,
        hf_token=os.environ.get("HF_TOKEN"),
        project_root=project_root,
    )

    persistence = LocalDiskPersistence(settings.cache_dir)
    data_store = DataStore(settings, persistence=persistence)
    dict_repo = CompositeDictRepository(
        primary=LocalJsonRepository(settings.dictionary_path),
        mirror=(
            HuggingFaceRepository(settings.hf_repo_id, settings.hf_token)
            if settings.hf_token else None
        ),
    )

    app.state.settings = settings
    app.state.data_store = data_store
    app.state.dict_repo = dict_repo
    cache_maxsize = _env_int("INA_TAGGING_CACHE_MAXSIZE", 4)
    disable_prewarm = _env_bool("INA_DISABLE_PREWARM", default=False)
    app.state.cache = TaggingCache(maxsize=cache_maxsize, persistence=persistence)
    logger.info(
        "Tagging cache configured: maxsize=%d, prewarm=%s",
        cache_maxsize,
        "disabled" if disable_prewarm else "enabled",
    )

    logger.info("Loading corpus...")
    df, issues = data_store.load(prefer="auto")
    app.state.df_base = df
    app.state.load_issues = issues
    app.state.load_source = _resolve_load_source(issues)
    logger.info(
        "Corpus loaded: %d rows, source=%s, %d issue(s)",
        len(df), app.state.load_source, len(issues),
    )

    # Background prewarming: tag every known theme so the first /analysis
    # call is a cache hit (in-memory or disk). Daemon thread → never blocks
    # shutdown.
    if not disable_prewarm and not df.empty:
        dictionaries = dict_repo.load()
        if dictionaries:
            logger.info("Launching background prewarm for %d theme(s)...", len(dictionaries))
            prewarm_themes_async(app.state.cache, df, dictionaries)

    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="data_ina API",
        version="0.1.0",
        description="Thematic analysis of INA TV/radio news titles.",
        lifespan=lifespan,
    )

    # CORS for local dev (frontend on a different port)
    default_cors_origins = ",".join(
        [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
        ]
    )
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("INA_CORS_ORIGINS", default_cors_origins).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(metadata.router)
    app.include_router(themes.router)
    app.include_router(channels.router)
    app.include_router(analysis.router)
    app.include_router(export.router)
    return app


app = create_app()
