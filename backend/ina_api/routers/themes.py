"""CRUD for theme dictionaries."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ina_core.store import (
    CompositeDictRepository,
    ThemeAlreadyExists,
    ThemeNotFound,
)
from ina_core.cache import TaggingCache
from ina_api.deps import get_cache, get_dict_repo
from ina_api.schemas import (
    ThemeCreateRequest,
    ThemeDictionary,
    ThemeSummary,
    ThemeUpdateRequest,
)

router = APIRouter(prefix="/themes", tags=["themes"])


@router.get("", response_model=list[ThemeSummary])
def list_themes(repo: CompositeDictRepository = Depends(get_dict_repo)):
    dictionaries = repo.load()
    return [
        ThemeSummary(
            name=name,
            n_concept=len(theme.get("concept", [])),
            n_up=len(theme.get("up", [])),
            n_down=len(theme.get("down", [])),
        )
        for name, theme in sorted(dictionaries.items())
    ]


@router.get("/{name}", response_model=ThemeDictionary)
def get_theme(name: str, repo: CompositeDictRepository = Depends(get_dict_repo)):
    dictionaries = repo.load()
    if name not in dictionaries:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Theme {name!r} not found")
    return ThemeDictionary(name=name, **dictionaries[name])


@router.post("", response_model=ThemeDictionary, status_code=status.HTTP_201_CREATED)
def create_theme(
    payload: ThemeCreateRequest,
    repo: CompositeDictRepository = Depends(get_dict_repo),
    cache: TaggingCache = Depends(get_cache),
):
    new_theme = {
        "concept": payload.concept,
        "context": [],
        "up": payload.up,
        "down": payload.down,
    }
    try:
        repo.create_theme(payload.name, new_theme)
    except ThemeAlreadyExists:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Theme {payload.name!r} already exists"
        )
    cache.invalidate(payload.name)
    return ThemeDictionary(name=payload.name, **new_theme)


@router.put("/{name}", response_model=ThemeDictionary)
def update_theme(
    name: str,
    payload: ThemeUpdateRequest,
    repo: CompositeDictRepository = Depends(get_dict_repo),
    cache: TaggingCache = Depends(get_cache),
):
    # We need the current theme to apply partial updates; re-read from disk.
    dictionaries = repo.load()
    if name not in dictionaries:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Theme {name!r} not found")
    current = dict(dictionaries[name])
    if payload.concept is not None:
        current["concept"] = payload.concept
    if payload.up is not None:
        current["up"] = payload.up
    if payload.down is not None:
        current["down"] = payload.down
    try:
        repo.update_theme(name, current)
    except ThemeNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Theme {name!r} not found")
    cache.invalidate(name)
    return ThemeDictionary(name=name, **current)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_theme(
    name: str,
    repo: CompositeDictRepository = Depends(get_dict_repo),
    cache: TaggingCache = Depends(get_cache),
):
    dictionaries = repo.load()
    if name not in dictionaries:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Theme {name!r} not found")
    if len(dictionaries) <= 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Cannot delete the last theme"
        )
    try:
        repo.delete_theme(name)
    except ThemeNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Theme {name!r} not found")
    cache.invalidate(name)
