"""GET /health — liveness + corpus state."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, Request

from ina_api.deps import get_load_issues, get_load_source
from ina_api.schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    issues: list[str] = Depends(get_load_issues),
    source: str = Depends(get_load_source),
):
    df: pd.DataFrame = getattr(request.app.state, "df_base", pd.DataFrame())
    loaded = not df.empty
    return HealthResponse(
        status="ok" if loaded else "degraded",
        data_loaded=loaded,
        n_rows=int(len(df)) if loaded else 0,
        source=source if source in ("local", "hf") else "unknown",
        issues=issues,
    )
