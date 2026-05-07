"""GET /metadata — corpus summary."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from ina_api.deps import get_df_base, get_load_source
from ina_api.schemas import MetadataResponse

router = APIRouter(tags=["meta"])


@router.get("/metadata", response_model=MetadataResponse)
def metadata(
    df: pd.DataFrame = Depends(get_df_base),
    source: str = Depends(get_load_source),
):
    return MetadataResponse(
        n_total=int(len(df)),
        date_min=df["_date"].min().date(),
        date_max=df["_date"].max().date(),
        channels=sorted(c for c in df["_channel"].dropna().unique().tolist()),
        source=source if source in ("local", "hf") else "unknown",
    )
