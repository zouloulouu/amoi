"""GET /channels — coverage per channel + decade distribution."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from ina_core import build_channel_stats, build_decade_distribution
from ina_api.deps import get_df_base
from ina_api.schemas import ChannelCoverage, DecadePoint

router = APIRouter(tags=["coverage"])


@router.get("/channels", response_model=list[ChannelCoverage])
def channels(df: pd.DataFrame = Depends(get_df_base)):
    stats = build_channel_stats(df)
    return [
        ChannelCoverage(
            channel=row["_channel"],
            date_min=row["date_min"].date(),
            date_max=row["date_max"].date(),
            n_obs=int(row["n_obs"]),
            share_pct=float(row["share_pct"]),
        )
        for _, row in stats.iterrows()
    ]


@router.get("/coverage", response_model=list[DecadePoint])
def coverage(df: pd.DataFrame = Depends(get_df_base)):
    out = build_decade_distribution(df)
    return [
        DecadePoint(
            channel=row["_channel"],
            decade=int(row["decade"]),
            decade_label=row["decade_label"],
            n=int(row["n"]),
            pct=float(row["pct"]),
        )
        for _, row in out.iterrows()
    ]
