"""GET /export/csv — streaming CSV of an analysis."""
from __future__ import annotations

import csv
import io
from datetime import date
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from ina_core import (
    aggregate_by_period,
    prepare_keywords,
    tag_dataframe,
)
from ina_core.store import CompositeDictRepository
from ina_api.cache import TaggingCache
from ina_api.deps import get_cache, get_df_base, get_dict_repo
from ina_api.schemas import FREQUENCY_LABEL, Frequency

router = APIRouter(tags=["export"])


def _format_period_date(ts: pd.Timestamp, frequency: Frequency) -> str:
    if frequency == Frequency.quarterly:
        return f"{ts.year}-Q{ts.quarter}"
    if frequency == Frequency.yearly:
        return str(ts.year)
    return ts.strftime("%Y-%m")


@router.get("/export/csv")
def export_csv(
    theme: str,
    frequency: Frequency = Frequency.monthly,
    date_start: Optional[date] = Query(None),
    date_end: Optional[date] = Query(None),
    channels: Optional[list[str]] = Query(None),
    separator: str = Query(";", min_length=1, max_length=1),
    decimal: str = Query(",", min_length=1, max_length=1),
    df_base: pd.DataFrame = Depends(get_df_base),
    repo: CompositeDictRepository = Depends(get_dict_repo),
    cache: TaggingCache = Depends(get_cache),
):
    dictionaries = repo.load()
    if theme not in dictionaries:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Theme {theme!r} not found")

    theme_dict = dictionaries[theme]
    concept_norm = prepare_keywords(theme_dict.get("concept", []))
    up_norm = prepare_keywords(theme_dict.get("up", []))
    down_norm = prepare_keywords(theme_dict.get("down", []))
    if not concept_norm:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Theme {theme!r} has no concept keywords",
        )

    df_tagged = cache.get_or_compute(
        theme=theme,
        concept=tuple(concept_norm),
        up=tuple(up_norm),
        down=tuple(down_norm),
        compute_fn=lambda: tag_dataframe(
            df_base,
            title_col="title",
            concept_norm=concept_norm,
            up_norm=up_norm,
            down_norm=down_norm,
            title_norm_col="_title_norm",
        ),
    )

    mask = pd.Series(True, index=df_tagged.index)
    if date_start is not None:
        mask &= df_tagged["_date"] >= pd.Timestamp(date_start)
    if date_end is not None:
        mask &= df_tagged["_date"] <= pd.Timestamp(date_end)
    if channels:
        mask &= df_tagged["_channel"].isin(channels)

    df_filtered = df_tagged[mask]
    if df_filtered.empty:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No rows match the filters")

    stats = aggregate_by_period(df_filtered, frequency=FREQUENCY_LABEL[frequency])

    # Build the export rows (matches the old Streamlit export schema)
    def _format_decimal(value: float) -> str:
        return f"{value:.4f}".replace(".", decimal) if decimal != "." else f"{value:.4f}"

    def _row_iter():
        header = [
            "date", "total_titres", "volume", "frequence",
            "signal_up", "signal_down", "signal_net",
        ]
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=separator)
        writer.writerow(header)
        yield buf.getvalue()
        buf.seek(0); buf.truncate()

        for _, row in stats.iterrows():
            writer.writerow([
                _format_period_date(row["period_start"], frequency),
                int(row["total_titles"]),
                int(row["matched_titles"]),
                _format_decimal(float(row["frequency"])) if pd.notna(row["frequency"]) else "",
                int(row["up_titles"]),
                int(row["down_titles"]),
                int(row["net_signal"]),
            ])
            yield buf.getvalue()
            buf.seek(0); buf.truncate()

    fname_parts = [theme, frequency.value]
    if date_start is not None:
        fname_parts.append(date_start.strftime("%Y%m%d"))
    if date_end is not None:
        fname_parts.append(date_end.strftime("%Y%m%d"))
    filename = "_".join(fname_parts) + ".csv"

    return StreamingResponse(
        _row_iter(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
