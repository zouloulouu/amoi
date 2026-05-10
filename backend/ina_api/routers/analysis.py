"""POST /analysis — full thematic analysis on the corpus."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from ina_core import (
    DIRECTION_AMBIGUOUS,
    DIRECTION_DOWN,
    DIRECTION_FLAT,
    DIRECTION_UP,
    aggregate_by_period,
    build_descriptive_table,
    build_top_channels,
    prepare_keywords,
    tag_dataframe,
)
from ina_core.store import CompositeDictRepository
from ina_core.cache import TaggingCache
from ina_api.deps import get_cache, get_df_base, get_dict_repo
from ina_api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    DescriptiveRow,
    FREQUENCY_LABEL,
    KpiResponse,
    PreviewTitle,
    TimeSeriesPoint,
    TopChannel,
)

router = APIRouter(tags=["analysis"])


DIRECTION_LABELS = {
    DIRECTION_FLAT: "flat",
    DIRECTION_UP: "up",
    DIRECTION_DOWN: "down",
    DIRECTION_AMBIGUOUS: "ambiguous",
}


@router.post("/analysis", response_model=AnalysisResponse)
def analyze(
    payload: AnalysisRequest,
    df_base: pd.DataFrame = Depends(get_df_base),
    repo: CompositeDictRepository = Depends(get_dict_repo),
    cache: TaggingCache = Depends(get_cache),
):
    dictionaries = repo.load()
    if payload.theme not in dictionaries:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Theme {payload.theme!r} not found"
        )
    theme = dictionaries[payload.theme]

    concept_norm = prepare_keywords(theme.get("concept", []))
    up_norm = prepare_keywords(theme.get("up", []))
    down_norm = prepare_keywords(theme.get("down", []))

    if not concept_norm:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Theme {payload.theme!r} has no concept keywords",
        )

    # Tag once per (theme, dictionary signature) — cached process-wide.
    df_tagged = cache.get_or_compute(
        theme=payload.theme,
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

    # Apply filters
    mask = pd.Series(True, index=df_tagged.index)
    if payload.date_start is not None:
        mask &= df_tagged["_date"] >= pd.Timestamp(payload.date_start)
    if payload.date_end is not None:
        mask &= df_tagged["_date"] <= pd.Timestamp(payload.date_end)
    if payload.channels:
        mask &= df_tagged["_channel"].isin(payload.channels)

    df_filtered = df_tagged[mask]
    if df_filtered.empty:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No rows match the filters"
        )

    # Aggregate
    freq_label = FREQUENCY_LABEL[payload.frequency]
    stats = aggregate_by_period(df_filtered, frequency=freq_label)
    desc = build_descriptive_table(stats, df_filtered)
    top = build_top_channels(df_filtered)

    n_matched = int(df_filtered["is_match"].sum())
    occ_concept_total = int(
        df_filtered.loc[df_filtered["is_match"] == 1, "occ_concept"].sum()
    )
    freq_mean = float(stats["frequency"].mean()) if len(stats) else 0.0

    kpi = KpiResponse(
        n_total=int(len(df_filtered)),
        n_matched=n_matched,
        occ_concept_total=occ_concept_total,
        freq_mean=freq_mean,
        net_signal_total=int(stats["net_signal"].sum()),
        up_total=int(stats["up_titles"].sum()),
        down_total=int(stats["down_titles"].sum()),
        ambiguous_total=int(stats["ambiguous_titles"].sum()),
    )

    series = [
        TimeSeriesPoint(
            period_start=row["period_start"].date(),
            total_titles=int(row["total_titles"]),
            matched_titles=int(row["matched_titles"]),
            occurrences_concept=int(row["occurrences_concept"]),
            frequency=float(row["frequency"]) if pd.notna(row["frequency"]) else 0.0,
            up_titles=int(row["up_titles"]),
            down_titles=int(row["down_titles"]),
            ambiguous_titles=int(row["ambiguous_titles"]),
            net_signal=int(row["net_signal"]),
        )
        for _, row in stats.iterrows()
    ]

    top_channels = [
        TopChannel(
            channel=row["_channel"],
            total_titles=int(row["total_titles"]),
            matched_titles=int(row["matched_titles"]),
            frequency=float(row["frequency"]) if pd.notna(row["frequency"]) else 0.0,
            up_titles=int(row["up_titles"]),
            down_titles=int(row["down_titles"]),
            ambiguous_titles=int(row["ambiguous_titles"]),
            net_signal=int(row["net_signal"]),
        )
        for _, row in top.iterrows()
    ]

    descriptive = [
        DescriptiveRow(indicateur=row["indicateur"], valeur=row["valeur"])
        for _, row in desc.iterrows()
    ]

    preview_cols = [
        c
        for c in [
            "_date",
            "_channel",
            "title",
            "occ_concept",
            "occ_up",
            "occ_down",
            "direction",
            "source_file",
        ]
        if c in df_filtered.columns
    ]
    preview_df = (
        df_filtered[df_filtered["is_match"] == 1]
        .sort_values(["occ_concept", "_date"], ascending=[False, False])
        .head(300)[preview_cols]
        .copy()
    )
    preview_titles = [
        PreviewTitle(
            date=row["_date"].date(),
            channel=row.get("_channel"),
            title=str(row.get("title", "")),
            occ_concept=int(row.get("occ_concept", 0)),
            occ_up=int(row.get("occ_up", 0)),
            occ_down=int(row.get("occ_down", 0)),
            direction=int(row.get("direction", DIRECTION_FLAT)),
            direction_label=DIRECTION_LABELS.get(
                int(row.get("direction", DIRECTION_FLAT)), "flat"
            ),
            source_file=row.get("source_file"),
        )
        for _, row in preview_df.iterrows()
    ]

    return AnalysisResponse(
        kpi=kpi,
        series=series,
        top_channels=top_channels,
        descriptive=descriptive,
        preview_titles=preview_titles,
        request=payload,
    )
