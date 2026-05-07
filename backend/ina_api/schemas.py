"""Pydantic models for API requests and responses."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ─── Health & Metadata ──────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    data_loaded: bool
    n_rows: int
    source: Literal["local", "hf", "unknown"]
    issues: List[str] = Field(default_factory=list)


class MetadataResponse(BaseModel):
    n_total: int
    date_min: Optional[date]
    date_max: Optional[date]
    channels: List[str]
    source: Literal["local", "hf", "unknown"]


# ─── Themes ─────────────────────────────────────────────────────────────────


class ThemeDictionary(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    concept: List[str] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)
    up: List[str] = Field(default_factory=list)
    down: List[str] = Field(default_factory=list)


class ThemeSummary(BaseModel):
    name: str
    n_concept: int
    n_up: int
    n_down: int


class ThemeCreateRequest(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_\-]+$", min_length=1, max_length=64)
    concept: List[str] = Field(default_factory=list)
    up: List[str] = Field(default_factory=list)
    down: List[str] = Field(default_factory=list)


class ThemeUpdateRequest(BaseModel):
    concept: Optional[List[str]] = None
    up: Optional[List[str]] = None
    down: Optional[List[str]] = None


# ─── Analysis ───────────────────────────────────────────────────────────────


class Frequency(str, Enum):
    monthly = "monthly"
    quarterly = "quarterly"
    yearly = "yearly"


# Maps API enum (English) → ina_core's French frequency labels
FREQUENCY_LABEL = {
    Frequency.monthly: "Mensuelle",
    Frequency.quarterly: "Trimestrielle",
    Frequency.yearly: "Annuelle",
}


class CountMode(str, Enum):
    binary = "binary"
    intensity = "intensity"


class AnalysisRequest(BaseModel):
    theme: str
    frequency: Frequency = Frequency.monthly
    count_mode: CountMode = CountMode.binary
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    channels: Optional[List[str]] = None  # None = all available

    @model_validator(mode="after")
    def _check_dates(self):
        if self.date_start and self.date_end and self.date_start > self.date_end:
            raise ValueError("date_start must be <= date_end")
        return self


class TimeSeriesPoint(BaseModel):
    period_start: date
    total_titles: int
    matched_titles: int
    occurrences_concept: int
    frequency: float
    up_titles: int
    down_titles: int
    ambiguous_titles: int
    net_signal: int


class KpiResponse(BaseModel):
    n_total: int
    n_matched: int
    occ_concept_total: int
    freq_mean: float
    net_signal_total: int
    up_total: int
    down_total: int
    ambiguous_total: int


class TopChannel(BaseModel):
    channel: str
    total_titles: int
    matched_titles: int
    frequency: float
    up_titles: int
    down_titles: int
    ambiguous_titles: int
    net_signal: int


class DescriptiveRow(BaseModel):
    indicateur: str
    valeur: Any


class AnalysisResponse(BaseModel):
    kpi: KpiResponse
    series: List[TimeSeriesPoint]
    top_channels: List[TopChannel]
    descriptive: List[DescriptiveRow]
    request: AnalysisRequest


# ─── Channels & Coverage ────────────────────────────────────────────────────


class ChannelCoverage(BaseModel):
    channel: str
    date_min: date
    date_max: date
    n_obs: int
    share_pct: float


class DecadePoint(BaseModel):
    channel: str
    decade: int
    decade_label: str
    n: int
    pct: float
