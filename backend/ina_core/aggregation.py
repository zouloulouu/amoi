"""Pure aggregation logic over tagged dataframes."""
from __future__ import annotations

import pandas as pd

from ina_core.tagging import DIRECTION_AMBIGUOUS, DIRECTION_DOWN, DIRECTION_UP


def periodize(series: pd.Series, frequency: str) -> pd.Series:
    """Bucket a datetime series into period-start timestamps.

    Accepts French frequency labels: "Mensuelle" (default), "Trimestrielle",
    "Annuelle".
    """
    if frequency == "Trimestrielle":
        return series.dt.to_period("Q").dt.to_timestamp()
    if frequency == "Annuelle":
        return series.dt.to_period("Y").dt.to_timestamp()
    return series.dt.to_period("M").dt.to_timestamp()


def aggregate_by_period(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate a tagged DataFrame by period_start, returning per-period stats."""
    out = df.assign(period_start=periodize(df["_date"], frequency)).copy()
    out["_match_mode"] = out["is_match"].astype(int)
    out["occurrences_concept"] = out["occ_concept"] * out["_match_mode"]
    out["up_flag"] = (out["direction"] == DIRECTION_UP).astype(int)
    out["down_flag"] = (out["direction"] == DIRECTION_DOWN).astype(int)
    out["ambiguous_flag"] = (out["direction"] == DIRECTION_AMBIGUOUS).astype(int)

    stats = (
        out.groupby("period_start", as_index=False)
        .agg(
            total_titles=("_match_mode", "size"),
            broad_matched_titles=("is_match_broad", "sum"),
            strict_matched_titles=("is_match_strict", "sum"),
            matched_titles=("is_match", "sum"),
            occurrences_concept=("occurrences_concept", "sum"),
            up_titles=("up_flag", "sum"),
            down_titles=("down_flag", "sum"),
            ambiguous_titles=("ambiguous_flag", "sum"),
        )
        .sort_values("period_start")
    )
    stats["frequency"] = stats["broad_matched_titles"] / stats["total_titles"]
    stats["strict_frequency"] = stats["strict_matched_titles"] / stats["total_titles"]
    stats["net_signal"] = stats["up_titles"] - stats["down_titles"]
    stats["direction_share_up"] = stats["up_titles"] / stats["strict_matched_titles"].replace(0, pd.NA)
    stats["direction_share_down"] = stats["down_titles"] / stats["strict_matched_titles"].replace(0, pd.NA)
    return stats


def build_descriptive_table(stats: pd.DataFrame, df_tagged: pd.DataFrame) -> pd.DataFrame:
    """Long-format indicator/value descriptive table for display."""
    if stats.empty or df_tagged.empty:
        return pd.DataFrame(columns=["indicateur", "valeur"])

    total_titles = int(len(df_tagged))
    matched_titles = int(df_tagged["is_match"].sum())
    occ_concept_total = int(df_tagged.loc[df_tagged["is_match"] == 1, "occ_concept"].sum())
    up_titles = int((df_tagged["direction"] == DIRECTION_UP).sum())
    down_titles = int((df_tagged["direction"] == DIRECTION_DOWN).sum())
    ambiguous_titles = int((df_tagged["direction"] == DIRECTION_AMBIGUOUS).sum())
    net_signal = up_titles - down_titles

    return pd.DataFrame([
        {"indicateur": "Titres analysés", "valeur": total_titles},
        {"indicateur": "Titres matchés (présence concept, binaire)", "valeur": matched_titles},
        {"indicateur": "Occurrences brutes concept (intensité)", "valeur": occ_concept_total},
        {"indicateur": "Signal UP (sens haussier)", "valeur": up_titles},
        {"indicateur": "Signal DOWN (sens baissier)", "valeur": down_titles},
        {"indicateur": "Signal AMBIGU (UP + DOWN simultanés)", "valeur": ambiguous_titles},
        {"indicateur": "Signal net (UP - DOWN)", "valeur": net_signal},
        {"indicateur": "Fréquence moyenne", "valeur": round(float(stats["frequency"].mean()), 4)},
        {"indicateur": "Fréquence médiane", "valeur": round(float(stats["frequency"].median()), 4)},
        {"indicateur": "Fréquence max", "valeur": round(float(stats["frequency"].max()), 4)},
        {"indicateur": "Volume moyen (titres matchés / période)", "valeur": round(float(stats["matched_titles"].mean()), 1)},
        {"indicateur": "Nombre de périodes", "valeur": int(len(stats))},
    ])


def build_top_channels(df_tagged: pd.DataFrame) -> pd.DataFrame:
    """Top 10 channels by matched titles."""
    if "_channel" not in df_tagged.columns:
        return pd.DataFrame()
    work = df_tagged.copy()
    work["_match_mode"] = work["is_match"].astype(int)
    work["occurrences_concept"] = work["occ_concept"] * work["_match_mode"]
    work["up_flag"] = (work["direction"] == DIRECTION_UP).astype(int)
    work["down_flag"] = (work["direction"] == DIRECTION_DOWN).astype(int)
    work["ambiguous_flag"] = (work["direction"] == DIRECTION_AMBIGUOUS).astype(int)
    top = (
        work.groupby("_channel", as_index=False)
        .agg(
            total_titles=("_match_mode", "size"),
            matched_titles=("_match_mode", "sum"),
            strict_matched_titles=("is_match_strict", "sum"),
            occurrences_concept=("occurrences_concept", "sum"),
            up_titles=("up_flag", "sum"),
            down_titles=("down_flag", "sum"),
            ambiguous_titles=("ambiguous_flag", "sum"),
        )
        .sort_values("matched_titles", ascending=False)
    )
    top["frequency"] = top["matched_titles"] / top["total_titles"]
    top["net_signal"] = top["up_titles"] - top["down_titles"]
    return top.head(10)
