import json
import logging
import os
import re
import unicodedata
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import streamlit as st


st.set_page_config(page_title="INA - Dictionnaire (simple)", layout="wide")

DATA_DIR = "data"
DICTIONARY_PATH = "dictionaries.json"
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "streamlit_app.log"

DEFAULT_DICTIONARIES: Dict[str, Dict[str, List[str]]] = {
    "inflation": {
        "concept": [
            "inflation",
            "pouvoir d'achat",
            "cout de la vie",
            "coût de la vie",
            "indice des prix",
            "ipc",
        ],
        "context": [],
        "up": [],
        "down": [],
    }
}

TITLE_CANDIDATES = [
    "titre_propre",
    "titre",
    "title",
    "intitule",
    "libelle",
    "titre_programme",
    "titre_collection",
]
DATE_CANDIDATES = ["date_diffusion", "date", "date_notice", "date_publication"]
TIME_CANDIDATES = ["heure_diffusion", "heure", "time", "horaire"]
CHANNEL_CANDIDATES = ["chaine", "channel"]

DIRECTION_UP = 1
DIRECTION_DOWN = -1
DIRECTION_FLAT = 0

TARGET_SCHEMA_COLUMNS = [
    "id",
    "chaine",
    "source_file",
    "date",
    "date_diffusion",
    "heure_diffusion",
    "duree",
    "duree_sec",
    "month",
    "annee",
    "mois",
    "ym",
    "titre_propre",
    "titre_collection",
    "titre_programme",
    "genre",
    "url_notice",
    "inflation_extended",
    "clean_titre",
    "clean_programme",
    "type_contenu",
    "emission_std",
    "has_url_notice",
    "has_titre_programme",
    "has_titre_collection",
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "id": ["id", "indice", "identifiant", "notice_id"],
    "chaine": ["chaine", "channel", "chaines", "chaine_diffusion"],
    "date": ["date", "date_notice", "date_publication"],
    "date_diffusion": ["date_diffusion", "diffusion_date", "date_emission"],
    "heure_diffusion": ["heure_diffusion", "heure", "time", "horaire", "heure_de_diffusion"],
    "duree": ["duree", "duration", "duree_diffusion"],
    "titre_propre": ["titre_propre", "titre", "title", "intitule", "libelle", "titre_all"],
    "titre_collection": ["titre_collection", "collection", "nom_collection"],
    "titre_programme": ["titre_programme", "programme", "nom_programme", "emission"],
    "genre": ["genre", "type_genre"],
    "url_notice": ["url_notice", "url", "notice_url", "lien_notice"],
    "inflation_extended": ["inflation_extended", "inflation", "label_inflation"],
    "source_file": ["source_file", "fichier_source", "file_source"],
    "month": ["month", "mois_num", "month_num"],
}

CHANNEL_MAPPING = {
    "bfm tv": "BFMTV",
    "bfmtv": "BFMTV",
    "b f m tv": "BFMTV",
    "france 2": "France 2",
    "france2": "France 2",
    "fr2": "France 2",
    "f2": "France 2",
    "tf1": "TF1",
    "t f 1": "TF1",
    "france 3": "France 3",
    "france3": "France 3",
    "fr3": "France 3",
    "france inter": "France Inter",
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("data_ina.app")
    if logger.handlers:
        return logger

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    except Exception:
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = setup_logger()


def stop_with_error_log(user_message: str, context: str, exc: Exception) -> None:
    LOGGER.exception("%s: %s", context, exc)
    st.error(f"{user_message} Consulte le log `{LOG_PATH.as_posix()}`.")
    st.stop()


def empty_theme_dictionary() -> Dict[str, List[str]]:
    return {"concept": [], "context": [], "up": [], "down": []}


def clean_term_list(values) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def normalize_theme_dictionary(raw_theme) -> Dict[str, List[str]]:
    if isinstance(raw_theme, list):
        return {
            "concept": clean_term_list(raw_theme),
            "context": [],
            "up": [],
            "down": [],
        }
    if not isinstance(raw_theme, dict):
        return empty_theme_dictionary()
    return {
        "concept": clean_term_list(raw_theme.get("concept", [])),
        "context": clean_term_list(raw_theme.get("context", [])),
        "up": clean_term_list(raw_theme.get("up", [])),
        "down": clean_term_list(raw_theme.get("down", [])),
    }


def normalize_dictionaries_payload(raw_data) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    if not isinstance(raw_data, dict):
        return out
    for key, payload in raw_data.items():
        if not isinstance(key, str):
            continue
        theme = key.strip()
        if not theme:
            continue
        out[theme] = normalize_theme_dictionary(payload)
    return out


def clone_dictionaries(dictionaries: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    return json.loads(json.dumps(dictionaries, ensure_ascii=False))


def _is_missing_value(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def ensure_columns(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    created_columns: List[str] = []
    for col in required_columns:
        if col not in out.columns:
            out[col] = pd.NA
            created_columns.append(col)
    out.attrs["created_columns"] = created_columns
    return out


def normalize_text(value: Any):
    if _is_missing_value(value):
        return pd.NA
    text = str(value).replace("\u00a0", " ").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA


def strip_accents(text: Any):
    if _is_missing_value(text):
        return pd.NA
    raw = str(text)
    return "".join(c for c in unicodedata.normalize("NFD", raw) if unicodedata.category(c) != "Mn")


def clean_title(value: Any):
    normalized = normalize_text(value)
    if _is_missing_value(normalized):
        return pd.NA
    no_accents = strip_accents(normalized)
    if _is_missing_value(no_accents):
        return pd.NA
    cleaned = re.sub(r"[^\w\s]", " ", str(no_accents))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else pd.NA


def normalize_text_series(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    out = out.str.replace("\u00a0", " ", regex=False).str.lower()
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.replace("", pd.NA)


def strip_accents_series(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    out = out.str.normalize("NFD").str.replace(r"[\u0300-\u036f]", "", regex=True)
    return out.replace("", pd.NA)


def clean_title_series(series: pd.Series) -> pd.Series:
    out = normalize_text_series(series)
    out = strip_accents_series(out)
    out = out.str.replace(r"[^\w\s]", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.replace("", pd.NA)


def clean_title_series_from_normalized(normalized_series: pd.Series) -> pd.Series:
    out = normalized_series.astype("string")
    out = strip_accents_series(out)
    out = out.str.replace(r"[^\w\s]", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.replace("", pd.NA)


def _normalize_text_for_key(value: Any) -> str:
    normalized = normalize_text(value)
    if _is_missing_value(normalized):
        return ""
    no_accents = strip_accents(normalized)
    return "" if _is_missing_value(no_accents) else str(no_accents)


def canon_colname(name: str) -> str:
    c = _normalize_text_for_key(str(name).replace("\u00a0", " "))
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^a-z0-9_]", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def parse_duration_to_seconds(value: Any) -> float:
    if _is_missing_value(value):
        return float("nan")
    raw = str(value).strip()
    if not raw:
        return float("nan")
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        return float(raw)

    cleaned = re.sub(r"[^0-9:]", "", raw)
    if not cleaned:
        return float("nan")

    full_match = re.fullmatch(r"(\d{1,3}):(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", cleaned)
    if full_match:
        hours = int(full_match.group(1))
        minutes = int(full_match.group(2))
        seconds = int(full_match.group(3))
        centiseconds = int(full_match.group(4)) if full_match.group(4) is not None else 0
        if minutes >= 60 or seconds >= 60:
            return float("nan")
        return float(hours * 3600 + minutes * 60 + seconds + (centiseconds / 100))

    short_match = re.fullmatch(r"(\d{1,3}):(\d{1,2})", cleaned)
    if short_match:
        minutes = int(short_match.group(1))
        seconds = int(short_match.group(2))
        if seconds >= 60:
            return float("nan")
        return float(minutes * 60 + seconds)
    return float("nan")


def parse_duration_series_to_seconds(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    numeric_mask = values.str.match(r"^\d+(\.\d+)?$", na=False)
    out = pd.to_numeric(values.where(numeric_mask), errors="coerce").astype("float64")

    cleaned = values.str.replace(r"[^0-9:]", "", regex=True)
    hms = cleaned.str.extract(r"^(?P<h>\d{1,3}):(?P<m>\d{1,2}):(?P<s>\d{1,2})(?::(?P<c>\d{1,2}))?$")
    h = pd.to_numeric(hms["h"], errors="coerce")
    m = pd.to_numeric(hms["m"], errors="coerce")
    s = pd.to_numeric(hms["s"], errors="coerce")
    c = pd.to_numeric(hms["c"], errors="coerce").fillna(0)
    valid_hms = h.notna() & m.notna() & s.notna() & (m < 60) & (s < 60)
    hms_seconds = ((h * 3600) + (m * 60) + s + (c / 100)).astype("float64")
    out = out.where(out.notna(), hms_seconds.where(valid_hms))

    ms = cleaned.str.extract(r"^(?P<m>\d{1,3}):(?P<s>\d{1,2})$")
    mm = pd.to_numeric(ms["m"], errors="coerce")
    ss = pd.to_numeric(ms["s"], errors="coerce")
    valid_ms = mm.notna() & ss.notna() & (ss < 60)
    ms_seconds = ((mm * 60) + ss).astype("float64")
    out = out.where(out.notna(), ms_seconds.where(valid_ms))
    return out


def harmonize_channel(value: Any) -> Any:
    if _is_missing_value(value):
        return pd.NA
    key = _normalize_text_for_key(value)
    key = re.sub(r"[^\w\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    if not key:
        return pd.NA
    mapped = CHANNEL_MAPPING.get(key)
    if mapped:
        return mapped
    if "france 2" in key:
        return "France 2"
    if "france 3" in key:
        return "France 3"
    if "bfm" in key and "tv" in key:
        return "BFMTV"
    if "france inter" in key:
        return "France Inter"
    if "tf1" in key:
        return "TF1"
    return str(value).strip()


def harmonize_channel_series(series: pd.Series) -> pd.Series:
    raw = series.astype("string")
    key = strip_accents_series(normalize_text_series(raw))
    key = key.str.replace(r"[^\w\s]", " ", regex=True)
    key = key.str.replace(r"\s+", " ", regex=True).str.strip()
    mapped = key.map(CHANNEL_MAPPING).astype("string")
    mapped = mapped.mask(mapped.isna() & key.str.contains("france 2", na=False), "France 2")
    mapped = mapped.mask(mapped.isna() & key.str.contains("france 3", na=False), "France 3")
    mapped = mapped.mask(mapped.isna() & key.str.contains("bfm") & key.str.contains("tv"), "BFMTV")
    mapped = mapped.mask(mapped.isna() & key.str.contains("france inter", na=False), "France Inter")
    mapped = mapped.mask(mapped.isna() & key.str.contains("tf1", na=False), "TF1")
    mapped = mapped.mask(mapped.isna(), raw.str.strip())
    return mapped.replace("", pd.NA)


def normalize_channel(value: Any) -> Any:
    return harmonize_channel(value)


def classify_content(row: pd.Series) -> str:
    content_text = " ".join(
        [
            _normalize_text_for_key(row.get("titre_propre")),
            _normalize_text_for_key(row.get("titre_programme")),
            _normalize_text_for_key(row.get("titre_collection")),
        ]
    ).strip()
    if "plateau" in content_text:
        return "plateau"
    if any(token in content_text for token in ("programme du", "emission du")):
        return "edition_complete"
    return "sujet"


def standardize_emission(row: pd.Series):
    content_text = " ".join(
        [
            _normalize_text_for_key(row.get("titre_propre")),
            _normalize_text_for_key(row.get("titre_programme")),
            _normalize_text_for_key(row.get("titre_collection")),
        ]
    ).strip()
    compact = content_text.replace(" ", "")
    if "20heures" in compact or "20h" in compact:
        return "20 heures"
    if "premiere edition" in content_text or "premiereedition" in compact:
        return "Premiere edition"
    return pd.NA


def _build_content_text(df: pd.DataFrame) -> pd.Series:
    part1 = normalize_text_series(df["titre_propre"])
    part2 = normalize_text_series(df["titre_programme"])
    part3 = normalize_text_series(df["titre_collection"])
    content = part1.fillna("") + " " + part2.fillna("") + " " + part3.fillna("")
    return content.str.replace(r"\s+", " ", regex=True).str.strip()


def classify_content_series(df: pd.DataFrame, content: Optional[pd.Series] = None) -> pd.Series:
    content = _build_content_text(df) if content is None else content
    is_plateau = content.str.contains(r"\bplateau\b", regex=True, na=False)
    is_edition = content.str.contains(r"programme du|emission du|émission du", regex=True, na=False)
    labels = np.select([is_plateau, is_edition], ["plateau", "edition_complete"], default="sujet")
    return pd.Series(labels, index=df.index, dtype="string")


def standardize_emission_series(df: pd.DataFrame, content: Optional[pd.Series] = None) -> pd.Series:
    content = _build_content_text(df) if content is None else content
    compact = content.str.replace(" ", "", regex=False)
    is_20h = compact.str.contains(r"20heures|20h", regex=True, na=False)
    is_premiere = content.str.contains(r"premiere edition|première édition|premiereedition|premièreédition", regex=True, na=False) | compact.str.contains(
        r"premiereedition|premièreédition", regex=True, na=False
    )
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out = out.mask(is_20h, "20 heures")
    out = out.mask(~is_20h & is_premiere, "Premiere edition")
    return out


def _resolve_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    source_by_canon = {canon_colname(col): col for col in df.columns}
    renamed = df.copy()
    for target_col, aliases in COLUMN_ALIASES.items():
        source_name = next((source_by_canon[a] for a in aliases if a in source_by_canon), None)
        if source_name and source_name != target_col and target_col not in renamed.columns:
            renamed = renamed.rename(columns={source_name: target_col})
    return renamed


def _coalesce_text_columns(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        source = df[col].astype("string")
        valid = source.notna() & source.str.strip().ne("")
        result = result.where(result.notna(), source.where(valid, pd.NA))
    return result.astype("string")


def clean_file(df: pd.DataFrame, source_name: Optional[str] = None) -> pd.DataFrame:
    out = _resolve_alias_columns(df.copy())
    initial_columns = list(out.columns)
    out = ensure_columns(out, TARGET_SCHEMA_COLUMNS)
    created_columns = out.attrs.get("created_columns", [])

    if source_name is not None:
        source_values = out["source_file"].astype("string")
        missing_source = source_values.isna() | source_values.str.strip().eq("")
        out["source_file"] = source_values.where(~missing_source, source_name)

    out["titre_propre"] = _coalesce_text_columns(out, ["titre_propre", "titre_programme", "titre_collection"]).astype("string")
    out["titre_programme"] = out["titre_programme"].astype("string")
    out["titre_collection"] = out["titre_collection"].astype("string")
    titre_propre_norm = normalize_text_series(out["titre_propre"])
    titre_programme_norm = normalize_text_series(out["titre_programme"])
    titre_collection_norm = normalize_text_series(out["titre_collection"])

    out["genre"] = normalize_text_series(out["genre"]).astype("string")
    out["url_notice"] = out["url_notice"].astype("string")
    out["chaine"] = harmonize_channel_series(out["chaine"]).astype("string")
    out["duree"] = out["duree"].astype("string")
    out["heure_diffusion"] = out["heure_diffusion"].astype("string")

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["date_diffusion"] = pd.to_datetime(out["date_diffusion"], errors="coerce", dayfirst=True)

    out["inflation_extended"] = pd.to_numeric(out["inflation_extended"], errors="coerce").round().astype("Int64")
    out["duree_sec"] = parse_duration_series_to_seconds(out["duree"])

    date_base = out["date_diffusion"].where(out["date_diffusion"].notna(), out["date"])
    out["annee"] = date_base.dt.year.astype("Int64")
    out["mois"] = date_base.dt.month.astype("Int64")
    out["month"] = out["mois"]
    out["ym"] = date_base.dt.to_period("M").astype("string")
    out["clean_titre"] = clean_title_series_from_normalized(titre_propre_norm).astype("string")
    out["clean_programme"] = clean_title_series_from_normalized(titre_programme_norm).astype("string")
    content_text = (titre_propre_norm.fillna("") + " " + titre_programme_norm.fillna("") + " " + titre_collection_norm.fillna(""))
    content_text = content_text.str.replace(r"\s+", " ", regex=True).str.strip()
    out["type_contenu"] = classify_content_series(out, content=content_text).astype("string")
    out["emission_std"] = standardize_emission_series(out, content=content_text).astype("string")
    out["has_url_notice"] = (
        out["url_notice"].notna() & out["url_notice"].str.strip().ne("").fillna(False)
    ).astype("int8")
    out["has_titre_programme"] = (
        out["titre_programme"].notna() & out["titre_programme"].astype("string").str.strip().ne("").fillna(False)
    ).astype("int8")
    out["has_titre_collection"] = (
        out["titre_collection"].notna() & out["titre_collection"].astype("string").str.strip().ne("").fillna(False)
    ).astype("int8")

    dedup_subset = [col for col in ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"] if col in out.columns]
    if dedup_subset:
        duplicate_mask = out.duplicated(subset=dedup_subset, keep="first")
    else:
        duplicate_mask = pd.Series(False, index=out.index)
    duplicates_removed = int(duplicate_mask.sum())
    duplicate_channel_counts = (
        out.loc[duplicate_mask, "chaine"]
        .fillna("(sans chaine)")
        .astype("string")
        .value_counts(dropna=False)
        .to_dict()
    )
    out = out.loc[~duplicate_mask].copy()

    out["title"] = _coalesce_text_columns(out, ["titre_propre", "titre_programme", "titre_collection"])
    out["channel"] = out["chaine"].astype("string")
    out["time"] = out["heure_diffusion"].astype("string")
    out["_date"] = parse_datetime(
        out.assign(_date_merge=out["date_diffusion"].where(out["date_diffusion"].notna(), out["date"])),
        "_date_merge",
        "heure_diffusion",
    )
    out["_date"] = out["_date"].where(out["_date"].notna(), out["date_diffusion"].where(out["date_diffusion"].notna(), out["date"]))
    out["_title_norm"] = out["title"].map(_normalize_text_for_key).fillna("").astype(str)
    out["_channel"] = out["chaine"].fillna("(sans chaine)").astype("string").replace("", "(sans chaine)")

    out.attrs["initial_columns"] = initial_columns
    out.attrs["final_columns"] = list(out.columns)
    out.attrs["created_columns"] = created_columns
    out.attrs["duplicates_removed"] = duplicates_removed
    out.attrs["duplicates_removed_by_channel"] = {str(k): int(v) for k, v in duplicate_channel_counts.items()}
    return out


def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "chaine",
                "nb_lignes",
                "doublons_supprimes",
                "tx_na_date_diffusion",
                "tx_na_titre_propre",
                "tx_na_titre_programme",
                "tx_na_duree_sec",
                "part_type_plateau",
                "part_type_edition_complete",
                "part_type_sujet",
                "part_inflation_1",
            ]
        )

    work = df.copy()
    work["chaine"] = work["chaine"].fillna("(sans chaine)").astype("string")
    work["type_contenu"] = work["type_contenu"].fillna("sujet").astype("string")
    work["inflation_flag"] = (work["inflation_extended"] == 1).astype("int8")
    grouped = work.groupby("chaine", dropna=False)

    report = grouped.size().reset_index(name="nb_lignes")
    missing_columns = ["date_diffusion", "titre_propre", "titre_programme", "duree_sec"]
    for col in missing_columns:
        if col in work.columns:
            report[f"tx_na_{col}"] = grouped[col].apply(lambda s: float(s.isna().mean())).to_numpy()
        else:
            report[f"tx_na_{col}"] = np.nan

    report["part_inflation_1"] = grouped["inflation_flag"].mean().to_numpy()
    type_share = (
        grouped["type_contenu"]
        .value_counts(normalize=True)
        .rename("part")
        .reset_index()
        .pivot(index="chaine", columns="type_contenu", values="part")
        .fillna(0.0)
    )
    for content_col in ["plateau", "edition_complete", "sujet"]:
        report[f"part_type_{content_col}"] = report["chaine"].map(type_share.get(content_col, pd.Series(dtype=float))).fillna(0.0)

    duplicates_by_channel = df.attrs.get("duplicates_removed_by_channel", {})
    report["doublons_supprimes"] = report["chaine"].astype(str).map(lambda x: int(duplicates_by_channel.get(x, 0)))
    report = report.sort_values("nb_lignes", ascending=False).reset_index(drop=True)
    return report


def build_monthly_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "chaine",
                "ym",
                "nb_total",
                "nb_inflation",
                "duree_totale",
                "duree_inflation",
                "part_inflation_nb",
                "part_inflation_duree",
            ]
        )

    work = df.copy()
    work["chaine"] = work["chaine"].fillna("(sans chaine)").astype("string")
    if "ym" not in work.columns:
        date_base = work["date_diffusion"].where(work["date_diffusion"].notna(), work["date"])
        work["ym"] = date_base.dt.to_period("M").astype("string")
    work["ym"] = work["ym"].fillna("inconnu").astype("string")
    work["inflation_flag"] = (work["inflation_extended"] == 1).astype("int8")
    work["duree_sec_num"] = pd.to_numeric(work["duree_sec"], errors="coerce").fillna(0.0)
    work["duree_inflation"] = np.where(work["inflation_flag"] == 1, work["duree_sec_num"], 0.0)

    monthly = (
        work.groupby(["chaine", "ym"], as_index=False)
        .agg(
            nb_total=("inflation_flag", "size"),
            nb_inflation=("inflation_flag", "sum"),
            duree_totale=("duree_sec_num", "sum"),
            duree_inflation=("duree_inflation", "sum"),
        )
        .sort_values(["chaine", "ym"])
        .reset_index(drop=True)
    )
    monthly["part_inflation_nb"] = np.where(monthly["nb_total"] > 0, monthly["nb_inflation"] / monthly["nb_total"], np.nan)
    monthly["part_inflation_duree"] = np.where(
        monthly["duree_totale"] > 0,
        monthly["duree_inflation"] / monthly["duree_totale"],
        np.nan,
    )
    return monthly


def load_dictionaries(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.exists(path):
        return clone_dictionaries(DEFAULT_DICTIONARIES)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return clone_dictionaries(DEFAULT_DICTIONARIES)
        data = json.loads(raw)
        out = normalize_dictionaries_payload(data)
        return out if out else clone_dictionaries(DEFAULT_DICTIONARIES)
    except Exception as exc:
        LOGGER.exception("Impossible de charger le dictionnaire %s: %s", path, exc)
        return clone_dictionaries(DEFAULT_DICTIONARIES)


def save_dictionaries(path: str, dictionaries: Dict[str, Dict[str, List[str]]]) -> None:
    normalized = normalize_dictionaries_payload(dictionaries)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def parquet_signature(folder: str) -> Tuple[Tuple[str, int, int], ...]:
    if not os.path.isdir(folder):
        return tuple()
    signature: List[Tuple[str, int, int]] = []
    for name in sorted(os.listdir(folder)):
        if not str(name).lower().endswith(".parquet"):
            continue
        path = os.path.join(folder, name)
        try:
            stat = os.stat(path)
            signature.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
        except OSError as exc:
            LOGGER.warning("Stat impossible sur %s: %s", path, exc)
    return tuple(signature)


@st.cache_resource(show_spinner=False)
def load_parquets_from_folder(
    folder: str, signature: Tuple[Tuple[str, int, int], ...], normalize_channels: bool
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    issues: List[str] = []
    load_diagnostics: List[Dict[str, Any]] = []
    if not os.path.isdir(folder):
        LOGGER.warning("Dossier data introuvable: %s", folder)
        issues.append(f"Dossier data introuvable: {folder}")
        return pd.DataFrame(), issues, pd.DataFrame()
    if not signature:
        LOGGER.warning("Aucun fichier parquet trouve dans %s", folder)
        issues.append(f"Aucun fichier `.parquet` trouve dans {folder}")
        return pd.DataFrame(), issues, pd.DataFrame()

    frames: List[pd.DataFrame] = []
    duplicates_removed_total = 0
    duplicates_removed_by_channel: Dict[str, int] = {}
    run_mode = "single_file" if len(signature) == 1 else "multi_file"
    for file_name, _, _ in signature:
        path = os.path.join(folder, file_name)
        try:
            schema_names = pq.ParquetFile(path).schema.names
            raw = pd.read_parquet(path)
            cleaned = clean_file(raw, source_name=file_name)

            if normalize_channels:
                cleaned["chaine"] = harmonize_channel_series(cleaned["chaine"]).astype("string")
                cleaned["channel"] = cleaned["chaine"].astype("string")
                cleaned["_channel"] = cleaned["chaine"].fillna("(sans chaine)").astype("string").replace("", "(sans chaine)")

            if cleaned["title"].isna().all():
                issues.append(f"Titre absent apres harmonisation dans {file_name}.")
            if cleaned["_date"].isna().all():
                issues.append(f"Dates invalides apres parsing dans {file_name}.")

            duplicates_removed = int(cleaned.attrs.get("duplicates_removed", 0))
            duplicates_removed_total += duplicates_removed
            file_dup_by_channel = cleaned.attrs.get("duplicates_removed_by_channel", {})
            for channel_name, count in file_dup_by_channel.items():
                duplicates_removed_by_channel[channel_name] = duplicates_removed_by_channel.get(channel_name, 0) + int(count)

            created_columns = cleaned.attrs.get("created_columns", [])
            load_diagnostics.append(
                {
                    "source_file": file_name,
                    "mode": run_mode,
                    "rows_raw": int(len(raw)),
                    "rows_clean": int(len(cleaned)),
                    "duplicates_removed": duplicates_removed,
                    "created_columns_count": int(len(created_columns)),
                    "created_columns": ", ".join(created_columns) if created_columns else "",
                    "columns_before": ", ".join(schema_names),
                    "columns_after": ", ".join(cleaned.columns),
                }
            )

            frames.append(cleaned)
            LOGGER.info(
                "Parquet charge et nettoye: %s | lignes_brutes=%s | lignes_nettoyees=%s",
                file_name,
                len(raw),
                len(cleaned),
            )
        except Exception as exc:
            LOGGER.exception("Lecture impossible sur %s: %s", file_name, exc)
            issues.append(f"Lecture impossible: {file_name} ({exc})")

    if not frames:
        return pd.DataFrame(), issues, pd.DataFrame(load_diagnostics)

    if len(frames) == 1:
        combined = frames[0].copy()
    else:
        combined = pd.concat(frames, ignore_index=True, sort=False)

    combined = combined[combined["_date"].notna()].copy()
    if combined.empty:
        issues.append("Aucune date valide apres parsing.")
        return combined, issues, pd.DataFrame(load_diagnostics)

    combined.attrs["duplicates_removed_total"] = duplicates_removed_total
    combined.attrs["duplicates_removed_by_channel"] = duplicates_removed_by_channel
    combined.attrs["load_diagnostics"] = load_diagnostics
    return combined, issues, pd.DataFrame(load_diagnostics)


def parse_datetime(df: pd.DataFrame, date_col: str, time_col: Optional[str]) -> pd.Series:
    dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if not time_col or time_col not in df.columns:
        return dt

    clean_time = df[time_col].astype(str).str.strip().str.lower()
    clean_time = clean_time.str.replace("h", ":", regex=False)
    clean_time = clean_time.str.replace(r"[^0-9:]", "", regex=True)

    def normalize_clock(x: str) -> str:
        if not x or x == "nan":
            return ""
        if x.isdigit() and len(x) == 4:
            return f"{x[:2]}:{x[2:]}:00"
        if re.match(r"^\d{1,2}:\d{2}$", x):
            return x + ":00"
        if re.match(r"^\d{1,2}:\d{2}:\d{2}$", x):
            return x
        return ""

    fixed = clean_time.map(normalize_clock)
    combo = pd.to_datetime(dt.dt.date.astype(str) + " " + fixed, errors="coerce")
    return combo.where(combo.notna(), dt)


def prepare_keywords(keywords: List[str]) -> List[str]:
    normalized = [_normalize_text_for_key(k) for k in keywords if str(k).strip()]
    dedup = sorted(set(k for k in normalized if k))
    return dedup


def count_occurrences(text_norm: Any, keywords_norm: List[str]) -> int:
    text_value = _normalize_text_for_key(text_norm)
    if not text_value:
        return 0
    total = 0
    for keyword in keywords_norm:
        if len(keyword) <= 4 and keyword.isalpha():
            total += len(re.findall(rf"\b{re.escape(keyword)}\b", text_value))
        else:
            total += text_value.count(keyword)
    return total


def add_tagging_columns_hier(
    df: pd.DataFrame,
    title_col: str,
    concept_norm: List[str],
    context_norm: List[str],
    up_norm: List[str],
    down_norm: List[str],
    title_norm_col: Optional[str] = None,
) -> pd.DataFrame:
    if title_norm_col and title_norm_col in df.columns:
        titles_norm = df[title_norm_col].fillna("").astype(str)
    else:
        titles_norm = df[title_col].fillna("").astype(str).map(_normalize_text_for_key)

    df["occ_concept"] = titles_norm.map(lambda x: count_occurrences(x, concept_norm))
    # Contexte conserve pour compatibilite JSON, mais non utilise dans le matching.
    df["occ_context"] = 0
    df["occ_up"] = titles_norm.map(lambda x: count_occurrences(x, up_norm))
    df["occ_down"] = titles_norm.map(lambda x: count_occurrences(x, down_norm))

    df["is_concept"] = (df["occ_concept"] > 0).astype("int8")
    df["is_context"] = 0
    df["is_match_broad"] = df["is_concept"].astype("int8")
    df["is_match_strict"] = df["is_match_broad"].astype("int8")
    df["is_match"] = df["is_match_broad"].astype("int8")

    direction = pd.Series(DIRECTION_FLAT, index=df.index, dtype="int8")
    matched_rows = df["is_match"] == 1
    direction.loc[matched_rows & (df["occ_up"] > df["occ_down"])] = DIRECTION_UP
    direction.loc[matched_rows & (df["occ_down"] > df["occ_up"])] = DIRECTION_DOWN
    df["direction"] = direction

    return df


def periodize(series: pd.Series, frequency: str) -> pd.Series:
    if frequency == "Trimestrielle":
        return series.dt.to_period("Q").dt.to_timestamp()
    if frequency == "Annuelle":
        return series.dt.to_period("Y").dt.to_timestamp()
    return series.dt.to_period("M").dt.to_timestamp()


def aggregate_by_period(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    match_col = "is_match"
    out = df.assign(period_start=periodize(df["_date"], frequency)).copy()
    out["_match_mode"] = out[match_col].astype(int)
    out["occurrences_concept"] = out["occ_concept"] * out["_match_mode"]
    out["up_flag"] = (out["direction"] == DIRECTION_UP).astype(int)
    out["down_flag"] = (out["direction"] == DIRECTION_DOWN).astype(int)

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
    if stats.empty or df_tagged.empty:
        return pd.DataFrame(columns=["indicateur", "valeur"])

    total_titles = int(len(df_tagged))
    matched_titles = int(df_tagged["is_match"].sum())
    occ_concept_total = int(df_tagged.loc[df_tagged["is_match"] == 1, "occ_concept"].sum())
    up_titles = int((df_tagged["direction"] == DIRECTION_UP).sum())
    down_titles = int((df_tagged["direction"] == DIRECTION_DOWN).sum())
    net_signal = up_titles - down_titles

    return pd.DataFrame(
        [
            {"indicateur": "Titres analyses", "valeur": total_titles},
            {"indicateur": "Titres matches", "valeur": matched_titles},
            {"indicateur": "Occurrences concept totales", "valeur": occ_concept_total},
            {"indicateur": "Up", "valeur": up_titles},
            {"indicateur": "Down", "valeur": down_titles},
            {"indicateur": "Signal net", "valeur": net_signal},
            {"indicateur": "Frequence moyenne", "valeur": float(stats["frequency"].mean())},
            {"indicateur": "Frequence mediane", "valeur": float(stats["frequency"].median())},
            {"indicateur": "Frequence max", "valeur": float(stats["frequency"].max())},
            {"indicateur": "Volume moyen (titres matches)", "valeur": float(stats["matched_titles"].mean())},
            {"indicateur": "Nb periodes", "valeur": int(len(stats))},
        ]
    )


def build_top_channels(df_tagged: pd.DataFrame) -> pd.DataFrame:
    if "_channel" not in df_tagged.columns:
        return pd.DataFrame()

    work = df_tagged.copy()
    work["_match_mode"] = work["is_match"].astype(int)
    work["occurrences_concept"] = work["occ_concept"] * work["_match_mode"]
    work["up_flag"] = (work["direction"] == DIRECTION_UP).astype(int)
    work["down_flag"] = (work["direction"] == DIRECTION_DOWN).astype(int)

    top = (
        work.groupby("_channel", as_index=False)
        .agg(
            total_titles=("_match_mode", "size"),
            matched_titles=("_match_mode", "sum"),
            strict_matched_titles=("is_match_strict", "sum"),
            occurrences_concept=("occurrences_concept", "sum"),
            up_titles=("up_flag", "sum"),
            down_titles=("down_flag", "sum"),
        )
        .sort_values("matched_titles", ascending=False)
    )
    top["frequency"] = top["matched_titles"] / top["total_titles"]
    top["net_signal"] = top["up_titles"] - top["down_titles"]
    return top.head(10)


def apply_time_axis_controls(fig) -> None:
    fig.update_xaxes(
        type="date",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(count=3, label="3a", step="year", stepmode="backward"),
                dict(count=5, label="5a", step="year", stepmode="backward"),
                dict(step="all", label="Tout"),
            ]
        ),
        rangeslider=dict(
            visible=True,
            thickness=0.14,
            bgcolor="rgba(120,120,120,0.15)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
        ),
    )


st.title("INA - Recherche de dictionnaires (version simplifiee)")
st.caption("Comptage de mots-cles dans les titres, stats descriptives, top chaines, et series temporelles.")
normalize_channels = True

with st.expander("Donnees", expanded=False):
    st.write("Chargement automatique de tous les fichiers `.parquet` du dossier local `data/`.")

data_signature = parquet_signature(DATA_DIR)
df_base, load_issues, df_load_diagnostics = load_parquets_from_folder(
    DATA_DIR,
    data_signature,
    normalize_channels=normalize_channels,
)
for issue in load_issues:
    st.warning(issue)

if df_base.empty:
    st.warning("Aucune donnee chargee (aucun fichier `.parquet` lisible dans `data/`).")
    st.stop()

df_quality = build_quality_report(df_base)
df_monthly = build_monthly_analysis(df_base)

with st.expander("Controle qualite et harmonisation", expanded=False):
    st.caption("Diagnostic de nettoyage des fichiers parquet et apercus des bases construites.")
    if not df_load_diagnostics.empty:
        st.write("Diagnostic par fichier charge")
        st.dataframe(df_load_diagnostics, width="stretch")

    st.write("Colonnes finales harmonisees")
    st.code(", ".join(list(df_base.columns)), language="text")
    st.metric("Doublons supprimes", int(df_base.attrs.get("duplicates_removed_total", 0)))

    show_cleaned_data = st.checkbox("Afficher les donnees nettoyees", value=False)
    if show_cleaned_data:
        preview_cols = [col for col in TARGET_SCHEMA_COLUMNS + ["_date", "_channel"] if col in df_base.columns]
        st.dataframe(df_base.head(200)[preview_cols], width="stretch")

    show_quality = st.checkbox("Afficher le rapport qualite", value=False)
    if show_quality:
        st.dataframe(df_quality, width="stretch")

    show_monthly = st.checkbox("Afficher la base mensuelle", value=False)
    if show_monthly:
        st.dataframe(df_monthly, width="stretch")

if "dictionaries" not in st.session_state:
    st.session_state["dictionaries"] = load_dictionaries(DICTIONARY_PATH)

dictionaries = normalize_dictionaries_payload(st.session_state["dictionaries"])
if not dictionaries:
    dictionaries = clone_dictionaries(DEFAULT_DICTIONARIES)
st.session_state["dictionaries"] = dictionaries

columns = list(df_base.columns)
title_col = "title" if "title" in columns else None
has_source_channel = "channel" in columns
channel_col = "_channel" if "_channel" in columns else None

if not title_col or "_date" not in columns:
    st.error("Colonnes minimales introuvables: il faut au moins une colonne titre et une colonne date.")
    st.stop()

st.sidebar.header("Parametres")
frequency = st.sidebar.selectbox("Frequence", ["Mensuelle", "Trimestrielle", "Annuelle"], index=0)
with st.sidebar.expander("Diagnostics", expanded=False):
    st.caption(f"Log erreurs: `{LOG_PATH.as_posix()}`")
    if st.checkbox("Afficher les 30 dernieres lignes du log", value=False):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                last_lines = f.readlines()[-30:]
            st.code("".join(last_lines) if last_lines else "(log vide)", language="text")
        except Exception as exc:
            LOGGER.exception("Lecture du log impossible: %s", exc)
            st.warning(f"Lecture du log impossible ({exc})")

themes = sorted(dictionaries.keys())
if not themes:
    st.session_state["dictionaries"] = clone_dictionaries(DEFAULT_DICTIONARIES)
    dictionaries = st.session_state["dictionaries"]
    themes = sorted(dictionaries.keys())

if "theme" not in st.session_state or st.session_state["theme"] not in themes:
    st.session_state["theme"] = themes[0]
theme = st.sidebar.selectbox(
    "Theme",
    options=themes,
    index=themes.index(st.session_state["theme"]) if st.session_state["theme"] in themes else 0,
)
st.session_state["theme"] = theme

with st.expander("Dictionnaires", expanded=False):
    if st.session_state.get("dict_flash"):
        st.success(st.session_state["dict_flash"])
        st.session_state.pop("dict_flash", None)

    st.markdown(
        "1. Saisis uniquement le nom du nouveau theme, puis clique `Ajouter un theme`.\n"
        "2. Renseigne Concept + Sens (UP/DOWN), puis clique `Enregistrer dictionnaire`."
    )

    with st.form("add_theme_form", clear_on_submit=True):
        new_theme = st.text_input("Nouveau theme", value="")
        add_theme_submitted = st.form_submit_button("Ajouter un theme")
    if add_theme_submitted:
        nt = new_theme.strip()
        if not nt:
            st.warning("Nom de theme vide.")
        elif nt in dictionaries:
            st.warning("Ce theme existe deja.")
        else:
            dictionaries[nt] = empty_theme_dictionary()
            st.session_state["dictionaries"] = dictionaries
            st.session_state["theme"] = nt
            st.session_state["dict_flash"] = f"Theme '{nt}' cree. Ajoute maintenant ses mots-cles."
            try:
                save_dictionaries(DICTIONARY_PATH, dictionaries)
            except Exception as exc:
                LOGGER.exception("Ecriture dictionnaire impossible (creation theme): %s", exc)
                st.warning(f"Ecriture du fichier dictionnaire impossible ({exc}).")
            st.rerun()

    current_theme_dict = normalize_theme_dictionary(dictionaries.get(theme, empty_theme_dictionary()))

    with st.form("edit_theme_form"):
        concept_text = st.text_area(
            f"Concept ({theme})",
            value="\n".join(current_theme_dict["concept"]),
            height=140,
        )
        up_text = st.text_area(
            f"Sens UP ({theme})",
            value="\n".join(current_theme_dict["up"]),
            height=120,
        )
        down_text = st.text_area(
            f"Sens DOWN ({theme})",
            value="\n".join(current_theme_dict["down"]),
            height=120,
        )
        save_theme_submitted = st.form_submit_button("Enregistrer dictionnaire")
    if save_theme_submitted:
        dictionaries[theme] = {
            "concept": [k.strip() for k in concept_text.splitlines() if k.strip()],
            "context": current_theme_dict.get("context", []),
            "up": [k.strip() for k in up_text.splitlines() if k.strip()],
            "down": [k.strip() for k in down_text.splitlines() if k.strip()],
        }
        st.session_state["dictionaries"] = dictionaries
        try:
            save_dictionaries(DICTIONARY_PATH, dictionaries)
            st.success("Dictionnaire enregistre.")
        except Exception as exc:
            LOGGER.exception("Ecriture dictionnaire impossible (enregistrement): %s", exc)
            st.warning(f"Ecriture du fichier dictionnaire impossible ({exc}).")

    if st.button("Reset dictionnaires par defaut", width="stretch"):
        st.session_state["dictionaries"] = clone_dictionaries(DEFAULT_DICTIONARIES)
        st.session_state["theme"] = sorted(DEFAULT_DICTIONARIES.keys())[0]
        try:
            save_dictionaries(DICTIONARY_PATH, st.session_state["dictionaries"])
        except Exception as exc:
            LOGGER.exception("Ecriture dictionnaire impossible (reset): %s", exc)
            st.warning(f"Ecriture du fichier dictionnaire impossible ({exc}).")
        st.rerun()

theme_dict = normalize_theme_dictionary(st.session_state["dictionaries"].get(theme, empty_theme_dictionary()))
concept_norm = prepare_keywords(theme_dict["concept"])
up_norm = prepare_keywords(theme_dict["up"])
down_norm = prepare_keywords(theme_dict["down"])

if not concept_norm:
    st.info("Ajoute au moins un mot-cle dans `Concept` pour le theme selectionne.")
    st.stop()

tagging_cache = st.session_state.setdefault("_tagged_theme_cache", {})
cache_payload = {
    "data_signature": data_signature,
    "theme": theme,
    "concept": concept_norm,
    "up": up_norm,
    "down": down_norm,
}
tag_cache_key = json.dumps(cache_payload, ensure_ascii=False, sort_keys=True)

if tag_cache_key not in tagging_cache:
    try:
        tagging_cache[tag_cache_key] = add_tagging_columns_hier(
            df_base.copy(),
            title_col=title_col,
            concept_norm=concept_norm,
            context_norm=[],
            up_norm=up_norm,
            down_norm=down_norm,
            title_norm_col="_title_norm",
        )
    except Exception as exc:
        stop_with_error_log("Erreur pendant le tagging des titres.", "add_tagging_columns_hier", exc)
    while len(tagging_cache) > 2:
        oldest_key = next(iter(tagging_cache.keys()))
        tagging_cache.pop(oldest_key, None)

df_tagged = tagging_cache[tag_cache_key]

min_date_ts = df_tagged["_date"].min()
max_date_ts = df_tagged["_date"].max()
default_start_ts = max_date_ts - pd.DateOffset(years=2)
if default_start_ts < min_date_ts:
    default_start_ts = min_date_ts
try:
    date_start, date_end = st.sidebar.slider(
        "Periode",
        min_value=min_date_ts.to_pydatetime(),
        max_value=max_date_ts.to_pydatetime(),
        value=(default_start_ts.to_pydatetime(), max_date_ts.to_pydatetime()),
    )
except Exception as exc:
    stop_with_error_log("Erreur pendant la creation du filtre de periode.", "sidebar.slider Periode", exc)

df_period = df_tagged[(df_tagged["_date"] >= pd.Timestamp(date_start)) & (df_tagged["_date"] <= pd.Timestamp(date_end))].copy()
if df_period.empty:
    st.warning("Aucune donnee dans la periode selectionnee.")
    st.stop()

all_channels = sorted(c for c in df_period["_channel"].dropna().unique().tolist() if str(c).strip())
if not all_channels:
    all_channels = ["(sans chaine)"]

selected_channels = st.sidebar.multiselect("Filtre chaines", options=all_channels, default=all_channels)
if not selected_channels:
    st.warning("Selectionne au moins une chaine.")
    st.stop()

df_filtered = df_period[df_period["_channel"].isin(selected_channels)].copy()
if df_filtered.empty:
    st.warning("Aucune ligne apres filtre chaine.")
    st.stop()

try:
    stats = aggregate_by_period(df_filtered, frequency=frequency)
    desc = build_descriptive_table(stats, df_filtered)
    top_channels = build_top_channels(df_period)
except Exception as exc:
    stop_with_error_log("Erreur pendant le calcul des indicateurs.", "aggregate/build tables", exc)

match_col = "is_match"
freq_col = "frequency"
occ_concept_total = int(df_filtered.loc[df_filtered[match_col] == 1, "occ_concept"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Titres analyses", f"{len(df_filtered):,}")
k2.metric("Titres matches", f"{int(df_filtered[match_col].sum()):,}")
k3.metric("Occurrences concept", f"{occ_concept_total:,}")
k4.metric("Frequence moyenne", f"{stats[freq_col].mean():.3f}")

st.subheader("Frequence du theme")
fig_freq = px.line(
    stats,
    x="period_start",
    y=freq_col,
    markers=True,
    render_mode="svg",
    title=f"Frequence ({frequency.lower()}) - theme '{theme}'",
    labels={"period_start": "Date", freq_col: "Part de titres matches"},
)
apply_time_axis_controls(fig_freq)
fig_freq.update_layout(height=420)
st.plotly_chart(fig_freq, width="stretch")

st.subheader("Volumes")
fig_vol = px.line(
    stats,
    x="period_start",
    y=["matched_titles", "occurrences_concept"],
    markers=True,
    render_mode="svg",
    title=f"Volumes ({frequency.lower()}) - theme '{theme}'",
    labels={"period_start": "Date", "value": "Volume", "variable": "Serie"},
)
apply_time_axis_controls(fig_vol)
fig_vol.update_layout(height=420)
st.plotly_chart(fig_vol, width="stretch")

if up_norm or down_norm:
    st.subheader("Sens du signal")
    signal_series = st.multiselect(
        "Series a afficher",
        options=["net_signal", "up_titles", "down_titles"],
        default=["net_signal"],
        help="Selectionne les series pour simplifier la lecture du signal.",
    )
    if not signal_series:
        signal_series = ["net_signal"]
    fig_signal = px.bar(
        stats,
        x="period_start",
        y=signal_series,
        barmode="group",
        title=f"Sens du signal ({frequency.lower()}) - theme '{theme}'",
        labels={"period_start": "Date", "value": "Titres", "variable": "Indicateur"},
    )
    apply_time_axis_controls(fig_signal)
    fig_signal.update_layout(height=420)
    st.plotly_chart(fig_signal, width="stretch")

st.subheader("Statistiques descriptives")
st.dataframe(desc, width="stretch")

if has_source_channel:
    st.subheader("Top 10 chaines (sur la periode)")
    st.caption("Calcule sur la periode choisie, avant application du filtre chaine.")
    st.dataframe(top_channels, width="stretch")
    if not top_channels.empty:
        fig_top = px.bar(
            top_channels,
            x="_channel",
            y="matched_titles",
            title="Top chaines par titres matches",
            labels={"_channel": "Chaine", "matched_titles": "Titres matches"},
        )
        fig_top.update_layout(height=400)
        st.plotly_chart(fig_top, width="stretch")

st.subheader("Apercu des titres matches")
preview_cols = [
    c
    for c in [
        "_date",
        "_channel",
        title_col,
        "occ_concept",
        "occ_up",
        "occ_down",
        "direction",
        "source_file",
    ]
    if c in df_filtered.columns
]
st.dataframe(
    df_filtered[df_filtered[match_col] == 1]
    .sort_values(["occ_concept", "_date"], ascending=[False, False])
    .head(300)[preview_cols],
    width="stretch",
)
