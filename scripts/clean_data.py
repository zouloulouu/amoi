#!/usr/bin/env python
"""Build clean parquet snapshots from immutable raw parquet files.

Pipeline design:
1) technical clean (types, harmonization, robust schema handling)
2) analytical layer (monthly aggregates + quality controls)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PIPELINE_VERSION = "clean_v2"

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

SOURCE_CANDIDATES = {
    "id": ["id", "indice"],
    "chaine": ["chaine", "channel", "_channel", "raw_channel"],
    "date": ["date", "_date"],
    "date_diffusion": ["date_diffusion", "date", "raw_date"],
    "heure_diffusion": ["heure_diffusion", "heure", "time", "horaire", "raw_time"],
    "duree": ["duree", "duration"],
    "month": ["month"],
    "titre_propre": ["titre_propre", "title", "titre", "libelle", "intitule"],
    "titre_collection": ["titre_collection"],
    "titre_programme": ["titre_programme"],
    "genre": ["genre"],
    "url_notice": ["url_notice", "url"],
    "inflation_extended": ["inflation_extended"],
}

TARGET_SCHEMA_COLUMNS = [
    "id",
    "id_source",
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
    "weekday",
    "hour",
    "titre_propre",
    "titre_propre_raw",
    "titre_collection",
    "titre_collection_raw",
    "titre_programme",
    "titre_programme_raw",
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
    "flag_duree_invalide",
    "flag_inflation_invalide",
    "flag_missing_essential",
    "flag_missing_useful",
    "flag_missing_accessory",
    "title",
    "_title_norm",
    "_date",
    "_channel",
    "raw_date",
    "raw_time",
    "raw_channel",
    "row_hash",
    "pipeline_version",
    "cleaned_at_utc",
]

ESSENTIAL_COLUMNS = ["date_diffusion", "heure_diffusion", "titre_propre", "inflation_extended"]
USEFUL_COLUMNS = ["titre_programme", "titre_collection", "url_notice"]
ACCESSORY_COLUMNS = ["genre", "duree"]

LOCAL_DEDUP_SUBSET = ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"]
GLOBAL_DEDUP_SUBSET = ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"]

CHANNEL_MAPPING = {
    "bfmtv": "BFMTV",
    "bfm": "BFMTV",
    "france2": "France 2",
    "fr2": "France 2",
    "f2": "France 2",
    "france3": "France 3",
    "fr3": "France 3",
    "f3": "France 3",
    "franceinter": "France Inter",
    "franceinterinformation": "France Inter",
    "tf1": "TF1",
}

INFLATION_TRUE_TOKENS = {"1", "true", "vrai", "yes", "oui", "y", "t"}
INFLATION_FALSE_TOKENS = {"0", "false", "faux", "no", "non", "n", "f"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    missing = [c for c in required_columns if c not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return missing


def strip_accents(text: Any) -> str:
    if text is None or pd.isna(text):
        return ""
    value = str(text)
    return "".join(c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn")


def clean_title(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = strip_accents(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_duration_to_seconds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None

    if re.fullmatch(r"\d+(\.\d+)?", raw):
        try:
            return float(raw)
        except ValueError:
            return None

    parts = raw.split(":")
    if len(parts) not in (2, 3, 4):
        return None

    try:
        if len(parts) == 2:
            mm, ss = int(parts[0]), int(parts[1])
            return float(mm * 60 + ss)
        hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
        centisec = int(parts[3]) if len(parts) == 4 else 0
    except ValueError:
        return None

    return float(hh * 3600 + mm * 60 + ss + (centisec / 100.0))


def canon_colname(name: str) -> str:
    c = normalize_text(str(name).replace("\u00a0", " "))
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^a-z0-9_]", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def harmonize_channel(value: Any) -> str:
    raw = normalize_text(value)
    if not raw:
        return ""
    compact = clean_title(raw).replace(" ", "")
    if compact in CHANNEL_MAPPING:
        return CHANNEL_MAPPING[compact]
    if "franceinter" in compact:
        return "France Inter"
    return str(value).strip()


def classify_content(row: pd.Series) -> str:
    text = " ".join(
        [
            clean_title(row.get("titre_propre")),
            clean_title(row.get("titre_programme")),
            clean_title(row.get("titre_collection")),
        ]
    ).strip()
    if "plateau" in text:
        return "plateau"
    if re.search(r"\b(?:programme du|emission du|edition du)\b", text):
        return "edition_complete"
    return "sujet"


def standardize_emission(row: pd.Series) -> str:
    text = " ".join(
        [
            clean_title(row.get("titre_propre")),
            clean_title(row.get("titre_programme")),
            clean_title(row.get("titre_collection")),
        ]
    ).strip()
    if re.search(r"\b(?:20heures|20 heures|20h)\b", text):
        return "20 heures"
    if re.search(r"\bpremiere edition\b", text):
        return "Premiere edition"
    return ""


def normalize_clock(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip().lower().replace("h", ":")
    raw = re.sub(r"[^0-9:]", "", raw)
    if raw.isdigit() and len(raw) == 4:
        return f"{raw[:2]}:{raw[2:]}:00"
    if re.match(r"^\d{1,2}:\d{2}$", raw):
        return raw + ":00"
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", raw):
        return raw
    if re.match(r"^\d{1,2}:\d{2}:\d{2}:\d{2}$", raw):
        return ":".join(raw.split(":")[:3])
    return ""


def parse_datetime(date_series: pd.Series, time_series: pd.Series | None) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce", dayfirst=True)
    if time_series is None:
        return dt
    fixed_time = time_series.map(normalize_clock)
    combo = pd.to_datetime(dt.dt.date.astype(str) + " " + fixed_time, errors="coerce")
    return combo.where(combo.notna(), dt)


def parse_inflation_extended(value: Any) -> tuple[int | pd._libs.missing.NAType, bool]:
    if value is None or pd.isna(value):
        return pd.NA, False
    if isinstance(value, (bool, np.bool_)):
        return (1 if bool(value) else 0), False
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return int(value), False
        return pd.NA, True
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return pd.NA, False
        if value in (0.0, 1.0):
            return int(value), False
        return pd.NA, True

    token = normalize_text(value)
    if token in INFLATION_TRUE_TOKENS:
        return 1, False
    if token in INFLATION_FALSE_TOKENS:
        return 0, False
    if re.fullmatch(r"\d+(\.\d+)?", token):
        try:
            numeric = float(token)
            if numeric in (0.0, 1.0):
                return int(numeric), False
        except ValueError:
            pass
    return pd.NA, True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def choose_source_columns(schema_names: list[str]) -> dict[str, str] | None:
    source_by_canon = {canon_colname(c): c for c in schema_names}
    mapping: dict[str, str] = {}

    for target, candidates in SOURCE_CANDIDATES.items():
        chosen = next((source_by_canon[c] for c in candidates if c in source_by_canon), None)
        if chosen:
            mapping[target] = chosen

    source_title = next((source_by_canon[c] for c in TITLE_CANDIDATES if c in source_by_canon), None)
    source_date = next((source_by_canon[c] for c in DATE_CANDIDATES if c in source_by_canon), None)
    source_time = next((source_by_canon[c] for c in TIME_CANDIDATES if c in source_by_canon), None)
    source_channel = next((source_by_canon[c] for c in CHANNEL_CANDIDATES if c in source_by_canon), None)

    if source_title and "titre_propre" not in mapping:
        mapping["titre_propre"] = source_title
    if source_date and "date_diffusion" not in mapping:
        mapping["date_diffusion"] = source_date
    if source_date and "date" not in mapping:
        mapping["date"] = source_date
    if source_time and "heure_diffusion" not in mapping:
        mapping["heure_diffusion"] = source_time
    if source_channel and "chaine" not in mapping:
        mapping["chaine"] = source_channel

    if "titre_propre" not in mapping or ("date_diffusion" not in mapping and "date" not in mapping):
        return None
    return mapping


def _standardize_string_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        series = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        df[col] = series.where(series.ne(""), pd.NA)


def _build_row_hash(df: pd.DataFrame) -> pd.Series:
    key = (
        df["clean_titre"].fillna("")
        + "|"
        + pd.to_datetime(df["_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        + "|"
        + df["_channel"].fillna("")
        + "|"
        + df["source_file"].fillna("")
    )
    rank = key.groupby(key).cumcount().astype("string")
    return (
        key.astype("string")
        .str.cat(rank, sep="|")
        .map(lambda v: hashlib.sha256(str(v).encode("utf-8")).hexdigest())
        .astype("string")
    )


def clean_file(
    df: pd.DataFrame,
    source_name: str,
    normalize_channels: bool,
    input_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out["source_file"] = source_name

    created_columns = ensure_columns(out, TARGET_SCHEMA_COLUMNS)

    _standardize_string_columns(
        out,
        [
            "chaine",
            "source_file",
            "date",
            "date_diffusion",
            "heure_diffusion",
            "duree",
            "month",
            "titre_propre",
            "titre_collection",
            "titre_programme",
            "genre",
            "url_notice",
            "inflation_extended",
        ],
    )

    out["titre_propre_raw"] = out["titre_propre"]
    out["titre_programme_raw"] = out["titre_programme"]
    out["titre_collection_raw"] = out["titre_collection"]

    out["raw_date"] = out["date_diffusion"].where(out["date_diffusion"].notna(), out["date"])
    out["raw_time"] = out["heure_diffusion"]
    out["raw_channel"] = out["chaine"]

    out["date_diffusion"] = pd.to_datetime(out["date_diffusion"], errors="coerce", dayfirst=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["date"] = out["date"].where(out["date"].notna(), out["date_diffusion"])
    out["date_diffusion"] = out["date_diffusion"].where(out["date_diffusion"].notna(), out["date"])

    out["heure_diffusion"] = out["heure_diffusion"].map(normalize_clock).astype("string")
    out["heure_diffusion"] = out["heure_diffusion"].where(out["heure_diffusion"].ne(""), pd.NA)

    out["_date"] = parse_datetime(out["raw_date"], out["raw_time"])
    fallback_date = pd.to_datetime(out["date_diffusion"], errors="coerce")
    out["_date"] = out["_date"].where(out["_date"].notna(), fallback_date)

    if normalize_channels:
        out["chaine"] = out["chaine"].map(harmonize_channel).astype("string")
        out["chaine"] = out["chaine"].where(out["chaine"].str.strip().ne(""), pd.NA)
    else:
        out["chaine"] = out["chaine"].astype("string")

    out["_channel"] = out["chaine"].fillna("(sans chaine)").astype("string")

    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
    base_id = "id:" + out["id"].astype("string")
    composite_id = (
        "cmp:"
        + pd.to_datetime(out["date_diffusion"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        + "|"
        + out["heure_diffusion"].fillna("")
        + "|"
        + out["titre_propre"].map(clean_title).fillna("")
    )
    out["id_source"] = np.where(out["id"].notna(), base_id, composite_id)
    out["id_source"] = pd.Series(out["id_source"], index=out.index, dtype="string")
    id_rank = out.groupby("id_source").cumcount().astype("Int64")
    out["id_source"] = np.where(id_rank > 0, out["id_source"] + "|" + id_rank.astype("string"), out["id_source"])
    out["id_source"] = pd.Series(out["id_source"], index=out.index, dtype="string")

    parsed = out["inflation_extended"].map(parse_inflation_extended)
    out["inflation_extended"] = pd.Series([x[0] for x in parsed], index=out.index, dtype="Int64")
    out["flag_inflation_invalide"] = pd.Series([1 if x[1] else 0 for x in parsed], index=out.index, dtype="Int8")

    parsed_duree = out["duree"].map(parse_duration_to_seconds)
    existing_duree = pd.to_numeric(out["duree_sec"], errors="coerce")
    out["duree_sec"] = existing_duree.where(existing_duree.notna(), parsed_duree)
    duree_present = out["duree"].fillna("").astype("string").str.strip().ne("")
    out["flag_duree_invalide"] = (duree_present & out["duree_sec"].isna()).astype("Int8")

    out["clean_titre"] = out["titre_propre"].map(clean_title).astype("string")
    out["clean_programme"] = out["titre_programme"].map(clean_title).astype("string")
    out["title"] = out["titre_propre"].fillna(out["titre_programme"]).fillna(out["titre_collection"]).astype("string")
    out["_title_norm"] = out["title"].map(clean_title).astype("string")

    out["type_contenu"] = out.apply(classify_content, axis=1).astype("string")
    out["emission_std"] = out.apply(standardize_emission, axis=1).astype("string")
    out["emission_std"] = out["emission_std"].where(out["emission_std"].str.strip().ne(""), pd.NA)

    base_date = pd.to_datetime(out["date_diffusion"], errors="coerce")
    out["annee"] = base_date.dt.year.astype("Int64")
    out["mois"] = base_date.dt.month.astype("Int64")
    out["ym"] = base_date.dt.strftime("%Y-%m").astype("string")
    out["ym"] = out["ym"].where(base_date.notna(), pd.NA)
    out["month"] = out["month"].where(out["month"].notna(), out["ym"]).astype("string")
    out["weekday"] = base_date.dt.dayofweek.astype("Int64")
    out["hour"] = pd.to_numeric(out["heure_diffusion"].astype("string").str.split(":").str[0], errors="coerce").astype(
        "Int64"
    )

    out["has_url_notice"] = (
        out["url_notice"].notna() & out["url_notice"].astype("string").str.strip().ne("")
    ).astype("Int8")
    out["has_titre_programme"] = (
        out["titre_programme"].notna() & out["titre_programme"].astype("string").str.strip().ne("")
    ).astype("Int8")
    out["has_titre_collection"] = (
        out["titre_collection"].notna() & out["titre_collection"].astype("string").str.strip().ne("")
    ).astype("Int8")

    out["flag_missing_essential"] = out[ESSENTIAL_COLUMNS].isna().any(axis=1).astype("Int8")
    out["flag_missing_useful"] = out[USEFUL_COLUMNS].isna().any(axis=1).astype("Int8")
    out["flag_missing_accessory"] = out[ACCESSORY_COLUMNS].isna().any(axis=1).astype("Int8")

    dedup_subset = [c for c in LOCAL_DEDUP_SUBSET if c in out.columns]
    before_local_dedup = len(out)
    local_dup_by_channel: dict[str, int] = {}
    if dedup_subset:
        local_dup_mask = out.duplicated(subset=dedup_subset, keep="first")
        local_dup_by_channel = (
            out.loc[local_dup_mask, "chaine"].fillna("(sans chaine)").astype("string").value_counts().to_dict()
        )
        out = out.loc[~local_dup_mask].copy()
    after_local_dedup = len(out)

    out["pipeline_version"] = PIPELINE_VERSION
    out["cleaned_at_utc"] = now_utc().isoformat()
    out["row_hash"] = _build_row_hash(out)

    ensure_columns(out, TARGET_SCHEMA_COLUMNS)
    out = out[TARGET_SCHEMA_COLUMNS]

    file_report = {
        "file": source_name,
        "rows_input": int(before_local_dedup),
        "rows_after_local_clean": int(after_local_dedup),
        "dropped_duplicates_local": int(before_local_dedup - after_local_dedup),
        "dropped_duplicates_local_by_channel": local_dup_by_channel,
        "invalid_date_rows": int(pd.to_datetime(out["_date"], errors="coerce").isna().sum()),
        "invalid_duration_rows": int(out["flag_duree_invalide"].sum()),
        "invalid_inflation_rows": int(out["flag_inflation_invalide"].sum()),
        "missing_essential_rows": int(out["flag_missing_essential"].sum()),
        "created_columns": created_columns,
        "columns_before": input_columns,
        "columns_after": list(out.columns),
        "date_min": str(pd.to_datetime(out["_date"], errors="coerce").min()) if not out.empty else None,
        "date_max": str(pd.to_datetime(out["_date"], errors="coerce").max()) if not out.empty else None,
    }
    return out, file_report


def process_file(path: Path, normalize_channels: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema_names = pq.ParquetFile(path).schema.names
    mapping = choose_source_columns(schema_names)
    if not mapping:
        raise ValueError("Missing required columns: title/date")

    selected_columns = list(dict.fromkeys(mapping.values()))
    raw = pd.read_parquet(path, columns=selected_columns)

    base = pd.DataFrame(index=raw.index)
    for target_col, source_col in mapping.items():
        base[target_col] = raw[source_col]

    cleaned, report = clean_file(
        base,
        source_name=path.name,
        normalize_channels=normalize_channels,
        input_columns=list(raw.columns),
    )
    report["rows_input"] = int(len(raw))
    return cleaned, report


def build_quality_report(
    df: pd.DataFrame,
    local_dup_by_channel: dict[str, int],
    global_dup_by_channel: dict[str, int],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "chaine",
                "nb_lignes",
                "dup_local_suppr",
                "dup_global_suppr",
                "tx_na_date_diffusion",
                "tx_na_heure_diffusion",
                "tx_na_titre_propre",
                "tx_na_inflation_extended",
                "part_plateau",
                "part_edition_complete",
                "part_sujet",
                "part_inflation_1",
            ]
        )

    work = df.copy()
    work["chaine_group"] = work["chaine"].fillna("(sans chaine)").astype("string")
    grp = work.groupby("chaine_group", dropna=False)

    out = grp.size().rename("nb_lignes").to_frame()
    out["tx_na_date_diffusion"] = grp["date_diffusion"].apply(lambda s: float(s.isna().mean()))
    out["tx_na_heure_diffusion"] = grp["heure_diffusion"].apply(lambda s: float(s.isna().mean()))
    out["tx_na_titre_propre"] = grp["titre_propre"].apply(lambda s: float(s.isna().mean()))
    out["tx_na_inflation_extended"] = grp["inflation_extended"].apply(lambda s: float(s.isna().mean()))
    out["part_plateau"] = grp["type_contenu"].apply(lambda s: float((s == "plateau").mean()))
    out["part_edition_complete"] = grp["type_contenu"].apply(lambda s: float((s == "edition_complete").mean()))
    out["part_sujet"] = grp["type_contenu"].apply(lambda s: float((s == "sujet").mean()))
    out["part_inflation_1"] = grp["inflation_extended"].apply(lambda s: float((s == 1).mean()))
    out["dup_local_suppr"] = out.index.map(lambda c: int(local_dup_by_channel.get(str(c), 0)))
    out["dup_global_suppr"] = out.index.map(lambda c: int(global_dup_by_channel.get(str(c), 0)))

    out = out.reset_index().rename(columns={"chaine_group": "chaine"})
    return out


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
    work["ym"] = work["ym"].astype("string")
    work["inflation_bin"] = (work["inflation_extended"] == 1).astype("int8")
    work["duree_num"] = pd.to_numeric(work["duree_sec"], errors="coerce").fillna(0.0)
    work["duree_inflation_component"] = np.where(work["inflation_bin"] == 1, work["duree_num"], 0.0)

    out = (
        work.groupby(["chaine", "ym"], dropna=False)
        .agg(
            nb_total=("source_file", "size"),
            nb_inflation=("inflation_bin", "sum"),
            duree_totale=("duree_num", "sum"),
            duree_inflation=("duree_inflation_component", "sum"),
        )
        .reset_index()
    )

    out["part_inflation_nb"] = np.where(out["nb_total"] > 0, out["nb_inflation"] / out["nb_total"], 0.0)
    out["part_inflation_duree"] = np.where(
        out["duree_totale"] > 0,
        out["duree_inflation"] / out["duree_totale"],
        0.0,
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate versioned clean parquet snapshots from raw parquet files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data") / "raw", help="Raw parquet directory.")
    parser.add_argument("--output-root", type=Path, default=Path("data") / "clean", help="Clean snapshots root.")
    parser.add_argument(
        "--output-version",
        type=str,
        default="",
        help="Optional version name. If omitted: vYYYYMMDD_HHMMSS_utc.",
    )
    parser.add_argument("--normalize-channels", action="store_true", help="Apply canonical channel mapping.")
    parser.add_argument("--set-current", action="store_true", help="Write output version into data/clean/CURRENT.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    raw_files = sorted(args.input_dir.glob("*.parquet"))
    if not raw_files:
        raise SystemExit(f"No parquet files found in: {args.input_dir}")

    version = args.output_version.strip() or f"v{now_utc().strftime('%Y%m%d_%H%M%S')}_utc"
    output_dir = args.output_root / version
    output_dir.mkdir(parents=True, exist_ok=False)
    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    manifest_inputs: list[dict[str, Any]] = []
    quality_files: list[dict[str, Any]] = []
    cleaned_by_file: dict[str, pd.DataFrame] = {}
    reports_by_file: dict[str, dict[str, Any]] = {}
    total_rows_input = 0

    for index, path in enumerate(raw_files, start=1):
        print(f"[{index}/{len(raw_files)}] Cleaning {path.name}...", flush=True)
        cleaned_df, file_report = process_file(path, normalize_channels=args.normalize_channels)
        total_rows_input += int(file_report["rows_input"])
        cleaned_by_file[path.name] = cleaned_df
        reports_by_file[path.name] = file_report
        manifest_inputs.append(
            {
                "file": path.name,
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )

    combined_before_global = pd.concat([cleaned_by_file[p.name] for p in raw_files], ignore_index=True, sort=False)
    global_subset = [c for c in GLOBAL_DEDUP_SUBSET if c in combined_before_global.columns]
    global_dup_by_source: dict[str, int] = {}
    global_dup_by_channel: dict[str, int] = {}

    combined = combined_before_global.copy()
    if global_subset:
        global_dup_mask = combined.duplicated(subset=global_subset, keep="first")
        if global_dup_mask.any():
            dup_rows = combined.loc[global_dup_mask].copy()
            global_dup_by_source = dup_rows["source_file"].value_counts().to_dict()
            global_dup_by_channel = dup_rows["chaine"].fillna("(sans chaine)").astype("string").value_counts().to_dict()
        combined = combined.loc[~global_dup_mask].copy()
    dropped_global_dedup_total = int(len(combined_before_global) - len(combined))

    combined["row_hash"] = _build_row_hash(combined)

    local_dup_counter: Counter[str] = Counter()
    for report in reports_by_file.values():
        local_dup_counter.update(report.get("dropped_duplicates_local_by_channel", {}))

    total_rows_output = 0
    for path in raw_files:
        source_name = path.name
        source_df = combined[combined["source_file"] == source_name].copy()
        out_name = f"{path.stem}_clean.parquet"
        out_path = output_dir / out_name
        source_df.to_parquet(out_path, index=False)

        report = reports_by_file[source_name]
        report["rows_output"] = int(len(source_df))
        report["dropped_duplicates_global"] = int(global_dup_by_source.get(source_name, 0))
        quality_files.append(report)
        total_rows_output += int(len(source_df))

    quality_by_channel = build_quality_report(
        combined,
        local_dup_by_channel=dict(local_dup_counter),
        global_dup_by_channel=global_dup_by_channel,
    )
    monthly_analysis = build_monthly_analysis(combined)
    combined.to_parquet(analytics_dir / "all_channels_clean.parquet", index=False)
    quality_by_channel.to_parquet(analytics_dir / "quality_by_channel.parquet", index=False)
    monthly_analysis.to_parquet(analytics_dir / "monthly_analysis.parquet", index=False)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": now_utc().isoformat(),
        "input_dir": str(args.input_dir.as_posix()),
        "output_dir": str(output_dir.as_posix()),
        "version": version,
        "inputs": manifest_inputs,
    }
    quality_report = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": now_utc().isoformat(),
        "version": version,
        "files": quality_files,
        "summary": {
            "files": len(quality_files),
            "rows_input": int(total_rows_input),
            "rows_after_local_clean": int(len(combined_before_global)),
            "rows_output": int(total_rows_output),
            "rows_retention_ratio": (float(total_rows_output / total_rows_input) if total_rows_input else 0.0),
            "dropped_duplicates_local_total": int(total_rows_input - len(combined_before_global)),
            "dropped_duplicates_global_total": dropped_global_dedup_total,
            "missing_essential_rows_total": int(combined["flag_missing_essential"].sum()),
            "invalid_duration_rows_total": int(combined["flag_duree_invalide"].sum()),
            "invalid_inflation_rows_total": int(combined["flag_inflation_invalide"].sum()),
        },
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.set_current:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "CURRENT").write_text(version, encoding="utf-8")

    print(f"Clean snapshot created: {output_dir}")
    print(f"Rows input={total_rows_input} output={total_rows_output}")


if __name__ == "__main__":
    main()
