#!/usr/bin/env python
"""Build clean parquet snapshots from immutable raw parquet files.

This script is intentionally focused on a stable V1 pipeline:
- validate required columns per file
- normalize title/date/time/channel fields
- build canonical columns used by Streamlit
- export *_clean.parquet files into a versioned output folder
- produce manifest and quality report JSON files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

PIPELINE_VERSION = "clean_v1"

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

CHANNEL_MAPPING = {
    "bfmtv": "BFMTV",
    "bfm": "BFMTV",
    "france2": "France 2",
    "fr2": "France 2",
    "f2": "France 2",
    "tf1": "TF1",
    "france3": "France 3",
    "fr3": "France 3",
    "franceinter": "France Inter",
}

LOCAL_DEDUP_SUBSET = ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"]
GLOBAL_DEDUP_SUBSET = ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
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
    base = normalize_text(value)
    if not base:
        return ""
    no_accents = strip_accents(base)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", no_accents)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def parse_duration_to_seconds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    parts = raw.split(":")
    if len(parts) not in (3, 4):
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
        centisec = int(parts[3]) if len(parts) == 4 else 0
    except ValueError:
        return None
    return float(h * 3600 + m * 60 + s + (centisec / 100.0))


def canon_colname(name: str) -> str:
    c = normalize_text(str(name).replace("\u00a0", " "))
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^a-z0-9_]", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def normalize_channel(value: Any) -> str:
    s = clean_title(value)
    if not s:
        return ""
    compact = s.replace(" ", "")
    if compact in CHANNEL_MAPPING:
        return CHANNEL_MAPPING[compact]
    if "franceinter" in compact:
        return "France Inter"
    return str(value).strip()


def classify_content(titre_propre: Any, titre_programme: Any, titre_collection: Any) -> str:
    text = " ".join([clean_title(titre_propre), clean_title(titre_programme), clean_title(titre_collection)]).strip()
    if "plateau" in text:
        return "plateau"
    if re.search(r"\b(?:programme du|emission du|edition du)\b", text):
        return "edition_complete"
    return "sujet"


def standardize_emission(titre_propre: Any, titre_programme: Any, titre_collection: Any) -> str:
    text = " ".join([clean_title(titre_propre), clean_title(titre_programme), clean_title(titre_collection)]).strip()
    if re.search(r"\b(?:20heures|20 heures|20h)\b", text):
        return "20 heures"
    if re.search(r"\bpremiere edition\b", text):
        return "Premiere edition"
    return ""


def normalize_clock(value: str) -> str:
    if not value or value == "nan":
        return ""
    raw = value.strip().lower().replace("h", ":")
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
    fixed = time_series.astype(str).map(normalize_clock)
    combo = pd.to_datetime(dt.dt.date.astype(str) + " " + fixed, errors="coerce")
    return combo.where(combo.notna(), dt)


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


def process_file(path: Path, normalize_channels: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema_names = pq.ParquetFile(path).schema.names
    mapping = choose_source_columns(schema_names)
    if not mapping:
        raise ValueError("Missing required columns: title/date")

    selected_columns = list(dict.fromkeys(mapping.values()))
    raw = pd.read_parquet(path, columns=selected_columns)

    out = pd.DataFrame(index=raw.index)
    for target_col, source_col in mapping.items():
        out[target_col] = raw[source_col]
    out["source_file"] = path.name

    created_columns = ensure_columns(out, TARGET_SCHEMA_COLUMNS)

    out["raw_date"] = out["date_diffusion"].where(out["date_diffusion"].notna(), out["date"]).astype("string")
    out["raw_time"] = out["heure_diffusion"].astype("string")
    out["raw_channel"] = out["chaine"].astype("string")

    for col in [
        "chaine",
        "source_file",
        "heure_diffusion",
        "duree",
        "month",
        "titre_propre",
        "titre_collection",
        "titre_programme",
        "genre",
        "url_notice",
    ]:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace("", pd.NA)

    out["date_diffusion"] = pd.to_datetime(out["date_diffusion"], errors="coerce", dayfirst=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["date"] = out["date"].where(out["date"].notna(), out["date_diffusion"])
    out["date_diffusion"] = out["date_diffusion"].where(out["date_diffusion"].notna(), out["date"])
    parsed_dt = parse_datetime(out["raw_date"], out["raw_time"])
    out["_date"] = parsed_dt.where(parsed_dt.notna(), out["date"])

    if normalize_channels:
        out["chaine"] = out["chaine"].map(normalize_channel).astype("string")
    else:
        out["chaine"] = out["chaine"].astype("string")
    out["_channel"] = out["chaine"].fillna("(sans chaine)").astype("string")

    out["inflation_extended"] = pd.to_numeric(out["inflation_extended"], errors="coerce").astype("Int64")
    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")

    parsed_duree = out["duree"].map(parse_duration_to_seconds)
    existing_duree = pd.to_numeric(out["duree_sec"], errors="coerce")
    out["duree_sec"] = existing_duree.where(existing_duree.notna(), parsed_duree)

    out["clean_titre"] = out["titre_propre"].map(clean_title).astype("string")
    out["clean_programme"] = out["titre_programme"].map(clean_title).astype("string")
    text_combo = (
        out["clean_titre"].fillna("")
        + " "
        + out["clean_programme"].fillna("")
        + " "
        + out["titre_collection"].map(clean_title).fillna("")
    ).str.strip()
    is_plateau = text_combo.str.contains(r"\bplateau\b", regex=True, na=False)
    is_edition = text_combo.str.contains(r"\b(?:programme du|emission du|edition du)\b", regex=True, na=False)

    out["type_contenu"] = pd.Series("sujet", index=out.index, dtype="string")
    out.loc[is_edition, "type_contenu"] = "edition_complete"
    out.loc[is_plateau, "type_contenu"] = "plateau"

    out["emission_std"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[text_combo.str.contains(r"\b(?:20heures|20 heures|20h)\b", regex=True, na=False), "emission_std"] = (
        "20 heures"
    )
    out.loc[text_combo.str.contains(r"\bpremiere edition\b", regex=True, na=False), "emission_std"] = (
        "Premiere edition"
    )

    out["title"] = out["titre_propre"].fillna(out["titre_programme"]).fillna(out["titre_collection"]).astype("string")
    out["_title_norm"] = out["title"].map(clean_title).astype("string")

    base_date = pd.to_datetime(out["date"], errors="coerce")
    out["annee"] = base_date.dt.year.astype("Int64")
    out["mois"] = base_date.dt.month.astype("Int64")
    out["ym"] = base_date.dt.strftime("%Y-%m").astype("string")
    out["ym"] = out["ym"].where(base_date.notna(), pd.NA)
    out["month"] = out["month"].where(out["month"].notna(), out["ym"]).astype("string")

    out["has_url_notice"] = (
        out["url_notice"].notna() & out["url_notice"].astype("string").str.strip().ne("")
    ).astype("Int8")
    out["has_titre_programme"] = (
        out["titre_programme"].notna() & out["titre_programme"].astype("string").str.strip().ne("")
    ).astype("Int8")
    out["has_titre_collection"] = (
        out["titre_collection"].notna() & out["titre_collection"].astype("string").str.strip().ne("")
    ).astype("Int8")

    dedup_subset = [c for c in LOCAL_DEDUP_SUBSET if c in out.columns]
    before_local_dedup = len(out)
    if dedup_subset:
        out = out.drop_duplicates(subset=dedup_subset, keep="first").copy()
    after_local_dedup = len(out)

    dedup_key = (
        out["clean_titre"].fillna("")
        + "|"
        + out["_date"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        + "|"
        + out["_channel"].fillna("")
        + "|"
        + out["source_file"].fillna("")
    )
    dedup_rank = dedup_key.groupby(dedup_key).cumcount().astype("string")
    out["row_hash"] = (
        dedup_key.astype("string")
        .str.cat(dedup_rank, sep="|")
        .map(lambda s: hashlib.sha256(str(s).encode("utf-8")).hexdigest())
        .astype("string")
    )
    out["pipeline_version"] = PIPELINE_VERSION
    out["cleaned_at_utc"] = now_utc().isoformat()

    ensure_columns(out, TARGET_SCHEMA_COLUMNS)
    out = out[TARGET_SCHEMA_COLUMNS]

    invalid_date_rows = int(pd.to_datetime(out["_date"], errors="coerce").isna().sum())
    file_report = {
        "file": path.name,
        "rows_input": int(len(raw)),
        "rows_after_local_clean": int(len(out)),
        "dropped_duplicates_local": int(before_local_dedup - after_local_dedup),
        "invalid_date_rows": invalid_date_rows,
        "created_columns": created_columns,
        "title_empty_ratio": float((out["title"].fillna("").astype(str).str.strip() == "").mean()),
        "channel_empty_ratio": float((out["_channel"].fillna("").astype(str).str.strip() == "").mean()),
        "date_min": str(pd.to_datetime(out["_date"], errors="coerce").min()) if not out.empty else None,
        "date_max": str(pd.to_datetime(out["_date"], errors="coerce").max()) if not out.empty else None,
    }
    return out, file_report


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

    manifest_inputs: list[dict[str, Any]] = []
    quality_files: list[dict[str, Any]] = []
    cleaned_by_file: dict[str, pd.DataFrame] = {}
    reports_by_file: dict[str, dict[str, Any]] = {}
    total_rows_input = 0
    total_rows_output = 0

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

    combined = pd.concat([cleaned_by_file[p.name] for p in raw_files], ignore_index=True, sort=False)
    global_subset = [c for c in GLOBAL_DEDUP_SUBSET if c in combined.columns]
    before_global_dedup = len(combined)
    if global_subset:
        combined = combined.drop_duplicates(subset=global_subset, keep="first").copy()
    dropped_global_dedup_total = int(before_global_dedup - len(combined))

    for path in raw_files:
        source_name = path.name
        source_df = combined[combined["source_file"] == source_name].copy()
        out_name = f"{path.stem}_clean.parquet"
        out_path = output_dir / out_name
        source_df.to_parquet(out_path, index=False)

        report = reports_by_file[source_name]
        report["rows_output"] = int(len(source_df))
        report["dropped_duplicates_global"] = int(report["rows_after_local_clean"] - len(source_df))
        quality_files.append(report)
        total_rows_output += int(len(source_df))

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
            "rows_input": total_rows_input,
            "rows_output": total_rows_output,
            "rows_retention_ratio": (float(total_rows_output / total_rows_input) if total_rows_input else 0.0),
            "dropped_duplicates_global_total": dropped_global_dedup_total,
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
