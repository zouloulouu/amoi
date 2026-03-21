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


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text


def canon_colname(name: str) -> str:
    c = normalize_text(str(name).replace("\u00a0", " "))
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^a-z0-9_]", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def normalize_channel(value: Any) -> str:
    s = normalize_text(str(value))
    s = s.replace("france2", "france 2").replace("fr2", "france 2").replace("f2", "france 2")
    s = s.replace("t_f_1", "tf1").replace("tf1_", "tf1")
    if "france 2" in s or s == "2":
        return "France 2"
    if "tf1" in s:
        return "TF1"
    if "france 3" in s or s == "3":
        return "France 3"
    if "france inter" in s:
        return "France Inter"
    return str(value).strip()


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
    source_title = next((source_by_canon[c] for c in TITLE_CANDIDATES if c in source_by_canon), None)
    source_date = next((source_by_canon[c] for c in DATE_CANDIDATES if c in source_by_canon), None)
    source_time = next((source_by_canon[c] for c in TIME_CANDIDATES if c in source_by_canon), None)
    source_channel = next((source_by_canon[c] for c in CHANNEL_CANDIDATES if c in source_by_canon), None)
    if not source_title or not source_date:
        return None
    selected = {"title": source_title, "date": source_date}
    if source_time:
        selected["time"] = source_time
    if source_channel:
        selected["channel"] = source_channel
    return selected


def process_file(path: Path, normalize_channels: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema_names = pq.ParquetFile(path).schema.names
    mapping = choose_source_columns(schema_names)
    if not mapping:
        raise ValueError("Missing required columns: title/date")

    selected_columns = list(dict.fromkeys(mapping.values()))
    raw = pd.read_parquet(path, columns=selected_columns)

    out = pd.DataFrame(
        {
            "title": raw[mapping["title"]].astype("string"),
            "raw_date": raw[mapping["date"]].astype("string"),
            "source_file": path.name,
        }
    )
    if "time" in mapping:
        out["raw_time"] = raw[mapping["time"]].astype("string")
    else:
        out["raw_time"] = pd.Series([""] * len(out), dtype="string")
    if "channel" in mapping:
        out["raw_channel"] = raw[mapping["channel"]].astype("string")
    else:
        out["raw_channel"] = pd.Series(["(sans chaine)"] * len(out), dtype="string")

    out["_date"] = parse_datetime(out["raw_date"], out["raw_time"])
    before_date_filter = len(out)
    out = out[out["_date"].notna()].copy()
    after_date_filter = len(out)

    out["title"] = out["title"].fillna("").astype("string").str.strip()
    out["_title_norm"] = out["title"].map(normalize_text).astype("string")

    if normalize_channels:
        out["_channel"] = out["raw_channel"].map(normalize_channel).astype("string")
    else:
        out["_channel"] = out["raw_channel"].fillna("").astype("string").str.strip()

    dedup_key = (
        out["_title_norm"].fillna("")
        + "|"
        + out["_date"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        + "|"
        + out["_channel"].fillna("")
        + "|"
        + out["source_file"].fillna("")
    )
    out["row_hash"] = dedup_key.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()).astype("string")
    before_dedup = len(out)
    out = out.drop_duplicates(subset=["row_hash"]).copy()
    after_dedup = len(out)

    out["pipeline_version"] = PIPELINE_VERSION
    out["cleaned_at_utc"] = now_utc().isoformat()

    out = out[
        [
            "source_file",
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
    ]

    file_report = {
        "file": path.name,
        "rows_input": int(len(raw)),
        "rows_output": int(len(out)),
        "dropped_invalid_date": int(before_date_filter - after_date_filter),
        "dropped_duplicates": int(before_dedup - after_dedup),
        "title_empty_ratio": float((out["title"].fillna("").astype(str).str.strip() == "").mean()),
        "channel_empty_ratio": float((out["_channel"].fillna("").astype(str).str.strip() == "").mean()),
        "date_min": str(out["_date"].min()) if not out.empty else None,
        "date_max": str(out["_date"].max()) if not out.empty else None,
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
    total_rows_input = 0
    total_rows_output = 0

    for index, path in enumerate(raw_files, start=1):
        print(f"[{index}/{len(raw_files)}] Cleaning {path.name}...", flush=True)
        cleaned_df, file_report = process_file(path, normalize_channels=args.normalize_channels)
        out_name = f"{path.stem}_clean.parquet"
        out_path = output_dir / out_name
        cleaned_df.to_parquet(out_path, index=False)

        total_rows_input += int(file_report["rows_input"])
        total_rows_output += int(file_report["rows_output"])
        quality_files.append(file_report)
        manifest_inputs.append(
            {
                "file": path.name,
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )

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
