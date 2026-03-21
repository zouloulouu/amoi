#!/usr/bin/env python
"""Validate schema and core quality checks for clean parquet snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
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


def resolve_clean_dir(output_root: Path, version: str) -> Path:
    if version:
        return output_root / version
    current_path = output_root / "CURRENT"
    if not current_path.exists():
        raise SystemExit(f"Missing {current_path}. Pass --version explicitly or create CURRENT.")
    current = current_path.read_text(encoding="utf-8").strip()
    if not current:
        raise SystemExit(f"{current_path} is empty.")
    return output_root / current


def validate_file(path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    parquet_file = pq.ParquetFile(path)
    schema_names = parquet_file.schema.names
    missing = [c for c in REQUIRED_COLUMNS if c not in schema_names]
    if missing:
        errors.append(f"{path.name}: missing columns -> {', '.join(missing)}")
        return errors, {"file": path.name, "rows": int(parquet_file.metadata.num_rows), "missing": missing}

    df = pd.read_parquet(path, columns=["title", "_title_norm", "_date", "_channel", "row_hash"])
    title_empty_ratio = float((df["title"].fillna("").astype(str).str.strip() == "").mean())
    date_null_ratio = float(df["_date"].isna().mean())
    hash_dup_ratio = float(df.duplicated(subset=["row_hash"]).mean())

    if date_null_ratio > 0:
        errors.append(f"{path.name}: _date contains null values ({date_null_ratio:.4%}).")
    if hash_dup_ratio > 0:
        errors.append(f"{path.name}: row_hash duplicates detected ({hash_dup_ratio:.4%}).")
    if title_empty_ratio > 0.05:
        errors.append(f"{path.name}: title_empty_ratio too high ({title_empty_ratio:.4%} > 5%).")

    report = {
        "file": path.name,
        "rows": int(len(df)),
        "title_empty_ratio": title_empty_ratio,
        "date_null_ratio": date_null_ratio,
        "row_hash_duplicate_ratio": hash_dup_ratio,
    }
    return errors, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate clean parquet snapshot.")
    parser.add_argument("--output-root", type=Path, default=Path("data") / "clean", help="Root clean directory.")
    parser.add_argument("--version", type=str, default="", help="Snapshot version to validate.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional output JSON path (default: <snapshot>/validation_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dir = resolve_clean_dir(args.output_root, args.version.strip())
    if not clean_dir.exists():
        raise SystemExit(f"Clean snapshot does not exist: {clean_dir}")

    clean_files = sorted(clean_dir.glob("*_clean.parquet"))
    if not clean_files:
        raise SystemExit(f"No *_clean.parquet files found in: {clean_dir}")

    all_errors: list[str] = []
    files_report: list[dict[str, Any]] = []
    for path in clean_files:
        file_errors, report = validate_file(path)
        files_report.append(report)
        all_errors.extend(file_errors)

    out = {
        "snapshot": clean_dir.name,
        "clean_dir": str(clean_dir.as_posix()),
        "files": files_report,
        "errors": all_errors,
        "passed": len(all_errors) == 0,
    }
    report_path = args.report_path or (clean_dir / "validation_report.json")
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Validation report: {report_path}")
    if all_errors:
        for line in all_errors:
            print(f"- ERROR: {line}")
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
