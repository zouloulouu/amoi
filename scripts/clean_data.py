#!/usr/bin/env python
"""Clean raw INA parquet files and write one clean parquet per channel.

Output schema (matches app.py REQUIRED_CLEAN_COLUMNS + extras):
    id            : str  — SHA-256 hash (stable row identifier)
    source_file   : str  — origin filename
    _channel      : str  — normalised channel name
    _date         : datetime64[ns] — broadcast date (midnight)
    title         : str  — original title (titre_propre, stripped)
    _title_norm   : str  — title lowercased + accents removed (exact app.py logic)
    genre         : str  — nullable, kept as-is
    url_notice    : str  — nullable, kept as-is

Usage:
    python scripts/clean_data.py
    python scripts/clean_data.py --input-dir data/raw --output-dir data/clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Columns kept in the output (order matters for readability)
OUTPUT_COLUMNS = [
    "id",
    "source_file",
    "_channel",
    "_date",
    "title",
    "_title_norm",
    "genre",
    "url_notice",
]

# Canonical channel names — keys are lowercased compact forms
CHANNEL_MAPPING: dict[str, str] = {
    "bfmtv": "BFMTV",
    "bfm tv": "BFMTV",
    "bfm": "BFMTV",
    "france 2": "France 2",
    "france2": "France 2",
    "fr2": "France 2",
    "france 3": "France 3",
    "france3": "France 3",
    "fr3": "France 3",
    "france inter": "France Inter",
    "franceinter": "France Inter",
    "europe 1": "Europe 1",
    "europe1": "Europe 1",
    "rtl": "RTL",
    "tf1": "TF1",
}

# Guard: reject files that lose more than this fraction of rows during cleaning
MAX_DROP_RATIO = 0.30


# ──────────────────────────────────────────────────────────────────────────────
# PURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

from ina_core import normalize_text


def normalize_channel(raw: Any) -> str:
    """Map raw channel string to canonical name, or return stripped original."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ""
    key = str(raw).strip().lower()
    return CHANNEL_MAPPING.get(key, str(raw).strip())


def make_id(chaine_raw: Any, date_raw: Any, heure_raw: Any, titre_raw: Any) -> str:
    """Deterministic SHA-256 row identifier built from pre-transformation values."""
    parts = "|".join([
        str(chaine_raw or ""),
        str(date_raw or ""),
        str(heure_raw or ""),
        str(titre_raw or "").strip(),
    ])
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def strip_hhmmsff(value: Any) -> str:
    """Normalise heure_diffusion: truncate hh:mm:ss:ff → hh:mm:ss."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    s = str(value).strip()
    parts = s.split(":")
    if len(parts) == 4:
        return ":".join(parts[:3])
    return s


# ──────────────────────────────────────────────────────────────────────────────
# FILE-LEVEL CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def clean_file(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply the full cleaning pipeline to a single raw DataFrame.
    Returns (cleaned_df, report_dict).
    """
    report: dict[str, Any] = {"file": source_name, "rows_input": len(df)}

    # ── Step 1: source_file ──────────────────────────────────────────────────
    df = df.copy()
    if "source_file" not in df.columns:
        df["source_file"] = source_name

    # ── Step 2: id (hash before any transformation) ──────────────────────────
    df["id"] = [
        make_id(
            df["chaine"].iloc[i] if "chaine" in df.columns else None,
            df["date_diffusion"].iloc[i] if "date_diffusion" in df.columns else None,
            df["heure_diffusion"].iloc[i] if "heure_diffusion" in df.columns else None,
            df["titre_propre"].iloc[i] if "titre_propre" in df.columns else None,
        )
        for i in range(len(df))
    ]

    # ── Step 3: _channel ─────────────────────────────────────────────────────
    raw_channel = df["chaine"] if "chaine" in df.columns else pd.Series([""] * len(df))
    df["_channel"] = raw_channel.map(normalize_channel).astype("string")

    # ── Step 4: _date ────────────────────────────────────────────────────────
    # Prefer the already-parsed timestamp column; fall back to date_diffusion string.
    if "date" in df.columns:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "date_diffusion" in df.columns:
        df["_date"] = pd.to_datetime(df["date_diffusion"], errors="coerce", dayfirst=True)
    else:
        df["_date"] = pd.NaT

    # Normalise to midnight (drop time component if any)
    df["_date"] = df["_date"].dt.normalize()

    n_invalid_date = int(df["_date"].isna().sum())
    report["rows_invalid_date"] = n_invalid_date

    # ── Step 5: title ─────────────────────────────────────────────────────────
    if "titre_propre" in df.columns:
        df["title"] = df["titre_propre"].astype("string").str.strip()
    else:
        df["title"] = pd.NA

    # ── Step 6: _title_norm (exact app.py logic) ──────────────────────────────
    df["_title_norm"] = df["title"].map(normalize_text).astype("string")

    # ── Step 7: genre, url_notice (keep as-is, nullable) ─────────────────────
    for col in ("genre", "url_notice"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype("string")

    # ── Step 8: drop rows where title is missing or empty ────────────────────
    before_title_drop = len(df)
    df = df[df["title"].notna() & (df["title"].str.strip() != "")].copy()
    report["rows_dropped_empty_title"] = before_title_drop - len(df)

    # ── Step 9: select output columns ─────────────────────────────────────────
    df = df[OUTPUT_COLUMNS].copy()

    # ── Step 10: dedup level 1 — exact rows ───────────────────────────────────
    before_dedup1 = len(df)
    df = df.drop_duplicates(keep="first")
    report["rows_dropped_exact_dedup"] = before_dedup1 - len(df)

    # ── Step 11: dedup level 2 — normalised title per channel per day ─────────
    # Retain the row that has url_notice, otherwise keep first occurrence.
    before_dedup2 = len(df)
    df = df.copy()
    df["_date_day"] = df["_date"].dt.date
    df["_has_url"] = df["url_notice"].notna().astype(int)
    df = (
        df.sort_values(["_channel", "_date_day", "_title_norm", "_has_url"],
                       ascending=[True, True, True, False])
          .drop_duplicates(subset=["_channel", "_date_day", "_title_norm"], keep="first")
          .drop(columns=["_date_day", "_has_url"])
    )
    report["rows_dropped_norm_dedup"] = before_dedup2 - len(df)

    report["rows_output"] = len(df)
    report["retention_ratio"] = round(len(df) / report["rows_input"], 4) if report["rows_input"] else 0.0
    return df.reset_index(drop=True), report


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw INA parquet files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "clean")
    parser.add_argument("--report-path", type=Path, default=Path("reports") / "clean_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    raw_files = sorted(args.input_dir.glob("*.parquet"))
    if not raw_files:
        raise SystemExit(f"No parquet files found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    all_reports: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0

    for idx, path in enumerate(raw_files, start=1):
        print(f"[{idx}/{len(raw_files)}] {path.name} ...", flush=True)

        df_raw = pd.read_parquet(path)
        df_clean, report = clean_file(df_raw, source_name=path.name)

        drop_ratio = 1.0 - report["retention_ratio"]
        if drop_ratio > MAX_DROP_RATIO:
            print(
                f"  WARNING: {path.name} dropped {drop_ratio:.1%} of rows "
                f"(threshold={MAX_DROP_RATIO:.0%}). Check cleaning rules."
            )

        out_path = args.output_dir / f"{path.stem}_clean.parquet"
        df_clean.to_parquet(out_path, index=False, compression="zstd")

        total_in += report["rows_input"]
        total_out += report["rows_output"]
        all_reports.append(report)

        print(
            f"  {report['rows_input']:,} -> {report['rows_output']:,} rows "
            f"(-{report['rows_dropped_empty_title']} empty title, "
            f"-{report['rows_dropped_exact_dedup']} exact dup, "
            f"-{report['rows_dropped_norm_dedup']} norm dup)"
        )

    summary = {
        "files": len(all_reports),
        "rows_input": total_in,
        "rows_output": total_out,
        "retention_ratio": round(total_out / total_in, 4) if total_in else 0.0,
        "output_columns": OUTPUT_COLUMNS,
    }
    full_report = {"summary": summary, "files": all_reports}
    args.report_path.write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nDone. {total_in:,} -> {total_out:,} rows ({summary['retention_ratio']:.1%} retained)")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
