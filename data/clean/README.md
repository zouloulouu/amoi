# Clean Data Folder

This folder stores versioned clean snapshots generated from `data/raw`.

Conventions:
- One folder per run: `vYYYYMMDD_HHMMSS_utc/`
- Each snapshot contains:
  - `*_clean.parquet`
  - `manifest.json`
  - `quality_report.json`
  - `validation_report.json` (after validation)
- `CURRENT` (generated file, ignored in git) points to the active version.
