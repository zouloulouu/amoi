# Data Architecture (raw -> clean -> app)

## Goals
- Keep `data/raw` immutable (source of truth).
- Build reproducible snapshots in `data/clean/<version>`.
- Make `app.py` consume only clean parquet files.

## Target Structure
```text
data/
  raw/                     # immutable source parquet files
  clean/
    CURRENT               # active clean snapshot version (generated, ignored in git)
    vYYYYMMDD_HHMMSS_utc/
      *_clean.parquet
      manifest.json
      quality_report.json
      validation_report.json
scripts/
  clean_data.py
  validate_clean.py
app.py
```

## Pipeline Commands
1. Build a clean snapshot:
```powershell
python scripts/clean_data.py --input-dir data/raw --output-root data/clean --normalize-channels --set-current
```

2. Validate the active snapshot:
```powershell
python scripts/validate_clean.py --output-root data/clean
```

3. Validate a specific snapshot:
```powershell
python scripts/validate_clean.py --output-root data/clean --version v20260321_200000_utc
```

## Migration from current repository layout
Current project stores parquet files directly under `data/`.

Recommended one-time migration:
1. Create `data/raw`.
2. Copy existing `data/*.parquet` into `data/raw/`.
3. Keep original files untouched until Streamlit is migrated and verified.
4. Generate first clean snapshot with `scripts/clean_data.py`.
5. Run `scripts/validate_clean.py`.
6. Refactor `app.py` to read only `data/clean/<CURRENT>/*_clean.parquet`.

## Rules
- Never edit files in `data/raw` manually.
- Never write clean outputs into `data/raw`.
- Each run writes a new versioned folder in `data/clean`.
- Use `manifest.json` + reports to track lineage and quality.
