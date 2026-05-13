"""Settings for the I/O layer (HF endpoint, local paths, secrets)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_HF_PARQUET_FILES: Tuple[str, ...] = (
    "ina_inflation_stats_bfmtv_general_clean.parquet",
    "ina_inflation_stats_europe1_information_clean.parquet",
    "ina_inflation_stats_fr2_clean.parquet",
    "ina_inflation_stats_france3_jtelevise_clean.parquet",
    "ina_inflation_stats_france_inter_information_clean.parquet",
    "ina_inflation_stats_rtl_journal_parle_clean.parquet",
    "ina_merged_dedup_finalTF1_clean.parquet",
)


@dataclass(frozen=True)
class Settings:
    """All paths and credentials needed by the I/O layer.

    Designed to be built once at application start and passed by injection
    into DataStore / DictRepository instances.
    """

    hf_repo_id: str
    hf_parquet_files: Tuple[str, ...]
    hf_token: Optional[str]
    project_root: Path
    dictionary_path_override: Optional[Path] = None

    @property
    def clean_dir(self) -> Path:
        return self.project_root / "data" / "clean"

    @property
    def current_pointer_file(self) -> Path:
        return self.clean_dir / "CURRENT"

    @property
    def dictionary_path(self) -> Path:
        if self.dictionary_path_override is not None:
            return self.dictionary_path_override
        return self.project_root / "dictionaries.json"

    @property
    def cache_dir(self) -> Path:
        """Disk-cache root for df_base and tagged dataframes (gitignored)."""
        return self.project_root / "data" / "cache"
