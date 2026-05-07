"""Corpus loading: local snapshot first, HuggingFace fallback."""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

import pandas as pd
import requests

from ina_core.store.config import Settings

logger = logging.getLogger("ina_core.store.data")

REQUIRED_CLEAN_COLUMNS = ["source_file", "title", "_title_norm", "_date", "_channel"]


class DataStoreError(RuntimeError):
    pass


class DataStore:
    """Loads the cleaned INA corpus.

    Reads parquet files either from the local snapshot pointed to by
    `data/clean/CURRENT` or from HuggingFace, then concatenates and
    normalizes them into a single DataFrame.

    Stateless by design: caching is delegated to the caller (e.g. Streamlit's
    @st.cache_resource).
    """

    def __init__(self, settings: Settings):
        self._settings = settings

    # ─── Public API ─────────────────────────────────────────────────────────

    def load(self, prefer: Literal["local", "hf", "auto"] = "auto") -> Tuple[pd.DataFrame, List[str]]:
        """Load the corpus. Returns (df, issues).

        - "local": read from data/clean/CURRENT only
        - "hf": fetch from HuggingFace only
        - "auto": local first, fall back to HF if local is empty / unavailable
        """
        if prefer == "local":
            return self._load_local()
        if prefer == "hf":
            return self._load_hf()

        df, issues = self._load_local()
        if not df.empty:
            return df, issues

        issues.append("Cache local indisponible — bascule sur HuggingFace.")
        df_hf, hf_issues = self._load_hf()
        return df_hf, issues + hf_issues

    def metadata(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n_total": 0, "channels": [], "date_min": None, "date_max": None}
        return {
            "n_total": int(len(df)),
            "channels": sorted(c for c in df["_channel"].dropna().unique().tolist()),
            "date_min": df["_date"].min().date().isoformat(),
            "date_max": df["_date"].max().date().isoformat(),
        }

    # ─── Local snapshot ─────────────────────────────────────────────────────

    def _resolve_current_snapshot(self) -> Optional[Path]:
        pointer = self._settings.current_pointer_file
        if not pointer.exists():
            return None
        version = pointer.read_text(encoding="utf-8").strip()
        if not version:
            return None
        snapshot = self._settings.clean_dir / version
        return snapshot if snapshot.is_dir() else None

    def _load_local(self) -> Tuple[pd.DataFrame, List[str]]:
        snapshot = self._resolve_current_snapshot()
        if snapshot is None:
            return pd.DataFrame(), ["Snapshot local introuvable (data/clean/CURRENT)."]

        issues: List[str] = []
        frames: List[pd.DataFrame] = []
        for file_name in self._settings.hf_parquet_files:
            file_path = snapshot / file_name
            if not file_path.exists():
                issues.append(f"Fichier absent dans le snapshot local : {file_name}")
                continue
            try:
                df = pd.read_parquet(file_path)
            except Exception as exc:
                issues.append(f"Lecture locale impossible {file_name} : {exc}")
                logger.exception("Local read failed for %s", file_name)
                continue
            normalized, issue = self._normalize(df, file_name)
            if issue:
                issues.append(issue)
            if normalized is not None and not normalized.empty:
                frames.append(normalized)

        if not frames:
            return pd.DataFrame(), issues
        return pd.concat(frames, ignore_index=True, sort=False), issues

    # ─── HuggingFace ────────────────────────────────────────────────────────

    def _load_hf(self, attempts: int = 2, timeout: int = 30) -> Tuple[pd.DataFrame, List[str]]:
        headers = (
            {"Authorization": f"Bearer {self._settings.hf_token}"}
            if self._settings.hf_token
            else {}
        )
        base_url = f"https://huggingface.co/datasets/{self._settings.hf_repo_id}/resolve/main"

        issues: List[str] = []
        if not self._settings.hf_token:
            issues.append(
                "HF_TOKEN absent. Si le dataset est privé, définissez la variable "
                "d'environnement `HF_TOKEN` ou créez `.streamlit/secrets.toml`."
            )

        frames: List[pd.DataFrame] = []
        for file_name in self._settings.hf_parquet_files:
            url = f"{base_url}/{file_name}"
            df = self._fetch_with_retry(url, headers, file_name, issues, attempts, timeout)
            if df is None:
                continue
            normalized, issue = self._normalize(df, file_name)
            if issue:
                issues.append(issue)
            if normalized is not None and not normalized.empty:
                frames.append(normalized)

        if not frames:
            return pd.DataFrame(), issues
        return pd.concat(frames, ignore_index=True, sort=False), issues

    def _fetch_with_retry(self, url, headers, name, issues, attempts, timeout):
        last_exc: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                return pd.read_parquet(io.BytesIO(response.content))
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(0.5 * attempt)
        issues.append(f"Lecture HF impossible {name} ({last_exc})")
        logger.exception("HF read failed for %s after %d attempts", name, attempts)
        return None

    # ─── Normalization (shared by local + HF) ───────────────────────────────

    @staticmethod
    def _normalize(df: pd.DataFrame, source_name: str):
        missing = [c for c in REQUIRED_CLEAN_COLUMNS if c not in df.columns]
        if missing:
            return None, f"Colonnes manquantes dans {source_name} : {', '.join(missing)}"
        keep = [c for c in REQUIRED_CLEAN_COLUMNS if c in df.columns]
        out = df[keep].copy()
        out["_date"] = pd.to_datetime(out["_date"], errors="coerce")
        out = out[out["_date"].notna()].copy()
        if out.empty:
            return None, f"Aucune date valide dans {source_name}."
        out["title"] = out["title"].fillna("").astype(str)
        out["_title_norm"] = out["_title_norm"].fillna("").astype(str)
        out["_channel"] = out["_channel"].fillna("(sans chaîne)").astype(str).str.strip()
        return out, None
