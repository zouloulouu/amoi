import json
import logging
import os
import re
import unicodedata
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="INA - Dictionnaire (simple)", layout="wide")

CLEAN_ROOT = Path("data") / "clean"
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

REQUIRED_CLEAN_COLUMNS = ["source_file", "title", "_title_norm", "_date", "_channel"]
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

DIRECTION_UP = 1
DIRECTION_DOWN = -1
DIRECTION_FLAT = 0


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


def ensure_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    missing = [c for c in required_columns if c not in df.columns]
    for col in missing:
        df[col] = pd.NA
    return missing


def normalize_text(x):
    if x is None or pd.isna(x):
        return pd.NA
    text = str(x).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA


def strip_accents(text):
    if text is None or pd.isna(text):
        return pd.NA
    value = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn"
    )


def clean_title(x):
    normalized = normalize_text(x)
    if pd.isna(normalized):
        return pd.NA
    no_accents = strip_accents(normalized)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(no_accents))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned if cleaned else pd.NA


def parse_duration_to_seconds(x) -> float:
    if x is None or pd.isna(x):
        return np.nan
    raw = str(x).strip()
    if not raw:
        return np.nan
    parts = raw.split(":")
    if len(parts) not in (3, 4):
        return np.nan
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
        centisec = int(parts[3]) if len(parts) == 4 else 0
    except ValueError:
        return np.nan
    return float(h * 3600 + m * 60 + s + (centisec / 100.0))


def harmonize_channel(x):
    normalized = clean_title(x)
    if pd.isna(normalized):
        return pd.NA
    compact = str(normalized).replace(" ", "")
    mapping = {
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
    if compact in mapping:
        return mapping[compact]
    if "franceinter" in compact:
        return "France Inter"
    return str(x).strip() if str(x).strip() else pd.NA


def classify_content(row: pd.Series):
    def _safe(v):
        cleaned = clean_title(v)
        return "" if pd.isna(cleaned) else str(cleaned)

    text = " ".join(
        [
            _safe(row.get("titre_propre")),
            _safe(row.get("titre_programme")),
            _safe(row.get("titre_collection")),
        ]
    )
    if "plateau" in text:
        return "plateau"
    if re.search(r"\b(?:programme du|emission du|edition du)\b", text):
        return "edition_complete"
    return "sujet"


def standardize_emission(row: pd.Series):
    def _safe(v):
        cleaned = clean_title(v)
        return "" if pd.isna(cleaned) else str(cleaned)

    text = " ".join(
        [
            _safe(row.get("titre_propre")),
            _safe(row.get("titre_programme")),
            _safe(row.get("titre_collection")),
        ]
    )
    if re.search(r"\b(?:20heures|20 heures|20h)\b", text):
        return "20 heures"
    if re.search(r"\bpremiere edition\b", text):
        return "Premiere edition"
    return pd.NA


def clean_file(df: pd.DataFrame, source_name: Optional[str] = None) -> pd.DataFrame:
    work = df.copy()
    input_columns = list(work.columns)

    alias_candidates = {
        "id": ["id", "indice"],
        "chaine": ["chaine", "_channel", "channel", "raw_channel"],
        "date": ["date", "_date"],
        "date_diffusion": ["date_diffusion", "raw_date"],
        "heure_diffusion": ["heure_diffusion", "raw_time", "time"],
        "duree": ["duree"],
        "titre_propre": ["titre_propre", "title"],
        "titre_collection": ["titre_collection"],
        "titre_programme": ["titre_programme"],
        "genre": ["genre"],
        "url_notice": ["url_notice"],
        "inflation_extended": ["inflation_extended"],
        "source_file": ["source_file"],
        "month": ["month"],
    }
    for target, candidates in alias_candidates.items():
        if target in work.columns:
            continue
        for candidate in candidates:
            if candidate in work.columns:
                work[target] = work[candidate]
                break

    if "source_file" not in work.columns or work["source_file"].isna().all():
        work["source_file"] = source_name if source_name else pd.NA

    created_columns = ensure_columns(work, TARGET_SCHEMA_COLUMNS)

    work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
    work["date_diffusion"] = pd.to_datetime(work["date_diffusion"], errors="coerce", dayfirst=True)
    work["date"] = work["date"].where(work["date"].notna(), work["date_diffusion"])
    work["date_diffusion"] = work["date_diffusion"].where(work["date_diffusion"].notna(), work["date"])
    work["inflation_extended"] = pd.to_numeric(work["inflation_extended"], errors="coerce").astype("Int64")

    parsed_duree_sec = work["duree"].map(parse_duration_to_seconds)
    existing_duree_sec = pd.to_numeric(work["duree_sec"], errors="coerce")
    work["duree_sec"] = existing_duree_sec.where(existing_duree_sec.notna(), parsed_duree_sec)

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
        work[col] = work[col].astype("string").str.strip()
        work[col] = work[col].replace("", pd.NA)

    work["chaine"] = work["chaine"].map(harmonize_channel).astype("string")
    work["clean_titre"] = work["titre_propre"].map(clean_title).astype("string")
    work["clean_programme"] = work["titre_programme"].map(clean_title).astype("string")

    base_date = pd.to_datetime(work["date"], errors="coerce")
    work["annee"] = base_date.dt.year.astype("Int64")
    work["mois"] = base_date.dt.month.astype("Int64")
    work["ym"] = base_date.dt.strftime("%Y-%m").astype("string")
    work["ym"] = work["ym"].where(base_date.notna(), pd.NA)
    work["month"] = work["month"].where(work["month"].notna(), work["ym"]).astype("string")

    text_combo = (
        work["clean_titre"].fillna("")
        + " "
        + work["clean_programme"].fillna("")
        + " "
        + work["titre_collection"].map(clean_title).fillna("")
    ).str.strip()
    is_plateau = text_combo.str.contains(r"\bplateau\b", regex=True, na=False)
    is_edition = text_combo.str.contains(r"\b(?:programme du|emission du|edition du)\b", regex=True, na=False)

    work["type_contenu"] = pd.Series("sujet", index=work.index, dtype="string")
    work.loc[is_edition, "type_contenu"] = "edition_complete"
    work.loc[is_plateau, "type_contenu"] = "plateau"

    work["emission_std"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work.loc[text_combo.str.contains(r"\b(?:20heures|20 heures|20h)\b", regex=True, na=False), "emission_std"] = (
        "20 heures"
    )
    work.loc[text_combo.str.contains(r"\bpremiere edition\b", regex=True, na=False), "emission_std"] = (
        "Premiere edition"
    )

    has_url = work["url_notice"].notna() & work["url_notice"].astype("string").str.strip().ne("")
    has_prog = work["titre_programme"].notna() & work["titre_programme"].astype("string").str.strip().ne("")
    has_coll = work["titre_collection"].notna() & work["titre_collection"].astype("string").str.strip().ne("")
    work["has_url_notice"] = has_url.astype("Int8")
    work["has_titre_programme"] = has_prog.astype("Int8")
    work["has_titre_collection"] = has_coll.astype("Int8")

    dedup_subset = [c for c in ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"] if c in work]
    before_dedup = len(work)
    if dedup_subset:
        work = work.drop_duplicates(subset=dedup_subset, keep="first").copy()
    duplicates_removed = before_dedup - len(work)

    work["title"] = work["titre_propre"].fillna(work["titre_programme"]).fillna(work["titre_collection"]).astype(
        "string"
    )
    work["_title_norm"] = work["title"].map(clean_title).astype("string")
    work["_date"] = pd.to_datetime(work["date"], errors="coerce")
    work["_date"] = work["_date"].where(work["_date"].notna(), pd.to_datetime(work["date_diffusion"], errors="coerce"))
    work["_channel"] = work["chaine"].astype("string")

    ensure_columns(work, REQUIRED_CLEAN_COLUMNS)

    work.attrs["cleaning_diagnostics"] = {
        "source_file": source_name or "(unknown)",
        "input_columns": input_columns,
        "output_columns": list(work.columns),
        "created_columns": created_columns,
        "rows_before": int(len(df)),
        "rows_after": int(len(work)),
        "duplicates_removed": int(duplicates_removed),
    }
    return work


def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["chaine"] = work["chaine"].fillna("(sans chaine)")
    dedup_subset = [c for c in ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"] if c in work]
    key_missing = (
        work.groupby("chaine", as_index=False)
        .agg(
            nb_lignes=("chaine", "size"),
            taux_manquant_date=("date", lambda s: float(pd.to_datetime(s, errors="coerce").isna().mean())),
            taux_manquant_titre=("titre_propre", lambda s: float(s.isna().mean())),
            taux_manquant_url=("url_notice", lambda s: float(s.isna().mean())),
            taux_manquant_heure=("heure_diffusion", lambda s: float(s.isna().mean())),
            part_inflation=("inflation_extended", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).eq(1).mean())),
        )
        .sort_values("nb_lignes", ascending=False)
    )
    if dedup_subset:
        dup_by_chaine = work.assign(_is_dup=work.duplicated(subset=dedup_subset)).groupby("chaine")["_is_dup"].sum()
        key_missing["doublons_restants"] = key_missing["chaine"].map(dup_by_chaine).fillna(0).astype(int)
    else:
        key_missing["doublons_restants"] = 0
    key_missing["doublons_supprimes_total"] = int(df.attrs.get("duplicates_removed_total", 0))

    type_share = pd.crosstab(work["chaine"], work["type_contenu"], normalize="index")
    for col in ["plateau", "edition_complete", "sujet"]:
        if col in type_share.columns:
            key_missing[f"part_{col}"] = key_missing["chaine"].map(type_share[col]).fillna(0.0)
        else:
            key_missing[f"part_{col}"] = 0.0
    return key_missing


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
    work["chaine"] = work["chaine"].fillna("(sans chaine)")
    work["ym"] = work["ym"].fillna("inconnu")
    work["inflation_flag"] = pd.to_numeric(work["inflation_extended"], errors="coerce").fillna(0).eq(1).astype(int)
    work["duree_sec_num"] = pd.to_numeric(work["duree_sec"], errors="coerce").fillna(0.0)
    work["duree_inflation"] = work["duree_sec_num"] * work["inflation_flag"]

    grouped = (
        work.groupby(["chaine", "ym"], as_index=False)
        .agg(
            nb_total=("inflation_flag", "size"),
            nb_inflation=("inflation_flag", "sum"),
            duree_totale=("duree_sec_num", "sum"),
            duree_inflation=("duree_inflation", "sum"),
        )
        .sort_values(["chaine", "ym"])
    )
    grouped["part_inflation_nb"] = np.where(grouped["nb_total"] > 0, grouped["nb_inflation"] / grouped["nb_total"], 0.0)
    grouped["part_inflation_duree"] = np.where(
        grouped["duree_totale"] > 0, grouped["duree_inflation"] / grouped["duree_totale"], 0.0
    )
    return grouped


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


def resolve_active_clean_dir(clean_root: Path) -> Path:
    current_path = clean_root / "CURRENT"
    if current_path.exists():
        version = current_path.read_text(encoding="utf-8").strip()
        if version:
            target = clean_root / version
            if target.is_dir():
                return target
    candidates = sorted([p for p in clean_root.glob("v*_utc") if p.is_dir()])
    if candidates:
        return candidates[-1]
    return clean_root


@st.cache_resource(show_spinner=False)
def load_clean_parquets_from_folder(
    folder: str, signature: Tuple[Tuple[str, int, int], ...]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    issues: List[str] = []
    if not os.path.isdir(folder):
        LOGGER.warning("Dossier clean introuvable: %s", folder)
        issues.append(f"Dossier clean introuvable: {folder}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), issues
    if not signature:
        LOGGER.warning("Aucun fichier parquet trouve dans %s", folder)
        issues.append(f"Aucun fichier `.parquet` trouve dans {folder}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), issues

    frames: List[pd.DataFrame] = []
    diagnostics_rows: List[Dict[str, object]] = []
    for file_name, _, _ in signature:
        path = os.path.join(folder, file_name)
        try:
            df = pd.read_parquet(path)
            cleaned = clean_file(df, source_name=file_name)
            cleaned["_date"] = pd.to_datetime(cleaned["_date"], errors="coerce")
            cleaned = cleaned[cleaned["_date"].notna()].copy()
            if cleaned.empty:
                LOGGER.warning("Aucune date valide apres cleaning dans %s", file_name)
                issues.append(f"Aucune date valide apres cleaning dans {file_name}.")
                continue

            cleaned["title"] = cleaned["title"].fillna("").astype(str)
            cleaned["_title_norm"] = cleaned["_title_norm"].fillna("").astype(str)
            cleaned["_channel"] = cleaned["_channel"].fillna("(sans chaine)").astype(str).str.strip()

            diagnostics = cleaned.attrs.get("cleaning_diagnostics", {})
            diagnostics_rows.append(
                {
                    "source_file": diagnostics.get("source_file", file_name),
                    "rows_before": diagnostics.get("rows_before", len(df)),
                    "rows_after": diagnostics.get("rows_after", len(cleaned)),
                    "duplicates_removed": diagnostics.get("duplicates_removed", 0),
                    "created_columns": ", ".join(diagnostics.get("created_columns", [])),
                    "input_columns": ", ".join(diagnostics.get("input_columns", [])),
                    "output_columns": ", ".join(diagnostics.get("output_columns", [])),
                }
            )

            frames.append(cleaned)
            LOGGER.info("Parquet charge+harmonise: %s | lignes=%s", file_name, len(cleaned))
        except Exception as exc:
            LOGGER.exception("Lecture impossible sur %s: %s", file_name, exc)
            issues.append(f"Lecture impossible: {file_name} ({exc})")

    if not frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), issues

    combined = pd.concat(frames, ignore_index=True, sort=False)
    global_dedup_subset = [c for c in ["chaine", "date", "heure_diffusion", "titre_propre", "titre_programme"] if c in combined]
    before_global_dedup = len(combined)
    if global_dedup_subset:
        combined = combined.drop_duplicates(subset=global_dedup_subset, keep="first").copy()
    global_removed = int(before_global_dedup - len(combined))

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    if diagnostics_df.empty:
        diagnostics_df = pd.DataFrame(
            columns=[
                "source_file",
                "rows_before",
                "rows_after",
                "duplicates_removed",
                "created_columns",
                "input_columns",
                "output_columns",
            ]
        )
    diagnostics_df["duplicates_removed"] = pd.to_numeric(diagnostics_df["duplicates_removed"], errors="coerce").fillna(0).astype(int)
    diagnostics_df.attrs["global_duplicates_removed"] = global_removed
    combined.attrs["duplicates_removed_total"] = int(diagnostics_df["duplicates_removed"].sum()) + global_removed

    quality_report_df = build_quality_report(combined)
    monthly_df = build_monthly_analysis(combined)
    return combined, monthly_df, quality_report_df, diagnostics_df, issues


def prepare_keywords(keywords: List[str]) -> List[str]:
    normalized = []
    for keyword in keywords:
        cleaned = clean_title(keyword)
        if pd.isna(cleaned):
            continue
        normalized.append(str(cleaned))
    dedup = sorted(set(normalized))
    return dedup


def count_occurrences(text_norm: str, keywords_norm: List[str]) -> int:
    if text_norm is None or pd.isna(text_norm):
        return 0
    text = str(text_norm)
    if not text:
        return 0
    total = 0
    for keyword in keywords_norm:
        if keyword is None or pd.isna(keyword):
            continue
        kw = str(keyword)
        if len(kw) <= 4 and kw.isalpha():
            total += len(re.findall(rf"\b{re.escape(kw)}\b", text))
        else:
            total += text.count(kw)
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
        titles_norm = df[title_col].fillna("").astype(str).map(clean_title).fillna("").astype(str)

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

with st.expander("Donnees", expanded=False):
    st.write("Chargement des fichiers `*_clean.parquet` depuis le snapshot actif `data/clean/CURRENT`.")

active_clean_dir = resolve_active_clean_dir(CLEAN_ROOT)
data_signature = parquet_signature(active_clean_dir.as_posix())
df_base, df_monthly, df_quality, df_clean_diag, load_issues = load_clean_parquets_from_folder(
    active_clean_dir.as_posix(), data_signature
)
for issue in load_issues:
    st.warning(issue)

if df_base.empty:
    st.warning(f"Aucune donnee clean chargee dans `{active_clean_dir.as_posix()}`.")
    st.stop()

if "dictionaries" not in st.session_state:
    st.session_state["dictionaries"] = load_dictionaries(DICTIONARY_PATH)

dictionaries = normalize_dictionaries_payload(st.session_state["dictionaries"])
if not dictionaries:
    dictionaries = clone_dictionaries(DEFAULT_DICTIONARIES)
st.session_state["dictionaries"] = dictionaries

columns = list(df_base.columns)
title_col = "title" if "title" in columns else None
has_source_channel = "_channel" in columns

if not title_col or "_date" not in columns:
    st.error("Colonnes minimales introuvables: il faut au moins une colonne titre et une colonne date.")
    st.stop()

st.sidebar.header("Parametres")
frequency = st.sidebar.selectbox("Frequence", ["Mensuelle", "Trimestrielle", "Annuelle"], index=0)
with st.sidebar.expander("Diagnostics", expanded=False):
    show_cleaned = st.checkbox("Afficher les donnees nettoyees", value=False)
    show_quality = st.checkbox("Afficher le rapport qualite", value=False)
    show_monthly = st.checkbox("Afficher la base mensuelle", value=False)
    st.caption(f"Log erreurs: `{LOG_PATH.as_posix()}`")
    if st.checkbox("Afficher les 30 dernieres lignes du log", value=False):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                last_lines = f.readlines()[-30:]
            st.code("".join(last_lines) if last_lines else "(log vide)", language="text")
        except Exception as exc:
            LOGGER.exception("Lecture du log impossible: %s", exc)
            st.warning(f"Lecture du log impossible ({exc})")

with st.expander("Diagnostic nettoyage", expanded=False):
    input_union = sorted(
        {
            c.strip()
            for raw in df_clean_diag.get("input_columns", pd.Series(dtype="string")).fillna("")
            for c in str(raw).split(",")
            if c.strip()
        }
    )
    created_union = sorted(
        {
            c.strip()
            for raw in df_clean_diag.get("created_columns", pd.Series(dtype="string")).fillna("")
            for c in str(raw).split(",")
            if c.strip()
        }
    )
    st.caption(f"Colonnes presentes avant nettoyage (union): {len(input_union)}")
    st.caption(f"Colonnes finales apres harmonisation: {len(df_base.columns)}")
    st.caption(f"Colonnes absentes creees automatiquement: {len(created_union)}")
    if created_union:
        st.code(", ".join(created_union), language="text")
    total_removed = int(pd.to_numeric(df_clean_diag.get("duplicates_removed", pd.Series(dtype="int64")), errors="coerce").fillna(0).sum())
    total_removed += int(df_clean_diag.attrs.get("global_duplicates_removed", 0))
    st.caption(f"Doublons supprimes (fichiers + global): {total_removed}")
    if not df_clean_diag.empty:
        st.dataframe(
            df_clean_diag[
                [
                    "source_file",
                    "rows_before",
                    "rows_after",
                    "duplicates_removed",
                    "created_columns",
                ]
            ],
            width="stretch",
        )

if show_cleaned:
    st.subheader("Apercu base nettoyee")
    st.dataframe(df_base.head(300), width="stretch")

if show_monthly:
    st.subheader("Apercu base mensuelle")
    st.dataframe(df_monthly.head(300), width="stretch")

if show_quality:
    st.subheader("Rapport qualite")
    st.dataframe(df_quality, width="stretch")

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

st.subheader("Sens du signal")
if not (up_norm or down_norm):
    st.info("Aucun mot-cle UP/DOWN dans le dictionnaire de ce theme. Ajoute des termes pour activer le signal.")
else:
    up_total = int(stats["up_titles"].sum())
    down_total = int(stats["down_titles"].sum())
    non_zero_periods = int((stats["net_signal"] != 0).sum())
    st.caption(
        f"UP={up_total} | DOWN={down_total} | periodes avec signal net non nul={non_zero_periods}/{len(stats)}"
    )
    if up_total == 0 and down_total == 0:
        st.warning(
            "Aucun titre n'a active les termes UP/DOWN sur la periode/filtres courants. "
            "Essaie d'elargir la periode ou de retirer des filtres chaines."
        )

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
