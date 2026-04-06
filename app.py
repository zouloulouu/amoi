import json
import logging
import os
import re
import time
import unicodedata
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="INA — Analyse thématique", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

CLEAN_ROOT = Path("data") / "clean"
DICTIONARY_PATH = "dictionaries.json"
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "streamlit_app.log"

HF_REPO_ID = "zouloulouu/data_ina_clean"
HF_PARQUET_FILES = (
    "ina_inflation_stats_bfmtv_general_clean.parquet",
    "ina_inflation_stats_europe1_information_clean.parquet",
    "ina_inflation_stats_fr2_clean.parquet",
    "ina_inflation_stats_france3_jtelevise_clean.parquet",
    "ina_inflation_stats_france_inter_information_clean.parquet",
    "ina_inflation_stats_rtl_journal_parle_clean.parquet",
    "ina_merged_dedup_finalTF1_clean.parquet",
)

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

# Sens du signal : 4 états possibles
DIRECTION_UP = 1        # au moins un terme UP, aucun terme DOWN
DIRECTION_DOWN = -1     # au moins un terme DOWN, aucun terme UP
DIRECTION_FLAT = 0      # aucun terme UP ni DOWN (ou titre non matché)
DIRECTION_AMBIGUOUS = 2 # termes UP et DOWN présents simultanément dans le même titre


# ──────────────────────────────────────────────────────────────────────────────
# LOGGER
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS — DICTIONNAIRES
# ──────────────────────────────────────────────────────────────────────────────

def empty_theme_dictionary() -> Dict[str, List[str]]:
    return {"concept": [], "context": [], "up": [], "down": []}


def clean_term_list(values) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def normalize_theme_dictionary(raw_theme) -> Dict[str, List[str]]:
    if isinstance(raw_theme, list):
        return {"concept": clean_term_list(raw_theme), "context": [], "up": [], "down": []}
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


def clone_dictionaries(d: Dict) -> Dict:
    return json.loads(json.dumps(d, ensure_ascii=False))


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text


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


def load_dictionaries_from_hf(token: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    import requests as _req
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/dictionaries.json"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = _req.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return normalize_dictionaries_payload(r.json())
    except Exception as exc:
        LOGGER.warning("Chargement dictionnaires HF impossible: %s", exc)
    return {}


def save_dictionaries_to_hf(dictionaries: Dict[str, Dict[str, List[str]]], token: Optional[str]) -> None:
    if not token:
        return
    import io as _io
    from huggingface_hub import HfApi
    normalized = normalize_dictionaries_payload(dictionaries)
    content = json.dumps(normalized, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        HfApi().upload_file(
            path_or_fileobj=_io.BytesIO(content),
            path_in_repo="dictionaries.json",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message="Update dictionaries",
        )
    except Exception as exc:
        LOGGER.warning("Sauvegarde dictionnaires HF impossible: %s", exc)
        raise


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS — CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Chargement des données (première ouverture, ~2 min)...")
def load_clean_parquets_from_hf() -> Tuple[pd.DataFrame, List[str]]:
    import io
    import requests

    hf_token = st.secrets.get("HF_TOKEN", None)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    base_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"
    issues: List[str] = []
    frames = []

    for file_name in HF_PARQUET_FILES:
        try:
            url = f"{base_url}/{file_name}"
            response = requests.get(url, headers=headers, timeout=300)
            response.raise_for_status()
            df = pd.read_parquet(io.BytesIO(response.content))
            missing = [c for c in REQUIRED_CLEAN_COLUMNS if c not in df.columns]
            if missing:
                LOGGER.warning("Colonnes clean manquantes dans %s: %s", file_name, ", ".join(missing))
                issues.append(f"Colonnes manquantes dans {file_name} : {', '.join(missing)}")
                continue
            keep_cols = [c for c in REQUIRED_CLEAN_COLUMNS if c in df.columns]
            normalized = df[keep_cols].copy()
            normalized["_date"] = pd.to_datetime(normalized["_date"], errors="coerce")
            normalized = normalized[normalized["_date"].notna()].copy()
            if normalized.empty:
                LOGGER.warning("Aucune date valide dans %s", file_name)
                issues.append(f"Aucune date valide dans {file_name}.")
                continue
            normalized["title"] = normalized["title"].fillna("").astype(str)
            normalized["_title_norm"] = normalized["_title_norm"].fillna("").astype(str)
            normalized["_channel"] = (
                normalized["_channel"].fillna("(sans chaîne)").astype(str).str.strip()
            )
            frames.append(normalized)
            LOGGER.info("Chargé HF : %s | %d lignes", file_name, len(normalized))
        except Exception as exc:
            LOGGER.exception("Lecture HF impossible sur %s: %s", file_name, exc)
            issues.append(f"Lecture impossible depuis HuggingFace : {file_name} ({exc})")

    if not frames:
        return pd.DataFrame(), issues
    return pd.concat(frames, ignore_index=True, sort=False), issues


# ──────────────────────────────────────────────────────────────────────────────
# LOGIQUE MÉTIER — MARQUAGE (TAGGING)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_keywords(keywords: List[str]) -> List[str]:
    normalized = [normalize_text(k) for k in keywords if str(k).strip()]
    return sorted(set(k for k in normalized if k))


def count_occurrences(text_norm: str, keywords_norm: List[str]) -> int:
    """Compte les occurrences brutes (peut être > 1 par titre).
    Utiliser is_match (binaire) pour construire un signal de présence/absence.
    """
    if not text_norm:
        return 0
    total = 0
    for keyword in keywords_norm:
        if len(keyword) <= 4 and keyword.isalpha():
            total += len(re.findall(rf"\b{re.escape(keyword)}\b", text_norm))
        else:
            total += text_norm.count(keyword)
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
    """Marque chaque observation avec les colonnes de présence et de sens.

    Logique de matching (binaire) :
      - is_match = 1 si le titre contient au moins un mot-clé concept, 0 sinon.

    Logique de direction (4 états) :
      - FLAT (0)      : titre non matché, ou matché sans terme UP/DOWN
      - UP (1)        : matché + au moins un terme UP, aucun terme DOWN
      - DOWN (-1)     : matché + au moins un terme DOWN, aucun terme UP
      - AMBIGUOUS (2) : matché + termes UP ET DOWN présents simultanément

    Le cas AMBIGU est signalé explicitement plutôt que de choisir arbitrairement
    le signal dominant, ce qui biaiserait l'indicateur.
    """
    if title_norm_col and title_norm_col in df.columns:
        titles_norm = df[title_norm_col].fillna("").astype(str)
    else:
        titles_norm = df[title_col].fillna("").astype(str).map(normalize_text)

    df["occ_concept"] = titles_norm.map(lambda x: count_occurrences(x, concept_norm))
    df["occ_context"] = 0  # conservé pour compatibilité JSON, non utilisé dans le matching
    df["occ_up"] = titles_norm.map(lambda x: count_occurrences(x, up_norm))
    df["occ_down"] = titles_norm.map(lambda x: count_occurrences(x, down_norm))

    df["is_concept"] = (df["occ_concept"] > 0).astype("int8")
    df["is_context"] = 0
    df["is_match_broad"] = df["is_concept"].astype("int8")
    df["is_match_strict"] = df["is_match_broad"].astype("int8")
    df["is_match"] = df["is_match_broad"].astype("int8")

    # Direction 4 états — basée sur la présence binaire (pas sur le comptage brut)
    has_up = df["occ_up"] > 0
    has_down = df["occ_down"] > 0
    matched = df["is_match"] == 1

    direction = pd.Series(DIRECTION_FLAT, index=df.index, dtype="int8")
    direction.loc[matched & has_up & ~has_down] = DIRECTION_UP
    direction.loc[matched & has_down & ~has_up] = DIRECTION_DOWN
    direction.loc[matched & has_up & has_down] = DIRECTION_AMBIGUOUS
    df["direction"] = direction

    return df


# ──────────────────────────────────────────────────────────────────────────────
# LOGIQUE MÉTIER — AGRÉGATION TEMPORELLE
# ──────────────────────────────────────────────────────────────────────────────

def periodize(series: pd.Series, frequency: str) -> pd.Series:
    if frequency == "Trimestrielle":
        return series.dt.to_period("Q").dt.to_timestamp()
    if frequency == "Annuelle":
        return series.dt.to_period("Y").dt.to_timestamp()
    return series.dt.to_period("M").dt.to_timestamp()


def aggregate_by_period(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    out = df.assign(period_start=periodize(df["_date"], frequency)).copy()
    out["_match_mode"] = out["is_match"].astype(int)
    out["occurrences_concept"] = out["occ_concept"] * out["_match_mode"]
    out["up_flag"] = (out["direction"] == DIRECTION_UP).astype(int)
    out["down_flag"] = (out["direction"] == DIRECTION_DOWN).astype(int)
    out["ambiguous_flag"] = (out["direction"] == DIRECTION_AMBIGUOUS).astype(int)

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
            ambiguous_titles=("ambiguous_flag", "sum"),
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
    ambiguous_titles = int((df_tagged["direction"] == DIRECTION_AMBIGUOUS).sum())
    net_signal = up_titles - down_titles

    return pd.DataFrame([
        {"indicateur": "Titres analysés", "valeur": total_titles},
        {"indicateur": "Titres matchés (présence concept, binaire)", "valeur": matched_titles},
        {"indicateur": "Occurrences brutes concept (intensité)", "valeur": occ_concept_total},
        {"indicateur": "Signal UP (sens haussier)", "valeur": up_titles},
        {"indicateur": "Signal DOWN (sens baissier)", "valeur": down_titles},
        {"indicateur": "Signal AMBIGU (UP + DOWN simultanés)", "valeur": ambiguous_titles},
        {"indicateur": "Signal net (UP - DOWN)", "valeur": net_signal},
        {"indicateur": "Fréquence moyenne", "valeur": round(float(stats["frequency"].mean()), 4)},
        {"indicateur": "Fréquence médiane", "valeur": round(float(stats["frequency"].median()), 4)},
        {"indicateur": "Fréquence max", "valeur": round(float(stats["frequency"].max()), 4)},
        {"indicateur": "Volume moyen (titres matchés / période)", "valeur": round(float(stats["matched_titles"].mean()), 1)},
        {"indicateur": "Nombre de périodes", "valeur": int(len(stats))},
    ])


def build_top_channels(df_tagged: pd.DataFrame) -> pd.DataFrame:
    if "_channel" not in df_tagged.columns:
        return pd.DataFrame()
    work = df_tagged.copy()
    work["_match_mode"] = work["is_match"].astype(int)
    work["occurrences_concept"] = work["occ_concept"] * work["_match_mode"]
    work["up_flag"] = (work["direction"] == DIRECTION_UP).astype(int)
    work["down_flag"] = (work["direction"] == DIRECTION_DOWN).astype(int)
    work["ambiguous_flag"] = (work["direction"] == DIRECTION_AMBIGUOUS).astype(int)
    top = (
        work.groupby("_channel", as_index=False)
        .agg(
            total_titles=("_match_mode", "size"),
            matched_titles=("_match_mode", "sum"),
            strict_matched_titles=("is_match_strict", "sum"),
            occurrences_concept=("occurrences_concept", "sum"),
            up_titles=("up_flag", "sum"),
            down_titles=("down_flag", "sum"),
            ambiguous_titles=("ambiguous_flag", "sum"),
        )
        .sort_values("matched_titles", ascending=False)
    )
    top["frequency"] = top["matched_titles"] / top["total_titles"]
    top["net_signal"] = top["up_titles"] - top["down_titles"]
    return top.head(10)


# ──────────────────────────────────────────────────────────────────────────────
# LOGIQUE MÉTIER — STATISTIQUES PAR CHAÎNE
# ──────────────────────────────────────────────────────────────────────────────

def build_channel_stats(df_base: pd.DataFrame) -> pd.DataFrame:
    """Statistiques de couverture par chaîne sur le jeu de données complet (non filtré).

    Retourne pour chaque chaîne : date_min, date_max, n_obs, share_pct.
    Vectorisé via groupby pour éviter la boucle Python sur les groupes.
    """
    if "_channel" not in df_base.columns or "_date" not in df_base.columns:
        return pd.DataFrame()

    total = len(df_base)
    valid = df_base[df_base["_date"].notna()]
    if valid.empty:
        return pd.DataFrame()

    stats = (
        valid.groupby("_channel", as_index=False)
        .agg(date_min=("_date", "min"), date_max=("_date", "max"), n_obs=("_date", "count"))
    )
    stats["share_pct"] = (stats["n_obs"] / total * 100).round(2)
    return stats.sort_values("n_obs", ascending=False).reset_index(drop=True)


def build_decade_distribution(df_base: pd.DataFrame) -> pd.DataFrame:
    """Distribution des observations par décennie et par chaîne (format long).

    Retourne : _channel, decade (int), decade_label (str), n, pct (% dans la chaîne).
    """
    if "_channel" not in df_base.columns or "_date" not in df_base.columns:
        return pd.DataFrame()
    df = df_base[df_base["_date"].notna()].copy()
    df["decade"] = (df["_date"].dt.year // 10) * 10
    df["decade_label"] = df["decade"].astype(str) + "–" + (df["decade"] + 9).astype(str)
    cross = (
        df.groupby(["_channel", "decade_label", "decade"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    channel_totals = cross.groupby("_channel")["n"].transform("sum")
    cross["pct"] = (cross["n"] / channel_totals * 100).round(1)
    return cross.sort_values(["_channel", "decade"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS — GRAPHIQUES
# ──────────────────────────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# UI — DÉBUT DE L'APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

st.title("INA — Analyse thématique")
st.caption(
    "Détection de concepts dans les titres de journaux télévisés · "
    "Fréquence, volumes et sens du signal."
)

# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────

data_signature = HF_PARQUET_FILES
try:
    df_base, load_issues = load_clean_parquets_from_hf()
except Exception as _hf_exc:
    st.error(f"Erreur chargement HuggingFace : {_hf_exc}")
    st.stop()

with st.expander("Données chargées", expanded=False):
    st.caption(f"Source : `{HF_REPO_ID}` (HuggingFace Datasets)")
    for issue in load_issues:
        st.warning(issue)
    if not df_base.empty:
        st.success(
            f"{len(df_base):,} observations chargées depuis {len(data_signature)} fichier(s)."
        )
        st.caption(
            f"Période couverte : {df_base['_date'].min().strftime('%d/%m/%Y')} "
            f"→ {df_base['_date'].max().strftime('%d/%m/%Y')}"
        )

if df_base.empty:
    st.warning(f"Aucune donnée chargée depuis `{HF_REPO_ID}`.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE — DICTIONNAIRES
# ──────────────────────────────────────────────────────────────────────────────

if "dictionaries" not in st.session_state:
    _hf_token = st.secrets.get("HF_TOKEN", None)
    _hf_dicts = load_dictionaries_from_hf(_hf_token)
    st.session_state["dictionaries"] = _hf_dicts if _hf_dicts else load_dictionaries(DICTIONARY_PATH)
    st.session_state["_hf_write_token"] = _hf_token

# Les dicts sont normalisés au chargement (load_dictionaries) et à chaque
# sauvegarde (save_dictionaries). On fait confiance au session_state directement
# pour éviter de renormaliser tous les termes à chaque rerun.
dictionaries = st.session_state["dictionaries"]
if not isinstance(dictionaries, dict) or not dictionaries:
    dictionaries = clone_dictionaries(DEFAULT_DICTIONARIES)
    st.session_state["dictionaries"] = dictionaries

title_col = "title" if "title" in df_base.columns else None
has_source_channel = "_channel" in df_base.columns

if not title_col or "_date" not in df_base.columns:
    st.error("Colonnes minimales introuvables : il faut au moins `title` et `_date`.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# STATISTIQUES DES CHAÎNES (couverture du jeu de données)
# Calculées UNE SEULE FOIS par session (quand data_signature change),
# jamais répétées lors des reruns normaux (slider, filtres, etc.)
# ──────────────────────────────────────────────────────────────────────────────

# Cache invalidé uniquement si les fichiers parquet changent
if st.session_state.get("_ch_stats_sig") != data_signature:
    _t0 = time.perf_counter()
    _ch_stats = build_channel_stats(df_base)
    _decade_df = build_decade_distribution(df_base)
    st.session_state["_ch_stats_cache"] = (_ch_stats, _decade_df)
    st.session_state["_ch_stats_sig"] = data_signature
    LOGGER.info("Stats chaînes recalculées en %.0f ms", (time.perf_counter() - _t0) * 1000)

ch_stats, decade_df = st.session_state["_ch_stats_cache"]

with st.expander("Statistiques des chaînes — couverture du jeu de données", expanded=False):
    if ch_stats.empty:
        st.info("Pas de données de chaînes disponibles.")
    else:
        n_channels = len(ch_stats)
        total_obs = len(df_base)
        date_global_min = df_base["_date"].min().strftime("%d/%m/%Y")
        date_global_max = df_base["_date"].max().strftime("%d/%m/%Y")

        st.caption(
            f"**{n_channels} chaîne(s)** · **{total_obs:,} observations** · "
            f"Période totale du dataset : {date_global_min} → {date_global_max}"
        )

        # Tableau de couverture
        display_ch = ch_stats[["_channel", "date_min", "date_max", "n_obs", "share_pct"]].copy()
        display_ch["date_min"] = display_ch["date_min"].dt.date
        display_ch["date_max"] = display_ch["date_max"].dt.date
        display_ch.columns = ["Chaîne", "Début", "Fin", "Observations", "Part (%)"]

        st.dataframe(
            display_ch,
            column_config={
                "Part (%)": st.column_config.ProgressColumn(
                    "Part dans l'échantillon",
                    help="Part des observations de cette chaîne dans le total du dataset",
                    format="%.1f %%",
                    min_value=0,
                    max_value=100,
                ),
            },
            hide_index=True,
            width="stretch",
        )

        # Distribution par décennie
        if not decade_df.empty:
            st.markdown("**Distribution des observations par décennie** (% des obs. de chaque chaîne)")
            _sorted_decades = sorted(
                decade_df["decade_label"].unique(),
                key=lambda x: int(x.split("–")[0]),
            )
            fig_dec = px.bar(
                decade_df,
                x="decade_label",
                y="pct",
                color="_channel",
                barmode="group",
                category_orders={"decade_label": _sorted_decades},
                labels={
                    "decade_label": "Décennie",
                    "pct": "Part (%)",
                    "_channel": "Chaîne",
                },
                height=380,
            )
            fig_dec.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_dec, width="stretch")

        # Alertes sur cas limites
        for _, row in ch_stats.iterrows():
            n = int(row["n_obs"])
            span_days = (row["date_max"] - row["date_min"]).days
            channel_name = row["_channel"]

            if n < 50:
                st.warning(
                    f"Chaîne « {channel_name} » : série très courte ({n} observations). "
                    "Les statistiques sont peu fiables."
                )
            elif span_days > 0 and not decade_df.empty:
                ch_decades = decade_df[decade_df["_channel"] == channel_name]
                start_decade = (row["date_min"].year // 10) * 10
                end_decade = (row["date_max"].year // 10) * 10
                expected = set(range(start_decade, end_decade + 1, 10))
                actual = set(ch_decades["decade"].tolist())
                missing = expected - actual
                if missing:
                    missing_str = ", ".join(f"{d}–{d+9}" for d in sorted(missing))
                    st.info(
                        f"Chaîne « {channel_name} » : aucune donnée pour les décennies {missing_str} "
                        "(trou temporel probable)."
                    )


# ──────────────────────────────────────────────────────────────────────────────
# GESTION DES THÈMES ET DICTIONNAIRES
# ──────────────────────────────────────────────────────────────────────────────

themes = sorted(dictionaries.keys())
if not themes:
    st.session_state["dictionaries"] = clone_dictionaries(DEFAULT_DICTIONARIES)
    dictionaries = st.session_state["dictionaries"]
    themes = sorted(dictionaries.keys())

if "theme" not in st.session_state or st.session_state["theme"] not in themes:
    st.session_state["theme"] = themes[0]

dict_expander_open = st.session_state.pop("dict_expander_open", False)

with st.expander("Thèmes et dictionnaires", expanded=dict_expander_open):

    # Message flash après une action (création, renommage, suppression)
    if st.session_state.get("dict_flash"):
        st.success(st.session_state.pop("dict_flash"))

    # ── Créer un nouveau thème ──────────────────────────────────────────────
    st.markdown("### Créer un nouveau thème")
    st.caption(
        "Un thème correspond à un sujet d'analyse (ex. : inflation, emploi, logement). "
        "Donnez-lui un nom court, sans espace si possible."
    )
    with st.form("add_theme_form", clear_on_submit=True):
        col_a, col_b = st.columns([4, 1])
        new_theme_input = col_a.text_input(
            "Nom du nouveau thème",
            placeholder="ex. : energie, logement, sante...",
            label_visibility="collapsed",
        )
        add_submitted = col_b.form_submit_button("Créer le thème", use_container_width=True)

    if add_submitted:
        nt = new_theme_input.strip()
        if not nt:
            st.warning("Le nom du thème ne peut pas être vide.")
        elif nt in dictionaries:
            st.warning(f"Le thème « {nt} » existe déjà. Sélectionnez-le dans la barre latérale.")
        else:
            dictionaries[nt] = empty_theme_dictionary()
            st.session_state["dictionaries"] = dictionaries
            st.session_state["theme"] = nt
            st.session_state["dict_flash"] = (
                f"Thème « {nt} » créé. Complétez maintenant son dictionnaire ci-dessous, "
                "puis cliquez sur « Enregistrer »."
            )
            st.session_state["dict_expander_open"] = True
            try:
                save_dictionaries(DICTIONARY_PATH, dictionaries)
                save_dictionaries_to_hf(dictionaries, st.session_state.get("_hf_write_token"))
            except Exception as exc:
                LOGGER.exception("Écriture dictionnaire impossible (création) : %s", exc)
                st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")
            st.rerun()

    st.divider()

    # ── Modifier le thème actif ─────────────────────────────────────────────
    theme_edit = st.session_state["theme"]
    current_theme_dict = normalize_theme_dictionary(dictionaries.get(theme_edit, empty_theme_dictionary()))

    st.markdown(f"### Modifier le thème : **{theme_edit}**")

    # Étape 1 — Renommer
    st.markdown("**Étape 1 — Renommer ce thème** *(optionnel)*")
    with st.form("rename_theme_form", clear_on_submit=False):
        col_r1, col_r2 = st.columns([4, 1])
        rename_input = col_r1.text_input(
            "Nouveau nom",
            value=theme_edit,
            label_visibility="collapsed",
        )
        rename_submitted = col_r2.form_submit_button("Renommer", use_container_width=True)

    if rename_submitted:
        new_name = rename_input.strip()
        if not new_name or new_name == theme_edit:
            st.info("Nom identique ou vide — aucun changement.")
        elif new_name in dictionaries:
            st.warning(f"Un thème « {new_name} » existe déjà.")
        else:
            dictionaries[new_name] = dictionaries.pop(theme_edit)
            st.session_state["dictionaries"] = dictionaries
            st.session_state["theme"] = new_name
            st.session_state["dict_flash"] = f"Thème renommé : « {theme_edit} » → « {new_name} »."
            st.session_state["dict_expander_open"] = True
            try:
                save_dictionaries(DICTIONARY_PATH, dictionaries)
                save_dictionaries_to_hf(dictionaries, st.session_state.get("_hf_write_token"))
            except Exception as exc:
                LOGGER.exception("Écriture dictionnaire impossible (renommage) : %s", exc)
                st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")
            st.rerun()

    # Étapes 2–4 + Sauvegarde
    st.markdown("**Étape 2 — Mots-clés du concept**")
    st.caption(
        "Un titre sera retenu si il contient **au moins un** de ces mots ou expressions. "
        "Écrivez **un mot ou une expression par ligne**. "
        "Les majuscules et les accents sont ignorés automatiquement."
    )

    with st.form("edit_theme_form"):
        concept_text = st.text_area(
            "Mots-clés concept",
            value="\n".join(current_theme_dict["concept"]),
            height=150,
            placeholder="inflation\nhausse des prix\nindice des prix\nipc\n...",
            label_visibility="collapsed",
            help="Un mot-clé par ligne. Les doublons et les variations d'accents sont gérés automatiquement.",
        )

        st.markdown("**Étape 3 — Termes indiquant une hausse (sens UP)**")
        st.caption(
            "Si un titre matché contient l'un de ces termes **et aucun terme DOWN**, "
            "le signal sera orienté à la hausse. "
            "Si les deux listes sont vides, le sens ne sera pas calculé."
        )
        up_text = st.text_area(
            "Termes UP",
            value="\n".join(current_theme_dict["up"]),
            height=120,
            placeholder="hausse\naugmente\ngrimpe\nexplose\nflambée\n...",
            label_visibility="collapsed",
        )

        st.markdown("**Étape 4 — Termes indiquant une baisse (sens DOWN)**")
        st.caption(
            "Si un titre matché contient l'un de ces termes **et aucun terme UP**, "
            "le signal sera orienté à la baisse. "
            "Si les deux sont présents, le titre est marqué « ambigu » (ni UP ni DOWN)."
        )
        down_text = st.text_area(
            "Termes DOWN",
            value="\n".join(current_theme_dict["down"]),
            height=120,
            placeholder="baisse\nrecul\ndiminue\nralentit\nchute\n...",
            label_visibility="collapsed",
        )

        st.markdown("**Étape 5 — Enregistrer**")
        save_submitted = st.form_submit_button(
            "Enregistrer le dictionnaire", use_container_width=True, type="primary"
        )

    if save_submitted:
        new_concept = [k.strip() for k in concept_text.splitlines() if k.strip()]
        new_up = [k.strip() for k in up_text.splitlines() if k.strip()]
        new_down = [k.strip() for k in down_text.splitlines() if k.strip()]
        dictionaries[theme_edit] = {
            "concept": new_concept,
            "context": current_theme_dict.get("context", []),
            "up": new_up,
            "down": new_down,
        }
        st.session_state["dictionaries"] = dictionaries
        try:
            save_dictionaries(DICTIONARY_PATH, dictionaries)
            save_dictionaries_to_hf(dictionaries, st.session_state.get("_hf_write_token"))
            st.success(
                f"Dictionnaire « {theme_edit} » enregistré : "
                f"{len(new_concept)} concept(s) · {len(new_up)} terme(s) UP · {len(new_down)} terme(s) DOWN."
            )
        except Exception as exc:
            LOGGER.exception("Écriture dictionnaire impossible (enregistrement) : %s", exc)
            st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")

    st.divider()

    # ── Zone de danger ──────────────────────────────────────────────────────
    st.markdown("### Zone de danger")

    col_danger1, col_danger2 = st.columns(2)

    with col_danger1:
        st.markdown("**Supprimer ce thème**")
        remaining_themes = [t for t in themes if t != theme_edit]
        if not remaining_themes:
            st.info("Impossible de supprimer le seul thème restant.")
        else:
            confirm_key = f"confirm_delete_{theme_edit}"
            confirmed = st.checkbox(
                f"Je confirme la suppression définitive de « {theme_edit} »",
                key=confirm_key,
                value=False,
            )
            if st.button(
                "Supprimer ce thème",
                disabled=not confirmed,
                type="primary" if confirmed else "secondary",
                key=f"delete_btn_{theme_edit}",
            ):
                del dictionaries[theme_edit]
                st.session_state["dictionaries"] = dictionaries
                st.session_state["theme"] = sorted(dictionaries.keys())[0]
                st.session_state["dict_flash"] = f"Thème « {theme_edit} » supprimé."
                st.session_state["dict_expander_open"] = True
                try:
                    save_dictionaries(DICTIONARY_PATH, dictionaries)
                    save_dictionaries_to_hf(dictionaries, st.session_state.get("_hf_write_token"))
                except Exception as exc:
                    LOGGER.exception("Écriture dictionnaire impossible (suppression) : %s", exc)
                    st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")
                st.rerun()

    with col_danger2:
        st.markdown("**Réinitialiser tous les dictionnaires**")
        st.caption(
            "Recharge les dictionnaires par défaut (inflation, emploi, etc.). "
            "Toutes vos modifications seront perdues."
        )
        if st.button("Réinitialiser par défaut", key="reset_defaults_btn"):
            st.session_state["dictionaries"] = clone_dictionaries(DEFAULT_DICTIONARIES)
            st.session_state["theme"] = sorted(DEFAULT_DICTIONARIES.keys())[0]
            try:
                save_dictionaries(DICTIONARY_PATH, st.session_state["dictionaries"])
                save_dictionaries_to_hf(st.session_state["dictionaries"], st.session_state.get("_hf_write_token"))
            except Exception as exc:
                LOGGER.exception("Écriture dictionnaire impossible (reset) : %s", exc)
                st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# BARRE LATÉRALE — PARAMÈTRES
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Paramètres")

# Thème actif (reflète l'état après éventuel rerun du gestionnaire)
themes = sorted(dictionaries.keys())
current_theme = st.session_state.get("theme", themes[0] if themes else "")
theme = st.sidebar.selectbox(
    "Thème actif",
    options=themes,
    index=themes.index(current_theme) if current_theme in themes else 0,
    help="Le thème dont les concepts sont recherchés dans les titres. "
         "Créez ou modifiez des thèmes dans la section « Thèmes et dictionnaires » ci-dessus.",
)
st.session_state["theme"] = theme

frequency = st.sidebar.selectbox(
    "Fréquence",
    ["Mensuelle", "Trimestrielle", "Annuelle"],
    index=0,
    help="Période d'agrégation des résultats.",
)

count_mode = st.sidebar.radio(
    "Mode de comptage",
    options=["Binaire (présence / absence)", "Intensité (occurrences brutes)"],
    index=0,
    help=(
        "**Binaire** : chaque titre vaut 1 s'il contient le concept, 0 sinon. "
        "C'est l'approche recommandée pour construire un signal de saillance médiatique. \n\n"
        "**Intensité** : somme des occurrences brutes du concept dans les titres. "
        "Utile pour vérifier la robustesse, mais attention au surcompte."
    ),
)
binary_mode = count_mode.startswith("Binaire")

with st.sidebar.expander("Diagnostics", expanded=False):
    st.caption(f"Log erreurs : `{LOG_PATH.as_posix()}`")

    # Affichage de l'état des caches session pour vérifier les gains de perf
    if st.checkbox("Afficher l'état des caches", value=False):
        tagging_cache_info = st.session_state.get("_tagged_theme_cache", {})
        agg_cache_info = st.session_state.get("_agg_cache", {})
        ch_stats_cached = "_ch_stats_cache" in st.session_state
        kw_cached = "_kw_cache_key" in st.session_state
        st.caption(
            f"Cache tagging : {len(tagging_cache_info)} entrée(s)  \n"
            f"Cache agrégation : {len(agg_cache_info)} entrée(s)  \n"
            f"Cache stats chaînes : {'oui' if ch_stats_cached else 'non'}  \n"
            f"Cache keywords : {'oui' if kw_cached else 'non'}"
        )

    if st.checkbox("Afficher les 30 dernières lignes du log", value=False):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                last_lines = f.readlines()[-30:]
            st.code("".join(last_lines) if last_lines else "(log vide)", language="text")
        except Exception as exc:
            LOGGER.exception("Lecture du log impossible : %s", exc)
            st.warning(f"Lecture du log impossible ({exc})")


# ──────────────────────────────────────────────────────────────────────────────
# MARQUAGE (TAGGING)
# ──────────────────────────────────────────────────────────────────────────────

theme_dict = normalize_theme_dictionary(
    st.session_state["dictionaries"].get(theme, empty_theme_dictionary())
)

# Cache les keywords normalisés dans session_state pour éviter de relancer
# unicodedata.normalize sur chaque terme à chaque rerun.
# Clé = contenu brut du dictionnaire du thème actif.
_kw_cache_key = (theme, tuple(theme_dict["concept"]), tuple(theme_dict["up"]), tuple(theme_dict["down"]))
if st.session_state.get("_kw_cache_key") != _kw_cache_key:
    st.session_state["_kw_cache_key"] = _kw_cache_key
    st.session_state["_kw_cache"] = (
        prepare_keywords(theme_dict["concept"]),
        prepare_keywords(theme_dict["up"]),
        prepare_keywords(theme_dict["down"]),
    )
concept_norm, up_norm, down_norm = st.session_state["_kw_cache"]

if not concept_norm:
    st.info(
        f"Le thème « {theme} » n'a pas encore de mots-clés concept. "
        "Ouvrez la section **Thèmes et dictionnaires** ci-dessus et complétez l'étape 2."
    )
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
        stop_with_error_log("Erreur pendant le marquage des titres.", "add_tagging_columns_hier", exc)
    while len(tagging_cache) > 2:
        tagging_cache.pop(next(iter(tagging_cache)))

df_tagged = tagging_cache[tag_cache_key]


# ──────────────────────────────────────────────────────────────────────────────
# FILTRE PÉRIODE
# ──────────────────────────────────────────────────────────────────────────────

min_date_ts = df_tagged["_date"].min()
max_date_ts = df_tagged["_date"].max()
default_start_ts = max_date_ts - pd.DateOffset(years=2)
if default_start_ts < min_date_ts:
    default_start_ts = min_date_ts

try:
    if st.sidebar.button("Plage complète", key="date_full_range"):
        st.session_state["_date_range"] = (min_date_ts.date(), max_date_ts.date())

    _default_range = st.session_state.get(
        "_date_range",
        (default_start_ts.date(), max_date_ts.date()),
    )
    date_range = st.sidebar.date_input(
        "Période",
        value=_default_range,
        min_value=min_date_ts.date(),
        max_value=max_date_ts.date(),
        format="DD/MM/YYYY",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        date_start = pd.Timestamp(date_range[0])
        date_end = pd.Timestamp(date_range[1])
        st.session_state["_date_range"] = (date_range[0], date_range[1])
    else:
        date_start = date_end = pd.Timestamp(date_range[0]) if date_range else pd.Timestamp(min_date_ts)
    if date_start > date_end:
        st.sidebar.error("La date de début doit être avant la date de fin.")
        st.stop()
except Exception as exc:
    stop_with_error_log("Erreur lors du filtre de période.", "date_input Periode", exc)

# Masque de période (appliqué sur df_tagged directement, sans copy intermédiaire)
_period_mask = (
    (df_tagged["_date"] >= pd.Timestamp(date_start))
    & (df_tagged["_date"] <= pd.Timestamp(date_end))
)
if not _period_mask.any():
    st.warning("Aucune donnée dans la période sélectionnée.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# FILTRE CHAÎNES
# ──────────────────────────────────────────────────────────────────────────────

all_channels = sorted(
    c for c in df_tagged.loc[_period_mask, "_channel"].dropna().unique().tolist()
    if str(c).strip()
)
if not all_channels:
    all_channels = ["(sans chaîne)"]

st.sidebar.markdown("---")
_cc1, _cc2 = st.sidebar.columns(2)
if _cc1.button("Toutes", key="ch_all"):
    st.session_state["_ch_sel"] = all_channels
if _cc2.button("Aucune", key="ch_none"):
    st.session_state["_ch_sel"] = []

# Réinitialise si la sélection contient des chaînes absentes de la période courante
_current_sel = st.session_state.get("_ch_sel", all_channels)
_valid_sel = [c for c in _current_sel if c in all_channels]
if not _valid_sel:
    _valid_sel = all_channels
st.session_state["_ch_sel"] = _valid_sel

selected_channels = st.sidebar.multiselect(
    "Filtre chaînes",
    options=all_channels,
    default=_valid_sel,
)
if not selected_channels:
    st.warning("Sélectionnez au moins une chaîne.")
    st.stop()
st.session_state["_ch_sel"] = selected_channels

# df_period : période seule, avant filtre chaîne (utilisé pour top_channels)
# df_filtered : période + filtre chaîne (utilisé pour tout le reste)
df_period = df_tagged[_period_mask]
df_filtered = df_tagged[_period_mask & df_tagged["_channel"].isin(selected_channels)]

if df_filtered.empty:
    st.warning("Aucune ligne après filtre chaîne.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# AGRÉGATION
# Cache dans session_state : évite de relancer les groupby (coûteux) quand
# seule la vue change (ex. bascule Binaire/Intensité, qui est un pur toggle
# d'affichage et ne modifie ni les filtres ni les données sous-jacentes).
# Clé = (tag_cache_key, date_start, date_end, chaînes sélectionnées, fréquence)
# ──────────────────────────────────────────────────────────────────────────────

agg_cache_key = json.dumps({
    "tag": tag_cache_key,
    "date_start": str(date_start),
    "date_end": str(date_end),
    "channels": sorted(selected_channels),
    "frequency": frequency,
}, sort_keys=True)

agg_cache = st.session_state.setdefault("_agg_cache", {})

if agg_cache_key not in agg_cache:
    try:
        _t0 = time.perf_counter()
        stats = aggregate_by_period(df_filtered, frequency=frequency)
        desc = build_descriptive_table(stats, df_filtered)
        top_channels = build_top_channels(df_period)
        agg_cache[agg_cache_key] = (stats, desc, top_channels)
        LOGGER.info(
            "Agrégation recalculée en %.0f ms (freq=%s, n=%d)",
            (time.perf_counter() - _t0) * 1000, frequency, len(df_filtered),
        )
        # Limite la taille du cache (5 entrées max)
        while len(agg_cache) > 5:
            agg_cache.pop(next(iter(agg_cache)))
    except Exception as exc:
        stop_with_error_log("Erreur pendant le calcul des indicateurs.", "aggregate/build tables", exc)

stats, desc, top_channels = agg_cache[agg_cache_key]


# ──────────────────────────────────────────────────────────────────────────────
# KPI — MÉTRIQUES CLÉS
# ──────────────────────────────────────────────────────────────────────────────

n_total = len(df_filtered)
n_matched = int(df_filtered["is_match"].sum())
occ_concept_total = int(df_filtered.loc[df_filtered["is_match"] == 1, "occ_concept"].sum())
freq_mean = stats["frequency"].mean()
net_signal_total = int(stats["net_signal"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Titres analysés",
    f"{n_total:,}",
    help="Nombre total d'observations dans la période et les chaînes sélectionnées.",
)
k2.metric(
    "Titres matchés",
    f"{n_matched:,}",
    help="Titres contenant au moins un mot-clé concept (logique binaire : 0 ou 1 par titre).",
)
if binary_mode:
    k3.metric(
        "Fréquence moyenne",
        f"{freq_mean:.1%}",
        help="Part des titres matchés sur l'ensemble des titres analysés, moyennée sur les périodes.",
    )
    k4.metric(
        "Signal net (UP – DOWN)",
        f"{net_signal_total:+,}",
        help="Somme des signaux nets par période. Positif = tendance haussière dominante.",
    )
else:
    k3.metric(
        "Occurrences brutes",
        f"{occ_concept_total:,}",
        help="Somme des occurrences du concept dans les titres matchés (peut être > 1 par titre).",
    )
    k4.metric(
        "Fréquence moyenne",
        f"{freq_mean:.1%}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GRAPHIQUE — FRÉQUENCE
# ──────────────────────────────────────────────────────────────────────────────

st.subheader(f"Fréquence du thème « {theme} »")
st.caption(
    "Part des titres contenant au moins un mot-clé concept (logique binaire : 1 titre = 0 ou 1). "
    if binary_mode
    else "Part des titres matchés sur l'ensemble des titres de la période."
)

fig_freq = px.line(
    stats,
    x="period_start",
    y="frequency",
    markers=True,
    render_mode="svg",
    title=f"Fréquence ({frequency.lower()}) — thème « {theme} »",
    labels={"period_start": "Date", "frequency": "Part de titres matchés"},
)
apply_time_axis_controls(fig_freq)
fig_freq.update_layout(height=420)
st.plotly_chart(fig_freq, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# GRAPHIQUE — VOLUMES
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("Volumes")

if binary_mode:
    vol_y = ["matched_titles"]
    vol_title = f"Titres matchés ({frequency.lower()}) — présence binaire"
    vol_labels = {"period_start": "Date", "value": "Titres matchés", "variable": "Série"}
else:
    vol_y = ["matched_titles", "occurrences_concept"]
    vol_title = f"Volumes ({frequency.lower()}) — titres matchés et occurrences brutes"
    vol_labels = {"period_start": "Date", "value": "Volume", "variable": "Série"}

fig_vol = px.line(
    stats,
    x="period_start",
    y=vol_y,
    markers=True,
    render_mode="svg",
    title=vol_title,
    labels=vol_labels,
)
apply_time_axis_controls(fig_vol)
fig_vol.update_layout(height=420)
st.plotly_chart(fig_vol, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# GRAPHIQUE — SENS DU SIGNAL
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("Sens du signal")

if not (up_norm or down_norm):
    st.info(
        "Aucun mot-clé UP/DOWN dans le dictionnaire de ce thème. "
        "Ajoutez des termes dans **Thèmes et dictionnaires** (étapes 3 et 4) pour activer l'analyse de sens."
    )
else:
    up_total = int(stats["up_titles"].sum())
    down_total = int(stats["down_titles"].sum())
    ambiguous_total = int(stats["ambiguous_titles"].sum())
    non_zero_periods = int((stats["net_signal"] != 0).sum())

    col_sig1, col_sig2, col_sig3 = st.columns(3)
    col_sig1.metric("Titres UP", f"{up_total:,}")
    col_sig2.metric("Titres DOWN", f"{down_total:,}")
    col_sig3.metric(
        "Titres ambigus",
        f"{ambiguous_total:,}",
        help="Titres contenant à la fois des termes UP et DOWN. Ils sont exclus du signal net.",
    )

    if ambiguous_total > 0:
        ambiguous_pct = round(ambiguous_total / max(n_matched, 1) * 100, 1)
        st.caption(
            f"**{ambiguous_total} titre(s) ambigu(s)** ({ambiguous_pct} % des matchés) : "
            "ces titres contiennent simultanément des termes UP et DOWN — ils sont exclus du signal net. "
            "Si ce nombre est élevé, vérifiez si vos listes UP/DOWN contiennent des termes qui se chevauchent."
        )

    if up_total == 0 and down_total == 0:
        st.warning(
            "Aucun titre n'a activé les termes UP/DOWN sur la période et les filtres courants. "
            "Essayez d'élargir la période ou de retirer des filtres chaînes."
        )

    signal_series = st.multiselect(
        "Séries à afficher sur le graphique",
        options=["net_signal", "up_titles", "down_titles", "ambiguous_titles"],
        default=["net_signal"],
        help=(
            "net_signal = UP – DOWN · "
            "ambiguous_titles = titres avec UP et DOWN simultanés (exclus du signal net)"
        ),
    )
    if not signal_series:
        signal_series = ["net_signal"]

    fig_signal = px.bar(
        stats,
        x="period_start",
        y=signal_series,
        barmode="group",
        title=f"Sens du signal ({frequency.lower()}) — thème « {theme} »",
        labels={"period_start": "Date", "value": "Titres", "variable": "Indicateur"},
    )
    apply_time_axis_controls(fig_signal)
    fig_signal.update_layout(height=420)
    st.plotly_chart(fig_signal, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# STATISTIQUES DESCRIPTIVES
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("Statistiques descriptives")
st.dataframe(desc, hide_index=True, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# TOP CHAÎNES (sur la période, avant filtre chaîne)
# ──────────────────────────────────────────────────────────────────────────────

if has_source_channel:
    st.subheader("Top 10 chaînes — titres matchés sur la période")
    st.caption(
        "Calculé sur la période sélectionnée, **avant** application du filtre chaîne, "
        "pour permettre la comparaison entre chaînes."
    )

    if not top_channels.empty:
        display_top = top_channels.copy()
        display_top["frequency_pct"] = (display_top["frequency"] * 100).round(1)
        st.dataframe(
            display_top[["_channel", "total_titles", "matched_titles", "frequency_pct",
                          "up_titles", "down_titles", "ambiguous_titles", "net_signal"]],
            column_config={
                "_channel": st.column_config.TextColumn("Chaîne"),
                "total_titles": st.column_config.NumberColumn("Titres total"),
                "matched_titles": st.column_config.NumberColumn("Matchés"),
                "frequency_pct": st.column_config.ProgressColumn(
                    "Fréquence (%)",
                    format="%.1f %%",
                    min_value=0,
                    max_value=100,
                ),
                "up_titles": st.column_config.NumberColumn("UP"),
                "down_titles": st.column_config.NumberColumn("DOWN"),
                "ambiguous_titles": st.column_config.NumberColumn("Ambigus"),
                "net_signal": st.column_config.NumberColumn("Signal net"),
            },
            hide_index=True,
            width="stretch",
        )

        fig_top = px.bar(
            top_channels,
            x="_channel",
            y="matched_titles",
            color="net_signal",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Top chaînes — titres matchés (couleur = signal net)",
            labels={"_channel": "Chaîne", "matched_titles": "Titres matchés", "net_signal": "Signal net"},
        )
        fig_top.update_layout(height=400)
        st.plotly_chart(fig_top, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# APERÇU DES TITRES MATCHÉS
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("Aperçu des titres matchés")

DIRECTION_LABELS = {
    DIRECTION_FLAT: "—",
    DIRECTION_UP: "UP",
    DIRECTION_DOWN: "DOWN",
    DIRECTION_AMBIGUOUS: "Ambigu",
}

preview_cols = [
    c
    for c in ["_date", "_channel", title_col, "occ_concept", "occ_up", "occ_down", "direction", "source_file"]
    if c in df_filtered.columns
]
preview_df = (
    df_filtered[df_filtered["is_match"] == 1]
    .sort_values(["occ_concept", "_date"], ascending=[False, False])
    .head(300)[preview_cols]
    .copy()
)
if "direction" in preview_df.columns:
    preview_df["direction"] = preview_df["direction"].map(DIRECTION_LABELS).fillna("—")

st.dataframe(
    preview_df,
    hide_index=True,
    width="stretch",
    column_config={
        "_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "_channel": st.column_config.TextColumn("Chaîne"),
        "title": st.column_config.TextColumn("Titre"),
        "occ_concept": st.column_config.NumberColumn("Concept"),
        "occ_up": st.column_config.NumberColumn("UP"),
        "occ_down": st.column_config.NumberColumn("DOWN"),
        "source_file": st.column_config.TextColumn("Source"),
    },
)
