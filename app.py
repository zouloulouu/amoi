import json
import logging
import os
import re
import unicodedata
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
) -> Tuple[pd.DataFrame, List[str]]:
    issues: List[str] = []
    if not os.path.isdir(folder):
        LOGGER.warning("Dossier clean introuvable: %s", folder)
        issues.append(f"Dossier clean introuvable: {folder}")
        return pd.DataFrame(), issues
    if not signature:
        LOGGER.warning("Aucun fichier parquet trouve dans %s", folder)
        issues.append(f"Aucun fichier `.parquet` trouve dans {folder}")
        return pd.DataFrame(), issues

    frames = []
    for file_name, _, _ in signature:
        path = os.path.join(folder, file_name)
        try:
            df = pd.read_parquet(path)
            missing = [c for c in REQUIRED_CLEAN_COLUMNS if c not in df.columns]
            if missing:
                LOGGER.warning("Colonnes clean manquantes dans %s: %s", file_name, ", ".join(missing))
                issues.append(f"Colonnes clean manquantes dans {file_name}: {', '.join(missing)}")
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
            normalized["_channel"] = normalized["_channel"].fillna("(sans chaine)").astype(str).str.strip()

            frames.append(normalized)
            LOGGER.info("Clean parquet charge: %s | lignes=%s", file_name, len(normalized))
        except Exception as exc:
            LOGGER.exception("Lecture impossible sur %s: %s", file_name, exc)
            issues.append(f"Lecture impossible: {file_name} ({exc})")

    if not frames:
        return pd.DataFrame(), issues

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, issues


def prepare_keywords(keywords: List[str]) -> List[str]:
    normalized = [normalize_text(k) for k in keywords if str(k).strip()]
    dedup = sorted(set(k for k in normalized if k))
    return dedup


def count_occurrences(text_norm: str, keywords_norm: List[str]) -> int:
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
    if title_norm_col and title_norm_col in df.columns:
        titles_norm = df[title_norm_col].fillna("").astype(str)
    else:
        titles_norm = df[title_col].fillna("").astype(str).map(normalize_text)

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
df_base, load_issues = load_clean_parquets_from_folder(active_clean_dir.as_posix(), data_signature)
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
    st.caption(f"Log erreurs: `{LOG_PATH.as_posix()}`")
    if st.checkbox("Afficher les 30 dernieres lignes du log", value=False):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                last_lines = f.readlines()[-30:]
            st.code("".join(last_lines) if last_lines else "(log vide)", language="text")
        except Exception as exc:
            LOGGER.exception("Lecture du log impossible: %s", exc)
            st.warning(f"Lecture du log impossible ({exc})")

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
