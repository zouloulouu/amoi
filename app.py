import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="INA - Dictionnaire (simple)", layout="wide")

DATA_DIR = "data"
DICTIONARY_PATH = "dictionaries.json"

DEFAULT_DICTIONARIES: Dict[str, List[str]] = {
    "inflation": [
        "inflation",
        "pouvoir d'achat",
        "cout de la vie",
        "coût de la vie",
        "indice des prix",
        "ipc",
    ]
}


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text


def canon_colname(name: str) -> str:
    c = normalize_text(str(name).replace("\u00a0", " "))
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^a-z0-9_]", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def normalize_channel(value: str) -> str:
    s = normalize_text(str(value))
    s = s.replace("france2", "france 2").replace("fr2", "france 2").replace("f2", "france 2")
    s = s.replace("t_f_1", "tf1").replace("tf1_", "tf1")
    if "france 2" in s or s == "2":
        return "France 2"
    if "tf1" in s:
        return "TF1"
    return str(value).strip()


def load_dictionaries(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return DEFAULT_DICTIONARIES.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return DEFAULT_DICTIONARIES.copy()
        data = json.loads(raw)
        out: Dict[str, List[str]] = {}
        if isinstance(data, dict):
            for k, values in data.items():
                if isinstance(k, str) and isinstance(values, list):
                    clean = [str(v).strip() for v in values if str(v).strip()]
                    out[k.strip()] = clean
        return out if out else DEFAULT_DICTIONARIES.copy()
    except Exception:
        return DEFAULT_DICTIONARIES.copy()


def save_dictionaries(path: str, dictionaries: Dict[str, List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dictionaries, f, ensure_ascii=False, indent=2)


@st.cache_data(show_spinner=False)
def load_parquets_from_folder(folder: str, selected_files: List[str], max_rows_per_file: int) -> pd.DataFrame:
    if not os.path.isdir(folder):
        return pd.DataFrame()
    files = [f for f in selected_files if str(f).lower().endswith(".parquet")]
    if not files:
        return pd.DataFrame()

    frames = []
    for file_name in sorted(files):
        path = os.path.join(folder, file_name)
        try:
            df = pd.read_parquet(path)
            if max_rows_per_file > 0:
                df = df.head(int(max_rows_per_file))
            df.columns = [canon_colname(c) for c in df.columns]
            df["source_file"] = file_name
            frames.append(df)
        except Exception as exc:
            st.warning(f"Lecture impossible: {file_name} ({exc})")

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_uploaded_parquets(uploaded_files, max_rows_per_file: int) -> pd.DataFrame:
    frames = []
    for uploaded in uploaded_files:
        try:
            df = pd.read_parquet(uploaded)
            if max_rows_per_file > 0:
                df = df.head(int(max_rows_per_file))
            df.columns = [canon_colname(c) for c in df.columns]
            df["source_file"] = getattr(uploaded, "name", "uploaded.parquet")
            frames.append(df)
        except Exception as exc:
            st.warning(f"Upload illisible: {getattr(uploaded, 'name', 'uploaded.parquet')} ({exc})")
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def detect_column(columns: List[str], candidates: List[str], token: str) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    token_matches = [c for c in columns if token in c]
    return token_matches[0] if token_matches else None


def parse_datetime(df: pd.DataFrame, date_col: str, time_col: Optional[str]) -> pd.Series:
    dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if not time_col or time_col not in df.columns:
        return dt

    clean_time = df[time_col].astype(str).str.strip().str.lower()
    clean_time = clean_time.str.replace("h", ":", regex=False)
    clean_time = clean_time.str.replace(r"[^0-9:]", "", regex=True)

    def normalize_clock(x: str) -> str:
        if not x or x == "nan":
            return ""
        if x.isdigit() and len(x) == 4:
            return f"{x[:2]}:{x[2:]}:00"
        if re.match(r"^\d{1,2}:\d{2}$", x):
            return x + ":00"
        if re.match(r"^\d{1,2}:\d{2}:\d{2}$", x):
            return x
        return ""

    fixed = clean_time.map(normalize_clock)
    combo = pd.to_datetime(dt.dt.date.astype(str) + " " + fixed, errors="coerce")
    return combo.where(combo.notna(), dt)


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


def add_tagging_columns(df: pd.DataFrame, title_col: str, keywords_norm: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["_title"] = out[title_col].fillna("").astype(str)
    out["_title_norm"] = out["_title"].map(normalize_text)
    out["occurrences"] = out["_title_norm"].map(lambda x: count_occurrences(x, keywords_norm))
    out["is_match"] = (out["occurrences"] > 0).astype(int)
    return out


def periodize(series: pd.Series, frequency: str) -> pd.Series:
    if frequency == "Trimestrielle":
        return series.dt.to_period("Q").dt.to_timestamp()
    if frequency == "Annuelle":
        return series.dt.to_period("Y").dt.to_timestamp()
    return series.dt.to_period("M").dt.to_timestamp()


def aggregate_by_period(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    out = df.copy()
    out["period_start"] = periodize(out["_date"], frequency)
    stats = (
        out.groupby("period_start", as_index=False)
        .agg(
            total_titles=("_title", "size"),
            matched_titles=("is_match", "sum"),
            occurrences=("occurrences", "sum"),
        )
        .sort_values("period_start")
    )
    stats["frequency"] = stats["matched_titles"] / stats["total_titles"]
    stats["occurrences_per_100_titles"] = (stats["occurrences"] / stats["total_titles"]) * 100
    return stats


def build_descriptive_table(stats: pd.DataFrame, df_tagged: pd.DataFrame) -> pd.DataFrame:
    if stats.empty or df_tagged.empty:
        return pd.DataFrame(columns=["indicateur", "valeur"])
    total_titles = int(len(df_tagged))
    matched_titles = int(df_tagged["is_match"].sum())
    occ_total = int(df_tagged["occurrences"].sum())
    return pd.DataFrame(
        [
            {"indicateur": "Titres analyses", "valeur": total_titles},
            {"indicateur": "Titres avec au moins 1 match", "valeur": matched_titles},
            {"indicateur": "Occurrences totales", "valeur": occ_total},
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
    top = (
        df_tagged.groupby("_channel", as_index=False)
        .agg(
            total_titles=("_title", "size"),
            matched_titles=("is_match", "sum"),
            occurrences=("occurrences", "sum"),
        )
        .sort_values("matched_titles", ascending=False)
    )
    top["frequency"] = top["matched_titles"] / top["total_titles"]
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
    st.write("Charge des fichiers `.parquet` ou utilise le dossier local `data/`.")
    max_rows_per_file = st.number_input(
        "Limiter lignes par fichier (0 = tout)",
        min_value=0,
        max_value=2_000_000,
        value=200_000,
        step=50_000,
    )
    local_parquet_files = sorted(
        f for f in os.listdir(DATA_DIR) if f.lower().endswith(".parquet")
    ) if os.path.isdir(DATA_DIR) else []
    selected_local_files = st.multiselect(
        "Fichiers locaux a charger",
        options=local_parquet_files,
        default=[],
    )
    uploaded_files = st.file_uploader(
        "Uploader des fichiers Parquet", type=["parquet"], accept_multiple_files=True
    )

if uploaded_files:
    df_raw = load_uploaded_parquets(uploaded_files, max_rows_per_file=max_rows_per_file)
elif selected_local_files:
    df_raw = load_parquets_from_folder(
        DATA_DIR,
        selected_files=selected_local_files,
        max_rows_per_file=max_rows_per_file,
    )
else:
    st.info("Selectionne au moins un fichier local ou upload un/des fichier(s) Parquet.")
    st.stop()

if df_raw.empty:
    st.warning("Aucune donnee chargee.")
    st.stop()

if "dictionaries" not in st.session_state:
    st.session_state["dictionaries"] = load_dictionaries(DICTIONARY_PATH)

dictionaries = st.session_state["dictionaries"]
if not dictionaries:
    dictionaries = DEFAULT_DICTIONARIES.copy()
    st.session_state["dictionaries"] = dictionaries

columns = list(df_raw.columns)
default_title = detect_column(
    columns,
    ["titre_propre", "titre", "title", "intitule", "libelle", "titre_programme", "titre_collection"],
    "titre",
)
default_date = detect_column(columns, ["date_diffusion", "date", "date_notice", "date_publication"], "date")
default_time = detect_column(columns, ["heure_diffusion", "heure", "time", "horaire"], "heure")
default_channel = detect_column(columns, ["chaine", "channel"], "chaine")

if not default_title or not default_date:
    st.error("Colonnes minimales introuvables: il faut au moins une colonne titre et une colonne date.")
    st.stop()

st.sidebar.header("Parametres")
frequency = st.sidebar.selectbox("Frequence", ["Mensuelle", "Trimestrielle", "Annuelle"], index=0)
title_col = default_title
date_col = default_date
time_col = default_time if default_time in columns else None
channel_col = default_channel if default_channel in columns else None
normalize_channels = True

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
        "2. Ensuite, saisis les mots-cles (1 ligne = 1 mot-cle), puis clique `Enregistrer dictionnaire`."
    )
    new_theme = st.text_input("Nouveau theme", value="")
    if st.button("Ajouter un theme"):
        nt = new_theme.strip()
        if not nt:
            st.warning("Nom de theme vide.")
        elif nt in dictionaries:
            st.warning("Ce theme existe deja.")
        else:
            dictionaries[nt] = []
            st.session_state["dictionaries"] = dictionaries
            st.session_state["theme"] = nt
            st.session_state["dict_flash"] = f"Theme '{nt}' cree. Ajoute maintenant ses mots-cles."
            try:
                save_dictionaries(DICTIONARY_PATH, dictionaries)
            except Exception as exc:
                st.warning(f"Ecriture du fichier dictionnaire impossible ({exc}).")
            st.rerun()

    current_keywords = dictionaries.get(theme, [])
    keyword_text = st.text_area(
        f"Mots-cles pour '{theme}'",
        value="\n".join(current_keywords),
        height=180,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enregistrer dictionnaire", width="stretch"):
            dictionaries[theme] = [k.strip() for k in keyword_text.splitlines() if k.strip()]
            st.session_state["dictionaries"] = dictionaries
            try:
                save_dictionaries(DICTIONARY_PATH, dictionaries)
                st.success("Dictionnaire enregistre.")
            except Exception as exc:
                st.warning(f"Ecriture du fichier dictionnaire impossible ({exc}).")
    with c2:
        if st.button("Reset dictionnaires par defaut", width="stretch"):
            st.session_state["dictionaries"] = DEFAULT_DICTIONARIES.copy()
            st.session_state["theme"] = sorted(DEFAULT_DICTIONARIES.keys())[0]
            try:
                save_dictionaries(DICTIONARY_PATH, st.session_state["dictionaries"])
            except Exception:
                pass
            st.rerun()

keywords_norm = prepare_keywords(st.session_state["dictionaries"].get(theme, []))
if not keywords_norm:
    st.info("Ajoute au moins un mot-cle dans le theme selectionne.")
    st.stop()

df = df_raw.copy()
df["_date"] = parse_datetime(df, date_col, time_col)
df = df[df["_date"].notna()].copy()
if df.empty:
    st.error("Aucune date valide apres parsing.")
    st.stop()

if channel_col:
    if normalize_channels:
        df["_channel"] = df[channel_col].map(normalize_channel)
    else:
        df["_channel"] = df[channel_col].fillna("").astype(str).str.strip()
else:
    df["_channel"] = "(sans chaine)"

min_date = df["_date"].min().to_pydatetime()
max_date = df["_date"].max().to_pydatetime()
date_start, date_end = st.sidebar.slider(
    "Periode",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

df_period = df[(df["_date"] >= pd.Timestamp(date_start)) & (df["_date"] <= pd.Timestamp(date_end))].copy()
if df_period.empty:
    st.warning("Aucune donnee dans la periode selectionnee.")
    st.stop()

df_period = add_tagging_columns(df_period, title_col=title_col, keywords_norm=keywords_norm)

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

stats = aggregate_by_period(df_filtered, frequency=frequency)
desc = build_descriptive_table(stats, df_filtered)
top_channels = build_top_channels(df_period)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Titres analyses", f"{len(df_filtered):,}")
k2.metric("Titres matches", f"{int(df_filtered['is_match'].sum()):,}")
k3.metric("Occurrences totales", f"{int(df_filtered['occurrences'].sum()):,}")
k4.metric("Frequence moyenne", f"{stats['frequency'].mean():.3f}")

st.subheader("Frequence du theme")
fig_freq = px.line(
    stats,
    x="period_start",
    y="frequency",
    markers=True,
    render_mode="svg",
    title=f"Frequence ({frequency.lower()}) - theme '{theme}'",
    labels={"period_start": "Date", "frequency": "Part de titres matches"},
)
apply_time_axis_controls(fig_freq)
fig_freq.update_layout(height=420)
st.plotly_chart(fig_freq, width="stretch")

st.subheader("Volumes")
fig_vol = px.line(
    stats,
    x="period_start",
    y=["matched_titles", "occurrences"],
    markers=True,
    render_mode="svg",
    title=f"Volumes ({frequency.lower()}) - theme '{theme}'",
    labels={"period_start": "Date", "value": "Volume", "variable": "Serie"},
)
apply_time_axis_controls(fig_vol)
fig_vol.update_layout(height=420)
st.plotly_chart(fig_vol, width="stretch")

st.subheader("Statistiques descriptives")
st.dataframe(desc, width="stretch")

if channel_col:
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
preview_cols = [c for c in ["_date", "_channel", title_col, "occurrences", "source_file"] if c in df_filtered.columns]
st.dataframe(
    df_filtered[df_filtered["is_match"] == 1]
    .sort_values(["occurrences", "_date"], ascending=[False, False])
    .head(300)[preview_cols],
    width="stretch",
)
