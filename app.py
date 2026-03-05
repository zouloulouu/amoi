import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import streamlit as st


st.set_page_config(page_title="INA - Dictionnaire (simple)", layout="wide")

DATA_DIR = "data"
DICTIONARY_PATH = "dictionaries.json"

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

TITLE_CANDIDATES = [
    "titre_propre",
    "titre",
    "title",
    "intitule",
    "libelle",
    "titre_programme",
    "titre_collection",
]
DATE_CANDIDATES = ["date_diffusion", "date", "date_notice", "date_publication"]
TIME_CANDIDATES = ["heure_diffusion", "heure", "time", "horaire"]
CHANNEL_CANDIDATES = ["chaine", "channel"]

DIRECTION_UP = 1
DIRECTION_DOWN = -1
DIRECTION_FLAT = 0


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
    except Exception:
        return clone_dictionaries(DEFAULT_DICTIONARIES)


def save_dictionaries(path: str, dictionaries: Dict[str, Dict[str, List[str]]]) -> None:
    normalized = normalize_dictionaries_payload(dictionaries)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


@st.cache_data(show_spinner=False)
def load_parquets_from_folder(folder: str) -> pd.DataFrame:
    if not os.path.isdir(folder):
        return pd.DataFrame()
    files = [f for f in os.listdir(folder) if str(f).lower().endswith(".parquet")]
    if not files:
        return pd.DataFrame()

    frames = []
    for file_name in sorted(files):
        path = os.path.join(folder, file_name)
        try:
            schema_names = pq.ParquetFile(path).schema.names
            source_by_canon = {canon_colname(c): c for c in schema_names}
            source_title = next((source_by_canon[c] for c in TITLE_CANDIDATES if c in source_by_canon), None)
            source_date = next((source_by_canon[c] for c in DATE_CANDIDATES if c in source_by_canon), None)
            source_time = next((source_by_canon[c] for c in TIME_CANDIDATES if c in source_by_canon), None)
            source_channel = next((source_by_canon[c] for c in CHANNEL_CANDIDATES if c in source_by_canon), None)

            if not source_title or not source_date:
                st.warning(f"Colonnes minimales absentes dans {file_name} (titre/date).")
                continue

            selected_columns = [source_title, source_date]
            if source_time:
                selected_columns.append(source_time)
            if source_channel:
                selected_columns.append(source_channel)

            raw = pd.read_parquet(path, columns=selected_columns)
            normalized = pd.DataFrame(
                {
                    "title": raw[source_title],
                    "date": raw[source_date],
                    "source_file": file_name,
                }
            )
            if source_time:
                normalized["time"] = raw[source_time]
            if source_channel:
                normalized["channel"] = raw[source_channel]

            frames.append(normalized)
        except Exception as exc:
            st.warning(f"Lecture impossible: {file_name} ({exc})")

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


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


def add_tagging_columns_hier(
    df: pd.DataFrame,
    title_col: str,
    concept_norm: List[str],
    context_norm: List[str],
    up_norm: List[str],
    down_norm: List[str],
) -> pd.DataFrame:
    titles_norm = df[title_col].fillna("").astype(str).map(normalize_text)

    df["occ_concept"] = titles_norm.map(lambda x: count_occurrences(x, concept_norm))
    df["occ_context"] = titles_norm.map(lambda x: count_occurrences(x, context_norm))
    df["occ_up"] = titles_norm.map(lambda x: count_occurrences(x, up_norm))
    df["occ_down"] = titles_norm.map(lambda x: count_occurrences(x, down_norm))

    df["is_concept"] = (df["occ_concept"] > 0).astype("int8")
    df["is_context"] = (df["occ_context"] > 0).astype("int8")
    df["is_match_broad"] = df["is_concept"].astype("int8")

    if context_norm:
        strict_mask = (df["is_concept"] == 1) & (df["is_context"] == 1)
        df["is_match_strict"] = strict_mask.astype("int8")
    else:
        df["is_match_strict"] = df["is_match_broad"].astype("int8")

    direction = pd.Series(DIRECTION_FLAT, index=df.index, dtype="int8")
    strict_rows = df["is_match_strict"] == 1
    direction.loc[strict_rows & (df["occ_up"] > df["occ_down"])] = DIRECTION_UP
    direction.loc[strict_rows & (df["occ_down"] > df["occ_up"])] = DIRECTION_DOWN
    df["direction"] = direction

    return df


def periodize(series: pd.Series, frequency: str) -> pd.Series:
    if frequency == "Trimestrielle":
        return series.dt.to_period("Q").dt.to_timestamp()
    if frequency == "Annuelle":
        return series.dt.to_period("Y").dt.to_timestamp()
    return series.dt.to_period("M").dt.to_timestamp()


def aggregate_by_period(df: pd.DataFrame, frequency: str, mode: str = "strict") -> pd.DataFrame:
    match_col = "is_match_strict" if mode == "strict" else "is_match_broad"
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
            matched_titles=("_match_mode", "sum"),
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


def build_descriptive_table(stats: pd.DataFrame, df_tagged: pd.DataFrame, mode: str) -> pd.DataFrame:
    if stats.empty or df_tagged.empty:
        return pd.DataFrame(columns=["indicateur", "valeur"])

    match_col = "is_match_strict" if mode == "strict" else "is_match_broad"
    freq_col = "strict_frequency" if mode == "strict" else "frequency"

    total_titles = int(len(df_tagged))
    matched_titles = int(df_tagged[match_col].sum())
    strict_titles = int(df_tagged["is_match_strict"].sum())
    occ_concept_total = int(df_tagged.loc[df_tagged[match_col] == 1, "occ_concept"].sum())
    up_titles = int((df_tagged["direction"] == DIRECTION_UP).sum())
    down_titles = int((df_tagged["direction"] == DIRECTION_DOWN).sum())
    net_signal = up_titles - down_titles

    return pd.DataFrame(
        [
            {"indicateur": "Titres analyses", "valeur": total_titles},
            {"indicateur": "Titres matches (mode)", "valeur": matched_titles},
            {"indicateur": "Occurrences concept totales", "valeur": occ_concept_total},
            {"indicateur": "Titres stricts", "valeur": strict_titles},
            {"indicateur": "Up", "valeur": up_titles},
            {"indicateur": "Down", "valeur": down_titles},
            {"indicateur": "Signal net", "valeur": net_signal},
            {"indicateur": "Frequence moyenne", "valeur": float(stats[freq_col].mean())},
            {"indicateur": "Frequence mediane", "valeur": float(stats[freq_col].median())},
            {"indicateur": "Frequence max", "valeur": float(stats[freq_col].max())},
            {"indicateur": "Volume moyen (titres matches)", "valeur": float(stats["matched_titles"].mean())},
            {"indicateur": "Nb periodes", "valeur": int(len(stats))},
        ]
    )


def build_top_channels(df_tagged: pd.DataFrame, mode: str) -> pd.DataFrame:
    if "_channel" not in df_tagged.columns:
        return pd.DataFrame()

    match_col = "is_match_strict" if mode == "strict" else "is_match_broad"
    work = df_tagged.copy()
    work["_match_mode"] = work[match_col].astype(int)
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
    st.write("Chargement automatique de tous les fichiers `.parquet` du dossier local `data/`.")

df_raw = load_parquets_from_folder(DATA_DIR)
if df_raw.empty:
    st.warning("Aucune donnee chargee (aucun fichier `.parquet` lisible dans `data/`).")
    st.stop()

if "dictionaries" not in st.session_state:
    st.session_state["dictionaries"] = load_dictionaries(DICTIONARY_PATH)

dictionaries = normalize_dictionaries_payload(st.session_state["dictionaries"])
if not dictionaries:
    dictionaries = clone_dictionaries(DEFAULT_DICTIONARIES)
st.session_state["dictionaries"] = dictionaries

columns = list(df_raw.columns)
title_col = "title" if "title" in columns else None
date_col = "date" if "date" in columns else None
time_col = "time" if "time" in columns else None
channel_col = "channel" if "channel" in columns else None

if not title_col or not date_col:
    st.error("Colonnes minimales introuvables: il faut au moins une colonne titre et une colonne date.")
    st.stop()

st.sidebar.header("Parametres")
frequency = st.sidebar.selectbox("Frequence", ["Mensuelle", "Trimestrielle", "Annuelle"], index=0)
strict_mode_enabled = st.sidebar.toggle("Mode strict (concept+contexte)", value=True)
mode = "strict" if strict_mode_enabled else "broad"
normalize_channels = True

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
        "2. Renseigne les 4 dictionnaires (1 ligne = 1 mot-cle), puis clique `Enregistrer dictionnaire`."
    )

    new_theme = st.text_input("Nouveau theme", value="")
    if st.button("Ajouter un theme"):
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
                st.warning(f"Ecriture du fichier dictionnaire impossible ({exc}).")
            st.rerun()

    current_theme_dict = normalize_theme_dictionary(dictionaries.get(theme, empty_theme_dictionary()))

    concept_text = st.text_area(
        f"Concept ({theme})",
        value="\n".join(current_theme_dict["concept"]),
        height=140,
    )
    context_text = st.text_area(
        f"Contexte ({theme})",
        value="\n".join(current_theme_dict["context"]),
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

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enregistrer dictionnaire", width="stretch"):
            dictionaries[theme] = {
                "concept": [k.strip() for k in concept_text.splitlines() if k.strip()],
                "context": [k.strip() for k in context_text.splitlines() if k.strip()],
                "up": [k.strip() for k in up_text.splitlines() if k.strip()],
                "down": [k.strip() for k in down_text.splitlines() if k.strip()],
            }
            st.session_state["dictionaries"] = dictionaries
            try:
                save_dictionaries(DICTIONARY_PATH, dictionaries)
                st.success("Dictionnaire enregistre.")
            except Exception as exc:
                st.warning(f"Ecriture du fichier dictionnaire impossible ({exc}).")
    with c2:
        if st.button("Reset dictionnaires par defaut", width="stretch"):
            st.session_state["dictionaries"] = clone_dictionaries(DEFAULT_DICTIONARIES)
            st.session_state["theme"] = sorted(DEFAULT_DICTIONARIES.keys())[0]
            try:
                save_dictionaries(DICTIONARY_PATH, st.session_state["dictionaries"])
            except Exception:
                pass
            st.rerun()

theme_dict = normalize_theme_dictionary(st.session_state["dictionaries"].get(theme, empty_theme_dictionary()))
concept_norm = prepare_keywords(theme_dict["concept"])
context_norm = prepare_keywords(theme_dict["context"])
up_norm = prepare_keywords(theme_dict["up"])
down_norm = prepare_keywords(theme_dict["down"])

if not concept_norm:
    st.info("Ajoute au moins un mot-cle dans `Concept` pour le theme selectionne.")
    st.stop()

df = df_raw
df["_date"] = parse_datetime(df, date_col, time_col)
df = df[df["_date"].notna()].copy()
if df.empty:
    st.error("Aucune date valide apres parsing.")
    st.stop()

min_date_ts = df["_date"].min()
max_date_ts = df["_date"].max()
default_start_ts = max_date_ts - pd.DateOffset(years=2)
if default_start_ts < min_date_ts:
    default_start_ts = min_date_ts
date_start, date_end = st.sidebar.slider(
    "Periode",
    min_value=min_date_ts.to_pydatetime(),
    max_value=max_date_ts.to_pydatetime(),
    value=(default_start_ts.to_pydatetime(), max_date_ts.to_pydatetime()),
)

df_period = df[(df["_date"] >= pd.Timestamp(date_start)) & (df["_date"] <= pd.Timestamp(date_end))].copy()
if df_period.empty:
    st.warning("Aucune donnee dans la periode selectionnee.")
    st.stop()

if channel_col:
    if normalize_channels:
        df_period["_channel"] = df_period[channel_col].map(normalize_channel)
    else:
        df_period["_channel"] = df_period[channel_col].fillna("").astype(str).str.strip()
else:
    df_period["_channel"] = "(sans chaine)"

df_period = add_tagging_columns_hier(
    df_period,
    title_col=title_col,
    concept_norm=concept_norm,
    context_norm=context_norm,
    up_norm=up_norm,
    down_norm=down_norm,
)

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

stats = aggregate_by_period(df_filtered, frequency=frequency, mode=mode)
desc = build_descriptive_table(stats, df_filtered, mode=mode)
top_channels = build_top_channels(df_period, mode=mode)

match_col = "is_match_strict" if mode == "strict" else "is_match_broad"
freq_col = "strict_frequency" if mode == "strict" else "frequency"
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
    title=f"Frequence ({frequency.lower()}) - theme '{theme}' ({mode})",
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
    title=f"Volumes ({frequency.lower()}) - theme '{theme}' ({mode})",
    labels={"period_start": "Date", "value": "Volume", "variable": "Serie"},
)
apply_time_axis_controls(fig_vol)
fig_vol.update_layout(height=420)
st.plotly_chart(fig_vol, width="stretch")

if up_norm or down_norm:
    st.subheader("Sens du signal")
    fig_signal = px.bar(
        stats,
        x="period_start",
        y=["up_titles", "down_titles", "net_signal"],
        barmode="group",
        title=f"Sens du signal ({frequency.lower()}) - theme '{theme}'",
        labels={"period_start": "Date", "value": "Titres", "variable": "Indicateur"},
    )
    apply_time_axis_controls(fig_signal)
    fig_signal.update_layout(height=420)
    st.plotly_chart(fig_signal, width="stretch")

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
            title=f"Top chaines par titres matches ({mode})",
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
        "occ_context",
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
