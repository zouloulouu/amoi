import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR / "backend"
if BACKEND_DIR.exists():
    import sys

    backend_path = str(BACKEND_DIR)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

import pandas as pd
import plotly.express as px
import streamlit as st

from ina_core import (
    DIRECTION_AMBIGUOUS,
    DIRECTION_DOWN,
    DIRECTION_FLAT,
    DIRECTION_UP,
    TaggingCache,
    aggregate_by_period,
    build_channel_stats,
    build_decade_distribution,
    build_descriptive_table,
    build_top_channels,
    clone_dictionaries,
    empty_theme_dictionary,
    normalize_theme_dictionary,
    prepare_keywords,
    tag_dataframe,
)
from ina_core.store import (
    DEFAULT_HF_PARQUET_FILES,
    CompositeDictRepository,
    DataStore,
    HuggingFaceRepository,
    LocalJsonRepository,
    Settings,
    ThemeAlreadyExists,
    ThemeNotFound,
)


st.set_page_config(page_title="INA — Analyse thématique", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

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


def get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN", None)
    except Exception as exc:
        LOGGER.info("Secret HF_TOKEN indisponible localement: %s", exc)
        return None


def stop_with_error_log(user_message: str, context: str, exc: Exception) -> None:
    LOGGER.exception("%s: %s", context, exc)
    st.error(f"{user_message} Consulte le log `{LOG_PATH.as_posix()}`.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# I/O LAYER — Settings, DataStore, DictRepository (cf. ina_core.store)
# ──────────────────────────────────────────────────────────────────────────────

SETTINGS = Settings(
    hf_repo_id="zouloulouu/data_ina_clean",
    hf_parquet_files=DEFAULT_HF_PARQUET_FILES,
    hf_token=get_hf_token(),
    project_root=Path.cwd(),
)

DATA_STORE = DataStore(SETTINGS)

DICT_REPO = CompositeDictRepository(
    primary=LocalJsonRepository(SETTINGS.dictionary_path),
    mirror=(
        HuggingFaceRepository(SETTINGS.hf_repo_id, SETTINGS.hf_token)
        if SETTINGS.hf_token else None
    ),
)


@st.cache_resource(show_spinner="Chargement des données (snapshot local sinon HuggingFace)...")
def load_corpus() -> Tuple[pd.DataFrame, List[str], str]:
    """Wrap DataStore.load with Streamlit cache. Returns (df, issues, source_label)."""
    df, issues = DATA_STORE.load(prefer="auto")
    # Détecte la source à partir des messages d'issues pour l'afficher dans l'UI.
    source = "local"
    if any("bascule sur HuggingFace" in i for i in issues):
        source = "hf"
    elif any("Snapshot local introuvable" in i for i in issues):
        source = "hf"  # local was missing from the start
    return df, issues, source


@st.cache_resource
def get_tagging_cache() -> TaggingCache:
    """Process-wide tagging cache, shared across all user sessions.

    Without this, each session_state would hold its own tagged DataFrame
    copy → mémoire ~150 Mo × N_users × N_themes_consultes.
    """
    return TaggingCache(maxsize=4)


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

data_signature = SETTINGS.hf_parquet_files
try:
    df_base, load_issues, load_source = load_corpus()
except Exception as _hf_exc:
    st.error(f"Erreur chargement corpus : {_hf_exc}")
    st.stop()

_SOURCE_LABEL = {
    "local": f"snapshot local (`{SETTINGS.clean_dir.as_posix()}/CURRENT`)",
    "hf": f"HuggingFace `{SETTINGS.hf_repo_id}`",
}.get(load_source, load_source)

with st.expander("Données chargées", expanded=False):
    st.caption(f"Source : {_SOURCE_LABEL}")
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
    st.warning("Aucune donnée chargée (ni snapshot local, ni HuggingFace).")
    if load_issues:
        with st.expander("Détail du chargement", expanded=True):
            for issue in load_issues:
                st.warning(issue)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# DICTIONNAIRES — relus à chaque rerun pour voir les modifs des autres users.
# Le coût est négligeable (~1 ms pour ~13 ko de JSON).
# ──────────────────────────────────────────────────────────────────────────────

_loaded = DICT_REPO.load()
dictionaries = _loaded if _loaded else clone_dictionaries(DEFAULT_DICTIONARIES)
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

        # Distribution par décennie — barres groupées (chaque décennie somme à 100% sur ses chaînes)
        if not decade_df.empty:
            st.markdown("**Part de chaque chaîne dans chaque décennie** (chaque décennie somme à 100 %)")
            st.caption(
                "Lecture : pour une décennie donnée, comment se répartissent les observations "
                "entre chaînes."
            )
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
                    "pct": "Part (%) dans la décennie",
                    "_channel": "Chaîne",
                },
                height=380,
            )
            fig_dec.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(ticksuffix=" %"),
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
# BARRE LATÉRALE — FORM UNIQUE (filtres bufferisés jusqu'au clic Appliquer)
# Tous les widgets de filtrage sont dans un st.form pour éviter les reruns en
# cascade quand l'utilisateur enchaîne plusieurs changements.
# ──────────────────────────────────────────────────────────────────────────────

themes = sorted(dictionaries.keys())
if not themes:
    st.session_state["dictionaries"] = clone_dictionaries(DEFAULT_DICTIONARIES)
    dictionaries = st.session_state["dictionaries"]
    themes = sorted(dictionaries.keys())

if "theme" not in st.session_state or st.session_state["theme"] not in themes:
    st.session_state["theme"] = themes[0]

# Métadonnées du corpus (df_base — indépendant du thème actif)
_min_date = df_base["_date"].min()
_max_date = df_base["_date"].max()
_default_start = _max_date - pd.DateOffset(years=2)
if _default_start < _min_date:
    _default_start = _min_date

_all_channels_full = sorted(
    c for c in df_base["_channel"].dropna().unique().tolist() if str(c).strip()
)
if not _all_channels_full:
    _all_channels_full = ["(sans chaîne)"]

_FREQUENCIES = ["Mensuelle", "Trimestrielle", "Annuelle"]
_COUNT_MODES = ["Binaire (présence / absence)", "Intensité (occurrences brutes)"]

# État appliqué — initialisé une fois, modifié uniquement sur clic Appliquer
if "filters" not in st.session_state:
    st.session_state["filters"] = {
        "frequency": _FREQUENCIES[0],
        "count_mode": _COUNT_MODES[0],
        "date_start": _default_start.date(),
        "date_end": _max_date.date(),
        "channels": list(_all_channels_full),
    }

# Garde-fou : si une chaîne appliquée a disparu du corpus, on filtre.
_applied_channels = [c for c in st.session_state["filters"]["channels"] if c in _all_channels_full]
if not _applied_channels:
    _applied_channels = list(_all_channels_full)
st.session_state["filters"]["channels"] = _applied_channels

st.sidebar.header("Paramètres")

with st.sidebar.form("filters_form", border=False):
    _pending_theme = st.selectbox(
        "Thème actif",
        options=themes,
        index=themes.index(st.session_state["theme"]),
        help="Le thème dont les concepts sont recherchés dans les titres. "
             "Créez ou modifiez des thèmes dans la section « Thèmes et dictionnaires ».",
    )
    _pending_freq = st.selectbox(
        "Fréquence",
        _FREQUENCIES,
        index=_FREQUENCIES.index(st.session_state["filters"]["frequency"]),
        help="Période d'agrégation des résultats.",
    )
    _pending_count_mode = st.radio(
        "Mode de comptage",
        options=_COUNT_MODES,
        index=_COUNT_MODES.index(st.session_state["filters"]["count_mode"]),
        help=(
            "**Binaire** : chaque titre vaut 1 s'il contient le concept, 0 sinon. "
            "C'est l'approche recommandée pour construire un signal de saillance médiatique. \n\n"
            "**Intensité** : somme des occurrences brutes du concept dans les titres. "
            "Utile pour vérifier la robustesse, mais attention au surcompte."
        ),
    )

    st.divider()
    st.markdown("**Période**")
    _dcol1, _dcol2 = st.columns(2)
    _pending_date_start = _dcol1.date_input(
        "Du",
        value=st.session_state["filters"]["date_start"],
        min_value=_min_date.date(),
        max_value=_max_date.date(),
        format="DD/MM/YYYY",
    )
    _pending_date_end = _dcol2.date_input(
        "Au",
        value=st.session_state["filters"]["date_end"],
        min_value=_min_date.date(),
        max_value=_max_date.date(),
        format="DD/MM/YYYY",
    )

    st.markdown("**Chaînes**")
    _pending_channels = st.multiselect(
        "Filtre chaînes",
        options=_all_channels_full,
        default=st.session_state["filters"]["channels"],
        label_visibility="collapsed",
    )

    _apply = st.form_submit_button(
        "Appliquer", type="primary", use_container_width=True
    )
    _full_range = st.form_submit_button(
        "Plage complète + toutes chaînes", use_container_width=True,
        help="Réinitialise la période et la sélection chaînes au maximum.",
    )

if _full_range:
    st.session_state["theme"] = _pending_theme
    st.session_state["filters"] = {
        "frequency": _pending_freq,
        "count_mode": _pending_count_mode,
        "date_start": _min_date.date(),
        "date_end": _max_date.date(),
        "channels": list(_all_channels_full),
    }
    st.rerun()

if _apply:
    if _pending_date_start > _pending_date_end:
        st.sidebar.error("La date de début doit être avant la date de fin.")
        st.stop()
    if not _pending_channels:
        st.sidebar.error("Sélectionnez au moins une chaîne.")
        st.stop()
    st.session_state["theme"] = _pending_theme
    st.session_state["filters"] = {
        "frequency": _pending_freq,
        "count_mode": _pending_count_mode,
        "date_start": _pending_date_start,
        "date_end": _pending_date_end,
        "channels": _pending_channels,
    }

# Lecture de l'état appliqué pour le reste du script
theme = st.session_state["theme"]
_filters = st.session_state["filters"]
frequency = _filters["frequency"]
count_mode = _filters["count_mode"]
binary_mode = count_mode.startswith("Binaire")
date_start = pd.Timestamp(_filters["date_start"])
date_end = pd.Timestamp(_filters["date_end"])
selected_channels = _filters["channels"]

with st.sidebar.expander("Diagnostics", expanded=False):
    st.caption(f"Log erreurs : `{LOG_PATH.as_posix()}`")

    if st.checkbox("Afficher l'état des caches", value=False):
        tagging_cache_size = len(get_tagging_cache())
        agg_cache_info = st.session_state.get("_agg_cache", {})
        ch_stats_cached = "_ch_stats_cache" in st.session_state
        st.caption(
            f"Cache tagging (process-wide) : {tagging_cache_size} entrée(s)  \n"
            f"Cache agrégation (par session) : {len(agg_cache_info)} entrée(s)  \n"
            f"Cache stats chaînes : {'oui' if ch_stats_cached else 'non'}"
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
# GESTION DES THÈMES ET DICTIONNAIRES
# ──────────────────────────────────────────────────────────────────────────────

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
            try:
                DICT_REPO.create_theme(nt, empty_theme_dictionary())
                get_tagging_cache().invalidate(nt)
                st.session_state["theme"] = nt
                st.session_state["dict_flash"] = (
                    f"Thème « {nt} » créé. Complétez maintenant son dictionnaire ci-dessous, "
                    "puis cliquez sur « Enregistrer »."
                )
                st.session_state["dict_expander_open"] = True
                st.rerun()
            except ThemeAlreadyExists:
                st.warning(
                    f"Le thème « {nt} » a été créé entre-temps par un autre utilisateur. "
                    "Rafraîchis la page pour le voir."
                )
            except Exception as exc:
                LOGGER.exception("Écriture dictionnaire impossible (création) : %s", exc)
                st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")

    st.divider()

    # ── Modifier le thème actif ─────────────────────────────────────────────
    theme_edit = theme  # toujours synchronisé avec le selectbox sidebar
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
            try:
                DICT_REPO.rename_theme(theme_edit, new_name)
                _cache = get_tagging_cache()
                _cache.invalidate(theme_edit)
                _cache.invalidate(new_name)
                st.session_state["theme"] = new_name
                st.session_state["dict_flash"] = f"Thème renommé : « {theme_edit} » → « {new_name} »."
                st.session_state["dict_expander_open"] = True
                st.rerun()
            except ThemeNotFound:
                st.warning(
                    f"Le thème « {theme_edit} » a été supprimé entre-temps. Rafraîchis la page."
                )
            except ThemeAlreadyExists:
                st.warning(f"Un thème « {new_name} » a été créé entre-temps par un autre utilisateur.")
            except Exception as exc:
                LOGGER.exception("Écriture dictionnaire impossible (renommage) : %s", exc)
                st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")

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
        new_theme = {
            "concept": new_concept,
            "context": current_theme_dict.get("context", []),
            "up": new_up,
            "down": new_down,
        }
        try:
            DICT_REPO.update_theme(theme_edit, new_theme)
            get_tagging_cache().invalidate(theme_edit)
            st.success(
                f"Dictionnaire « {theme_edit} » enregistré : "
                f"{len(new_concept)} concept(s) · {len(new_up)} terme(s) UP · {len(new_down)} terme(s) DOWN."
            )
        except ThemeNotFound:
            st.warning(
                f"Le thème « {theme_edit} » a été supprimé entre-temps. Rafraîchis la page."
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
                try:
                    new_state = DICT_REPO.delete_theme(theme_edit)
                    get_tagging_cache().invalidate(theme_edit)
                    remaining = sorted(new_state.keys())
                    st.session_state["theme"] = remaining[0] if remaining else ""
                    st.session_state["dict_flash"] = f"Thème « {theme_edit} » supprimé."
                    st.session_state["dict_expander_open"] = True
                    st.rerun()
                except ThemeNotFound:
                    st.warning(f"Le thème « {theme_edit} » a déjà été supprimé.")
                    st.rerun()
                except Exception as exc:
                    LOGGER.exception("Écriture dictionnaire impossible (suppression) : %s", exc)
                    st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")

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
                DICT_REPO.save(st.session_state["dictionaries"])
                get_tagging_cache().invalidate()  # tout vider
            except Exception as exc:
                LOGGER.exception("Écriture dictionnaire impossible (reset) : %s", exc)
                st.warning(f"Impossible d'écrire le fichier dictionnaire ({exc}).")
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# MARQUAGE (TAGGING)
# ──────────────────────────────────────────────────────────────────────────────

theme_dict = normalize_theme_dictionary(
    st.session_state["dictionaries"].get(theme, empty_theme_dictionary())
)

# Préparation des keywords normalisés (rapide, pas de cache nécessaire)
concept_norm = prepare_keywords(theme_dict["concept"])
up_norm = prepare_keywords(theme_dict["up"])
down_norm = prepare_keywords(theme_dict["down"])

if not concept_norm:
    st.info(
        f"Le thème « {theme} » n'a pas encore de mots-clés concept. "
        "Ouvrez la section **Thèmes et dictionnaires** ci-dessus et complétez l'étape 2."
    )
    st.stop()

# Tagging — cache PROCESS-WIDE partagé entre toutes les sessions utilisateur
TAGGING_CACHE = get_tagging_cache()
try:
    df_tagged = TAGGING_CACHE.get_or_compute(
        theme=theme,
        concept=tuple(concept_norm),
        up=tuple(up_norm),
        down=tuple(down_norm),
        compute_fn=lambda: tag_dataframe(
            df_base,
            title_col=title_col,
            concept_norm=concept_norm,
            up_norm=up_norm,
            down_norm=down_norm,
            title_norm_col="_title_norm",
        ),
    )
except Exception as exc:
    stop_with_error_log("Erreur pendant le marquage des titres.", "tag_dataframe", exc)

# Clé stable utilisée par le cache d'agrégation aval
tag_cache_key = json.dumps(
    {"theme": theme, "concept": list(concept_norm), "up": list(up_norm), "down": list(down_norm)},
    sort_keys=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# APPLICATION DES FILTRES (période + chaînes — déjà bufferisés via le form)
# ──────────────────────────────────────────────────────────────────────────────

# Masque de période
_period_mask = (df_tagged["_date"] >= date_start) & (df_tagged["_date"] <= date_end)
if not _period_mask.any():
    st.warning("Aucune donnée dans la période sélectionnée.")
    st.stop()

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
        top_channels = build_top_channels(df_filtered)
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
        "Calculé sur la période **et** les chaînes sélectionnées."
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

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Exporter les données")


def format_period_date(ts: pd.Timestamp, freq: str) -> str:
    if freq == "Trimestrielle":
        return f"{ts.year}-Q{ts.quarter}"
    if freq == "Annuelle":
        return str(ts.year)
    return ts.strftime("%Y-%m")


def build_export_df(stats: pd.DataFrame, freq: str) -> pd.DataFrame:
    export = stats[["period_start", "total_titles", "matched_titles",
                     "frequency", "up_titles", "down_titles", "net_signal"]].copy()
    export["date"] = export["period_start"].apply(lambda t: format_period_date(t, freq))
    export = export.rename(columns={
        "total_titles": "total_titres",
        "matched_titles": "volume",
        "frequency": "frequence",
        "up_titles": "signal_up",
        "down_titles": "signal_down",
        "net_signal": "signal_net",
    })
    export["frequence"] = export["frequence"].round(4)
    return export[["date", "total_titres", "volume", "frequence",
                    "signal_up", "signal_down", "signal_net"]]


_export_df = build_export_df(stats, frequency)
_fname = (
    f"{theme}_{frequency.lower()}"
    f"_{date_start.strftime('%Y%m%d')}"
    f"_{date_end.strftime('%Y%m%d')}.csv"
)

st.caption(
    f"{len(_export_df)} période(s) · thème **{theme}** · "
    f"{frequency.lower()} · "
    f"{date_start.strftime('%d/%m/%Y')} → {date_end.strftime('%d/%m/%Y')}"
)
st.download_button(
    label="Télécharger le CSV",
    data=_export_df.to_csv(index=False, sep=";", decimal=",").encode("utf-8"),
    file_name=_fname,
    mime="text/csv",
)
