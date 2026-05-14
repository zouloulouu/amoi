# Audit du projet data_ina

## 1. Résumé exécutif

Le projet `data_ina` est une application **Streamlit monolithique** (`app.py`, ~1460 lignes) qui charge depuis Hugging Face un corpus de titres de JT (7 chaînes, format Parquet), applique un tagging par dictionnaires thématiques (concept / up / down), agrège mensuellement / trimestriellement / annuellement et expose KPIs, graphiques Plotly, top chaînes, aperçu et export CSV. À côté vit un pipeline ML scripts/ (clean → gold → train → predict → evaluate → compare) pour deux thèmes (chômage, inflation).

Streamlit pose problème parce que **toute la logique métier est mélangée à la couche UI**, le state utilisateur sert de cache implicite, chaque rerun risque de re-tagger ou re-agréger un DataFrame entier copié, et la sauvegarde des dictionnaires écrit en parallèle le fichier local + Hugging Face dans le thread du rerun, sans verrou multi-utilisateur.

L'architecture cible recommandée est **simple** : extraire un package `core/` (fonctions pures déjà presque prêtes), exposer une API FastAPI minimale (lecture seule du corpus + CRUD dictionnaires + analyse), garder Parquet+HF en source, ajouter un cache mémoire process avec invalidation explicite, et un frontend Vite/React léger. **Pas de PostgreSQL, pas de Next.js**, pas de Redis pour la v1.

Prochaines actions : (1) extraire `core/` sans toucher à `app.py`, (2) écrire des tests sur les 4 fonctions critiques (normalisation, tagging, agrégation, normalisation dico), (3) FastAPI minimal en parallèle de Streamlit, (4) basculer le frontend une route à la fois.

---

## 2. Cartographie du projet actuel

```
data_ina/
├── app.py                       ← MONOLITHE Streamlit (1460 lignes)
├── analyze_parquets.py          ← profiling local CLI → reports/parquet_profile*.json
├── merge_for_econometrics.py    ← merge INA + INSEE chômage trimestriel
├── dictionaries.json            ← 11 thèmes (chomage, emploi, inflation, croissance,
│                                   pouvoir_achat, dette_deficit, fiscalite, salaires,
│                                   immobilier, energie, taux_interet)
├── requirements.txt             ← streamlit / pandas / plotly / pyarrow / huggingface-hub
├── scripts/                     ← pipeline data + ML (Python + R)
│   ├── clean_data.py            ← raw → clean (versionné v<UTC>)
│   ├── validate_clean.py        ← schéma + qualité du snapshot
│   ├── build_gold_sample[_inflation].py
│   ├── train_classifier[_inflation].py
│   ├── predict_corpus[_inflation].py
│   ├── evaluate_dictionary[_inflation].py
│   ├── compare_signals[_inflation].py
│   ├── analyze_inflation.py     ← corrélations, CCF, OLS
│   └── plot_chomage_regressions.py
├── data/
│   ├── raw/                     ← 7 parquet immutables (gitignore)
│   ├── clean/CURRENT → v20260330_224823_utc/  (versionné, sortie de clean_data.py)
│   ├── slim/                    ← copie clean (gitignore)
│   ├── gold/                    ← échantillons ML labelisés + prédictions
│   └── inflation/               ← séries fusionnées + résultats économétriques
├── docs/
│   ├── DATA_ARCHITECTURE.md     ← raw → clean documenté
│   ├── DATA_CLEANING.md
│   └── RESULTATS_PIPELINE_ML.md
├── reports/                     ← parquet_profile*.json, clean_report.json
├── logs/streamlit_app.log
├── *.csv                        ← exports app + INSEE bruts
└── visualize_results*.R / .RData  ← scripts R (hors scope migration web)
```

**Points d'entrée** : `app.py` (Streamlit), `analyze_parquets.py` (CLI), `merge_for_econometrics.py` (CLI), `scripts/*.py` (CLI).
**Secrets attendus** : `st.secrets["HF_TOKEN"]` (lecture publique fonctionne sans, écriture HF du dico en a besoin).
**À ignorer** : `__pycache__/`, `.RData`, `.Rhistory`, `data/raw/`, `data/slim/`, `logs/`, `reports/`.

---

## 3. Fonctionnalités existantes

| Fonctionnalité | Fichier | Fonctions principales | Entrée | Sortie |
|---|---|---|---|---|
| Chargement Parquet HF | [app.py:210-253](app.py#L210-L253) | `load_clean_parquets_from_hf` | URL HF + `HF_TOKEN` | `(df_base, issues)` concaténé |
| Chargement local Parquet | scripts/clean_data.py:222 | `clean_file`, `main` | `data/raw/*.parquet` | `data/clean/<v>/*.parquet` |
| Profiling qualité | [analyze_parquets.py](analyze_parquets.py) | `top_value_counts`, `date_parse_success`, `time_pattern_ratios` | parquets | `reports/parquet_profile.json` |
| Validation snapshot | [scripts/validate_clean.py:41](scripts/validate_clean.py#L41) | `validate_file` | snapshot | `validation_report.json` |
| Normalisation texte | [app.py:137-144](app.py#L137-L144) | `normalize_text` (NFD + accents) | str | str |
| Chargement dico local | [app.py:147-160](app.py#L147-L160) | `load_dictionaries` + `normalize_dictionaries_payload` | `dictionaries.json` | dict |
| Chargement dico HF | [app.py:171-181](app.py#L171-L181) | `load_dictionaries_from_hf` | URL HF | dict |
| Sauvegarde dico locale | [app.py:163-168](app.py#L163-L168) | `save_dictionaries` (write `.tmp` + `os.replace`) | dict | fichier |
| Sauvegarde dico HF | [app.py:184-202](app.py#L184-L202) | `save_dictionaries_to_hf` (HfApi.upload_file) | dict + token | commit HF |
| Tagging concept/up/down | [app.py:280-330](app.py#L280-L330) | `add_tagging_columns_hier`, `count_occurrences`, `prepare_keywords` | df + dico | df + occ_*, is_*, direction |
| Agrégation périodique | [app.py:337-372](app.py#L337-L372) | `periodize`, `aggregate_by_period` | df_tagged + freq | stats par period_start |
| Tableau descriptif | [app.py:375-400](app.py#L375-L400) | `build_descriptive_table` | stats + df_tagged | DataFrame indicateur/valeur |
| Top chaînes | [app.py:403-427](app.py#L403-L427) | `build_top_channels` | df_tagged | top 10 |
| Couverture chaînes | [app.py:434-453](app.py#L434-L453) | `build_channel_stats` | df_base | date_min/max, n_obs, share_pct |
| Distribution décennies | [app.py:456-473](app.py#L456-L473) | `build_decade_distribution` | df_base | long format |
| Export CSV | [app.py:1417-1459](app.py#L1417-L1459) | `format_period_date`, `build_export_df` | stats | bytes CSV `;`/`,` |
| Logs | [app.py:67-89](app.py#L67-L89) | `setup_logger`, RotatingFileHandler | — | `logs/streamlit_app.log` |
| Cache tagging | [app.py:985-1011](app.py#L985-L1011) | `_tagged_theme_cache` (session_state, max 2 entrées) | tag_cache_key | df_tagged |
| Cache agrégation | [app.py:1116-1143](app.py#L1116-L1143) | `_agg_cache` (session_state, max 5) | agg_cache_key | (stats, desc, top) |

---

## 4. Diagnostic technique

### Couplage Streamlit / métier (haut)
- `add_tagging_columns_hier` mute le DataFrame passé (`df["occ_concept"] = …`) — pure côté logique mais appelée avec `df_base.copy()` à chaque nouveau cache miss → **copie complète** du DataFrame source à chaque thème.
- Toute la logique de cache repose sur `st.session_state`, donc **par-utilisateur**. Une instance Streamlit avec 5 users a 5 copies indépendantes du tagging, du DataFrame de base, et des agrégations. Coût mémoire ×N.
- Les fonctions de calcul ne sont pas séparées des fonctions de rendu : `build_descriptive_table` dépend de pandas pur (OK), mais l'orchestration n'est jamais isolée → impossible à tester sans Streamlit.

### Recalculs et reruns
- Streamlit relance tout le script à chaque interaction. Les caches `session_state` mitigent, mais :
  - Le **slider de date** ne re-tagge pas (bien) mais re-calcule l'agrégation seulement si la clé change → OK.
  - Les **boutons « Toutes / Aucune »** sur les chaînes déclenchent un rerun complet et reconstruisent les `all_channels` à chaque fois ([app.py:1067-1096](app.py#L1067-L1096)).
  - Le toggle **Binaire / Intensité** ne change que l'affichage, mais déclenche le rendu Plotly complet (acceptable).
- `clone_dictionaries` fait `json.loads(json.dumps(...))` à chaque chargement — anodin mais inutile.

### Dépendances dangereuses à `st.session_state`
- `_tagged_theme_cache` stocke des **DataFrames complets**. Avec 7 fichiers concaténés et plusieurs colonnes booléennes/int8 ajoutées, c'est lourd. La limite codée en dur est **2 entrées** ([app.py:1008-1009](app.py#L1008-L1009)) → tout changement de thème invalide.
- `_kw_cache_key` mélange tuple de strings et nom de thème → OK mais fragile à un upgrade pandas/python (hash determinism).
- `_ch_sel` réinitialise silencieusement la sélection si une chaîne disparaît de la période → bug silencieux possible si la période contient peu de données.

### `@st.cache_resource` (ligne 210)
- `load_clean_parquets_from_hf` est cached **process-wide**. Bonne nouvelle : partagé entre users. Mauvaise : pas d'expiration, donc si HF reçoit un nouveau parquet, le serveur ne le verra qu'au redémarrage. Le `data_signature = HF_PARQUET_FILES` est statique → ne capture aucun changement.
- En cas d'erreur partielle (un fichier KO sur 7), `frames` part avec un sous-ensemble silencieusement, juste un warning dans `issues`. Risque d'analyses partielles.

### Filtres
- Le `_period_mask` est un masque booléen sur `df_tagged` complet → recopie d'index. Pas catastrophique mais pas idéal sur de gros DataFrames.
- `df_period = df_tagged[_period_mask]` puis `df_filtered = df_tagged[_period_mask & …]` → deux masques évalués à chaque rerun.

### Hugging Face
- `requests.get(timeout=300)` séquentiel sur 7 fichiers : si HF rame, l'app peut bloquer ~35 min cumulés au pire.
- Pas de retry, pas de fallback local (pourtant `data/clean/CURRENT` existe).
- L'écriture HF du dico est **synchrone bloquante** dans le rerun ([app.py:782-786](app.py#L782-L786)). Une lenteur HF gèle le bouton « Créer le thème ».

### Concurrence
- `save_dictionaries` utilise un `tmp + os.replace` (atomique sur POSIX, fonctionne sur NTFS récent) → **OK localement**.
- Mais **deux users qui sauvent en même temps** : last-writer-wins, un dico peut être écrasé. Pas de versioning, pas de verrou.
- HF upload n'a pas de verrou non plus → mêmes problèmes côté distant.

### Mémoire / performance
- DataFrame complet en RAM pour tous les users (cache_resource) → OK.
- Mais chaque user a sa propre copie taguée par thème (×2) en session_state → multiplie.
- `df_tagged.copy()` n'est pas fait, donc pas si grave, mais les colonnes ajoutées vivent dans le cache.

### Erreurs
- `stop_with_error_log` est correct mais brutal : `st.stop()` après chaque exception → un seul fichier HF KO et la page est inutilisable. À nuancer (ici seul l'orchestrateur appelle stop, les 7 lectures HF avalent leurs erreurs individuellement).

---

## 5. Code à conserver / refactoriser / remplacer

### A — À conserver (fonctions pures, déjà testables)

| Fichier | Fonction | Destination |
|---|---|---|
| app.py:100-130 | `clean_term_list`, `normalize_theme_dictionary`, `normalize_dictionaries_payload`, `clone_dictionaries` | `core/dictionaries.py` |
| app.py:137-144 | `normalize_text` | `core/text.py` (utilisé aussi par scripts/build_gold/clean_data — déduplication) |
| app.py:260-277 | `prepare_keywords`, `count_occurrences` | `core/tagging.py` |
| app.py:280-330 | `add_tagging_columns_hier` (pure si `df` est immuable et qu'on retourne une copie — voir B) | `core/tagging.py` |
| app.py:337-372 | `periodize`, `aggregate_by_period` | `core/aggregation.py` |
| app.py:375-400 | `build_descriptive_table` | `core/aggregation.py` |
| app.py:403-427 | `build_top_channels` | `core/aggregation.py` |
| app.py:434-473 | `build_channel_stats`, `build_decade_distribution` | `core/coverage.py` |
| app.py:1417-1439 | `format_period_date`, `build_export_df` | `core/export.py` |
| scripts/clean_data.py | tout le module | `core/cleaning.py` (réutilisable depuis l'API si besoin) |
| analyze_parquets.py | `top_value_counts`, `date_parse_success`, `time_pattern_ratios` | `core/profiling.py` |

### B — À refactoriser (mélange ou risque)

| Fichier | Bloc | Problème | Action |
|---|---|---|---|
| app.py:147-202 | `load_dictionaries`, `save_dictionaries`, `load_dictionaries_from_hf`, `save_dictionaries_to_hf` | mélange I/O local + HTTP + logger | Scinder en `core/dictionaries_repo.py` (interface `DictionaryRepository`) avec deux implémentations : `LocalJsonRepository` et `HuggingFaceRepository`. Permet de tester sans I/O. |
| app.py:210-253 | `load_clean_parquets_from_hf` | timeout 300s séquentiel, pas de retry, lié à `st.cache_resource` | Extraire `core/data_loader.py` indépendant ; ajouter retry+timeout court+fallback local `data/clean/CURRENT` ; cache géré par l'API (pas par Streamlit). |
| app.py:280-330 | `add_tagging_columns_hier` mute df | mutation in-place ambiguë | Toujours `df = df.copy()` en début et retourner le nouveau df, ou retourner uniquement les colonnes ajoutées. |
| app.py:985-1143 | caches `_tagged_theme_cache`, `_agg_cache`, `_kw_cache` | dépendent de session_state, sérialisation JSON de la clé à chaque tour | Remplacer par un cache process unique (`functools.lru_cache` ou cache custom à clé tuple) côté API, partagé entre users. |
| app.py:1018-1058 | logique de filtre date + chaînes | mélangée à la barre latérale | Côté API : params query, validation Pydantic ; côté front : composants contrôlés. |

### C — À remplacer

| Fichier | Bloc | Pourquoi | Remplacement |
|---|---|---|---|
| app.py:505-1460 (~1000 lignes UI) | toute la couche Streamlit | rerun-driven, état non isolable, multi-user fragile | Frontend Vite/React + composants spécifiques ; routes FastAPI documentées ; pas de session_state. |
| app.py:743-954 | bloc Thèmes & dictionnaires (formulaires Streamlit) | dépendance à `st.rerun()` et `st.form_submit_button` | Frontend : page `Themes` avec CRUD via API. |
| app.py:1190-1306 | tous les `st.plotly_chart` | dépendance Plotly Streamlit | Frontend : Plotly.js direct OU recharts/visx (préférer recharts pour la simplicité). L'API renvoie le JSON de série, pas la figure. |
| `_hf_write_token` lu côté Streamlit ([app.py:547](app.py#L547)) | secret en frontend | jamais en frontend | Token stocké côté backend uniquement (`.env` du serveur). |

---

## 6. Architecture cible recommandée

```
data_ina/
├── backend/
│   ├── pyproject.toml           ← package "ina_core" + extras [api]
│   ├── ina_core/                ← LOGIQUE MÉTIER PURE (testable, importable)
│   │   ├── __init__.py
│   │   ├── text.py              ← normalize_text
│   │   ├── dictionaries.py      ← normalize/clone/validate
│   │   ├── tagging.py           ← prepare_keywords, count_occurrences, tag_dataframe
│   │   ├── aggregation.py       ← periodize, aggregate_by_period, descriptive, top_channels
│   │   ├── coverage.py          ← channel_stats, decade_distribution
│   │   ├── export.py            ← build_export_df, format_period
│   │   └── profiling.py         ← analyze_parquets helpers
│   ├── ina_api/                 ← FASTAPI
│   │   ├── main.py              ← app FastAPI + middlewares
│   │   ├── config.py            ← Settings pydantic-settings (HF_TOKEN, paths…)
│   │   ├── deps.py              ← injection : data store, dict repo, cache
│   │   ├── schemas.py           ← modèles Pydantic
│   │   ├── cache.py             ← cache mémoire process avec invalidation
│   │   ├── store/
│   │   │   ├── data_store.py    ← charge/garde le df_base en mémoire ; load HF + fallback local
│   │   │   └── dict_repo.py     ← LocalJsonRepository / HuggingFaceRepository
│   │   └── routers/
│   │       ├── health.py
│   │       ├── metadata.py
│   │       ├── themes.py
│   │       ├── channels.py
│   │       ├── analysis.py
│   │       └── export.py
│   └── tests/
│       ├── fixtures/            ← petits parquets ~50 lignes
│       ├── test_text.py
│       ├── test_tagging.py
│       ├── test_aggregation.py
│       ├── test_dictionaries.py
│       └── test_api_*.py        ← TestClient FastAPI
├── frontend/
│   ├── package.json             ← Vite + React + TypeScript + recharts + tanstack-query
│   ├── src/
│   │   ├── api/                 ← client typé (fetch / openapi-typescript)
│   │   ├── pages/
│   │   │   ├── Analysis.tsx     ← équivalent du dashboard principal
│   │   │   ├── Themes.tsx       ← CRUD dictionnaires
│   │   │   └── Coverage.tsx     ← couverture chaînes
│   │   ├── components/          ← FilterBar, KpiCards, FrequencyChart, ChannelTable…
│   │   └── hooks/               ← useAnalysis, useThemes…
├── scripts/                     ← CONSERVÉS, importent ina_core
│   ├── clean_data.py
│   ├── validate_clean.py
│   ├── analyze_parquets.py
│   └── … (ML pipeline inchangé)
├── data/                        ← inchangé
├── docs/                        ← inchangé
├── app.py                       ← CONSERVÉ pendant la migration (Phase 6 le supprime)
├── dictionaries.json            ← repo "single source of truth" pendant la transition
└── README.md
```

Ce schéma garde **tout le pipeline ML existant** intact (les scripts importeront `ina_core` au lieu de redéfinir `normalize_text` 6 fois) et permet de migrer Streamlit progressivement.

---

## 7. Routes API FastAPI proposées

Toutes les routes sont **stateless** : pas de session, pas de cookie. Auth simple Bearer token (env var) si besoin de protéger l'écriture des dictionnaires.

| Méthode | Route | Rôle | Payload | Réponse | Fonctions appelées |
|---|---|---|---|---|---|
| GET | `/health` | liveness + version + état du data store | — | `{"status":"ok","data_loaded":true,"rows":N,"loaded_at":...}` | DataStore status |
| GET | `/metadata` | métadonnées du corpus | — | `{date_min,date_max,n_total,channels:[…],source:"hf://…"}` | DataStore.summary |
| GET | `/themes` | liste des thèmes + tailles | — | `[{name,n_concept,n_up,n_down}, …]` | DictRepository.list |
| GET | `/themes/{theme}` | dictionnaire complet | — | `ThemeDictionary` | DictRepository.get |
| POST | `/themes` | créer | `ThemeCreateRequest` | 201 + `ThemeDictionary` | DictRepository.create + cache.invalidate |
| PUT | `/themes/{theme}` | maj | `ThemeUpdateRequest` | `ThemeDictionary` | DictRepository.update + cache.invalidate |
| DELETE | `/themes/{theme}` | supprimer | — | 204 | DictRepository.delete + cache.invalidate |
| GET | `/channels` | chaînes + couverture | `?date_start&date_end` | `[ChannelCoverage, …]` | `build_channel_stats` |
| GET | `/coverage` | distribution décennies | — | `[{channel,decade,n,pct}, …]` | `build_decade_distribution` |
| POST | `/analysis` | analyse complète | `AnalysisRequest` | `AnalysisResponse` (kpi + series + top + ambiguous_total) | `tag_dataframe` + `aggregate_by_period` + `build_descriptive_table` + `build_top_channels` |
| GET | `/analysis/preview` | titres matchés (paginé) | params + `?limit&offset&sort` | `[PreviewTitle, …]` | filtrage + sort sur df_tagged caché |
| GET | `/export/csv` | export CSV de l'analyse | mêmes params que `/analysis` | `text/csv` (streaming) | `build_export_df` |
| GET | `/data/profile` | profiling rapide (sur le df chargé) | — | `{n_rows,n_channels,n_dates_invalid,top_channels:[…]}` | `core/profiling` |

**Points de vigilance** :
- `/analysis` doit être **idempotent** et sa réponse hashable côté client (pour cache HTTP).
- Toujours valider que `theme` existe avant d'appeler le tagging.
- Les exports CSV peuvent être gros : utiliser `StreamingResponse`.
- `/themes` PUT doit être **atomique** : verrou (asyncio.Lock) côté process pour la première version, ou verrou de fichier `.lock` à côté de `dictionaries.json` si plusieurs workers uvicorn.

---

## 8. Modèles Pydantic proposés

```python
# schemas.py — esquisse réaliste basée sur le code observé

class ThemeDictionary(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    concept: list[str] = Field(default_factory=list, max_length=500)
    context: list[str] = Field(default_factory=list, max_length=500)  # conservé pour compat
    up: list[str] = Field(default_factory=list, max_length=500)
    down: list[str] = Field(default_factory=list, max_length=500)

class ThemeCreateRequest(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_\-]+$", min_length=1, max_length=64)
    concept: list[str] = []
    up: list[str] = []
    down: list[str] = []

class ThemeUpdateRequest(BaseModel):
    concept: list[str] | None = None
    up: list[str] | None = None
    down: list[str] | None = None

class Frequency(str, Enum):
    monthly = "monthly"
    quarterly = "quarterly"
    yearly = "yearly"

class CountMode(str, Enum):
    binary = "binary"
    intensity = "intensity"

class AnalysisRequest(BaseModel):
    theme: str
    frequency: Frequency = Frequency.monthly
    count_mode: CountMode = CountMode.binary
    date_start: date | None = None
    date_end: date | None = None
    channels: list[str] | None = None  # None = toutes

    @model_validator(mode="after")
    def _check_dates(self):
        if self.date_start and self.date_end and self.date_start > self.date_end:
            raise ValueError("date_start > date_end")
        return self

class TimeSeriesPoint(BaseModel):
    period_start: date
    total_titles: int
    matched_titles: int
    occurrences_concept: int
    frequency: float
    up_titles: int
    down_titles: int
    ambiguous_titles: int
    net_signal: int

class KpiResponse(BaseModel):
    n_total: int
    n_matched: int
    occ_concept_total: int
    freq_mean: float
    net_signal_total: int
    up_total: int
    down_total: int
    ambiguous_total: int

class ChannelCoverage(BaseModel):
    channel: str
    date_min: date
    date_max: date
    n_obs: int
    share_pct: float

class TopChannel(BaseModel):
    channel: str
    total_titles: int
    matched_titles: int
    frequency: float
    up_titles: int
    down_titles: int
    ambiguous_titles: int
    net_signal: int

class PreviewTitle(BaseModel):
    date: date
    channel: str
    title: str
    occ_concept: int
    occ_up: int
    occ_down: int
    direction: Literal["UP","DOWN","AMBIGUOUS","FLAT"]
    source_file: str

class AnalysisResponse(BaseModel):
    kpi: KpiResponse
    series: list[TimeSeriesPoint]
    top_channels: list[TopChannel]
    descriptive: list[dict[str, Any]]   # indicateur/valeur — déjà construit
    request: AnalysisRequest            # echo pour traçabilité

class ExportRequest(AnalysisRequest):
    separator: Literal[",",";","\t"] = ";"
    decimal: Literal[".",","] = ","
```

Aligné sur les colonnes réellement produites par `aggregate_by_period` et `build_descriptive_table`.

---

## 9. Stratégie de cache

### Niveaux à gérer

| Donnée | Coût recalcul | Granularité d'invalidation | Solution recommandée v1 |
|---|---|---|---|
| `df_base` (parquets concaténés) | très élevé (35 Mo+ HF, ~1 min) | redémarrage backend ou endpoint `/admin/reload` | dict global au boot, `lifespan` FastAPI ; fallback local `data/clean/CURRENT` |
| Mots-clés normalisés par dico | faible | quand le dico change | `functools.lru_cache(maxsize=64)` keyée sur le tuple des termes |
| `df_tagged` par thème | élevé (groupby sur N lignes) | quand le dico du thème change | dict en mémoire `{theme_hash: df_tagged}`, invalidé sur PUT/POST/DELETE `/themes/{theme}` |
| Agrégations (stats, top, desc) | moyen | quand un filtre change | `lru_cache` sur `(theme_hash, freq, date_start, date_end, tuple(channels))` |
| Couverture chaînes (df_base) | faible | au boot | calculé une fois |

### Comparatif

| Option | Verdict v1 |
|---|---|
| **Cache mémoire Python pur (dict + lru_cache)** | retenu pour v1 — process unique, gratuit, suffisant. |
| **joblib.Memory** | bof : disque, ralentit pour des objets pandas qui se sérialisent mal en parallèle. |
| **diskcache** | utile si on veut survivre à un redémarrage, mais ajoute une dépendance pour peu de gain (HF re-télécharge en 1 min). À garder pour plus tard. |
| **Redis** | overkill v1, intéressant si plusieurs workers uvicorn. Phase 8. |
| **Pré-calculs Parquet** | bon pour les agrégations stables (théorique : pré-tagger les 11 thèmes une fois). À envisager Phase 8 si volume explose. |

**Au redémarrage** : tout est perdu. Le cold start charge HF (~1 min) ou local (instant). Acceptable. Gating : log explicite sur la durée du chargement, endpoint `/health` qui dit `data_loaded:false` tant que pas prêt.

**Limite mémoire** : avec ~7 fichiers et N lignes (à mesurer mais probablement <200k lignes total à voir le code), tagger les 11 thèmes en parallèle reste raisonnable. Mettre une **limite explicite** : max 4 `df_tagged` simultanés, LRU eviction.

**Invalidation** : centralisée. PUT/POST/DELETE sur un thème → `cache.invalidate_theme(theme)` → invalide tagging + agrégations qui en dépendent.

---

## 10. Stratégie de données

### Comparatif

| Option | Avantages | Inconvénients | Verdict |
|---|---|---|---|
| A. HF + pandas mémoire (statu quo) | aucun changement | dépendance internet ; cold start lent ; pas de fallback | tel quel : non |
| B. Télécharger HF au démarrage du backend, garder en mémoire | propre, déjà ce que fait `cache_resource` | si HF down au boot, le serveur ne démarre pas | partiel |
| **C. Local-first : parquets sur disque (`data/clean/CURRENT`), HF en source de mise à jour** | rapide, hors-ligne possible, déjà préparé par `clean_data.py` + `CURRENT` symlink | nécessite un mécanisme de sync HF → local | **recommandé v1** |
| D. PostgreSQL | requêtes SQL, multi-user clean | complexité, mapping schéma, backup, rien ne le justifie aujourd'hui | non pour la v1 — voir Phase 8 |
| E. Pré-calcul d'agrégats | gain perf si dictos figés | invalidation complexe quand on change un dico | non pour la v1 |

### Recommandation v1

**Stratégie locale-first avec sync HF optionnelle** :
1. Au boot, le backend lit `data/clean/CURRENT` → si présent, charge depuis le disque (instantané).
2. Sinon (ou si flag `--refresh`), télécharge depuis HF dans `data/clean/v<UTC>/`, met à jour `CURRENT`.
3. Endpoint admin `POST /admin/reload?source=hf|local` pour recharger sans redémarrer.
4. Les **dictionnaires** restent dans `dictionaries.json` local + miroir HF best-effort (asynchrone, en background task FastAPI, non bloquant).

Justification : le projet a déjà toute l'infrastructure (`clean_data.py`, `validate_clean.py`, `CURRENT`). Il suffit de la **connecter** au backend. Pas de PostgreSQL : le corpus tient en RAM, les requêtes sont des groupby pandas, pas du SQL relationnel.

---

## 11. Plan de migration progressif

### Phase 0 — Audit & sauvegarde *(0,5 j)*
- **Objectif** : figer un point de référence avant de toucher.
- **Livrables** : tag git `pre-migration`, copie de `dictionaries.json`, snapshot `data/clean/CURRENT`.
- **Risque** : aucun.
- **Validation** : `git tag` + checksum du dico.

### Phase 1 — Extraction des fonctions pures vers `ina_core` *(1–2 j)*
- **Objectif** : créer le package sans casser `app.py`.
- **Fichiers** : nouveau dossier `backend/ina_core/`. `app.py` reste le seul producteur, mais on **importe** depuis `ina_core` au lieu de définir les fonctions inline.
- **Livrables** : `ina_core` v0 + `app.py` qui passe par lui.
- **Risque** : régression silencieuse si la signature change. Mitigation : copies textuelles d'abord, refactor cosmétique seulement.
- **Validation** : `streamlit run app.py` se lance, KPIs identiques sur le thème `chomage`, période complète.

### Phase 2 — Tests unitaires sur `ina_core` *(1 j)*
- **Objectif** : couvrir les 4 fonctions critiques.
- **Livrables** : `tests/` avec ~20 tests, fixtures parquet ~50 lignes, CI locale `pytest`.
- **Risque** : faible. Cf. section 12.
- **Validation** : `pytest -q` vert + couverture > 80% sur `ina_core/{text,tagging,aggregation,dictionaries}.py`.

### Phase 3 — DataStore & DictRepository *(1 j)*
- **Objectif** : isoler les I/O de `core`.
- **Livrables** : `data_store.py`, `dict_repo.py` avec deux implémentations + tests par mock HTTP.
- **Risque** : la sauvegarde HF en background task peut perdre des écritures si le serveur crashe. Documenter le compromis.
- **Validation** : tests + benchmark cold start <60s en local.

### Phase 4 — FastAPI minimal en parallèle *(2 j)*
- **Objectif** : routes `/health`, `/metadata`, `/themes`, `/analysis`, `/export/csv`. Pas de frontend encore.
- **Livrables** : `ina_api/` runnable, `uvicorn ina_api.main:app`. Streamlit reste en service.
- **Risque** : divergence des résultats API vs Streamlit. Mitigation : test de cohérence dédié (chaque thème → comparaison `stats` byte-à-byte).
- **Validation** : Postman/curl sur les 6 routes, OpenAPI doc accessible.

### Phase 5 — Frontend Vite/React minimal *(3–4 j)*
- **Objectif** : page « Analysis » qui consomme `/analysis`. Filtres : thème, période, chaînes, fréquence, mode.
- **Livrables** : `frontend/` avec une page fonctionnelle, charts recharts, KPI cards.
- **Risque** : confort UX inférieur à Streamlit au début. Garder Streamlit accessible.
- **Validation** : feature parity sur la vue principale (KPIs + 3 graphes + preview + export).

### Phase 6 — Remplacement progressif de Streamlit *(2 j)*
- **Objectif** : pages secondaires (CRUD thèmes, couverture chaînes), retirer `app.py` du run par défaut.
- **Livrables** : frontend complet, `app.py` archivé en `legacy/app.py` (pas supprimé).
- **Risque** : utilisateurs qui passent à côté de la nouvelle URL. Mitigation : redirection.
- **Validation** : 1 semaine d'usage sans rapport de bug bloquant.

### Phase 7 — Déploiement VPS OVH *(1–2 j)*
- **Objectif** : prod minimale.
- **Stack** : nginx → uvicorn (gunicorn -k uvicorn) ; frontend statique servi par nginx ; systemd unit ; `.env` pour `HF_TOKEN`.
- **Livrables** : `Dockerfile` backend, `docker-compose.yml`, runbook.
- **Risque** : sécurité du `HF_TOKEN`, CORS. Mitigation : token en `/etc/...` 600, allow-list CORS.
- **Validation** : `https://…/health` 200, `/themes` lecture publique, `/themes` écriture protégée.

### Phase 8 — Optimisations *(à la demande)*
- Pré-calculer agrégats des thèmes stables.
- Migrer `dictionaries.json` vers PostgreSQL **uniquement si** : multi-écriture concurrente régulière, audit trail réclamé, ou >100 thèmes.
- Redis si plusieurs workers uvicorn (cache partagé).

---

## 12. Tests recommandés

Fixtures à créer : 2 petits parquets de ~50 lignes (`tests/fixtures/sample_a.parquet`, `sample_b.parquet`) avec colonnes `_date`, `_channel`, `title`, `_title_norm`, `source_file` — couvrant cas dégénérés (titre vide, accents, ambiguïté UP+DOWN, date manquante).

### Exemples concrets

```python
# test_text.py
def test_normalize_text_strips_accents():
    assert normalize_text("Inflation à la hausse") == "inflation a la hausse"
def test_normalize_text_handles_none(): assert normalize_text(None) == ""
def test_normalize_text_lowercases(): assert normalize_text("CHÔMAGE") == "chomage"

# test_tagging.py
def test_count_occurrences_short_word_uses_word_boundary():
    # "ipc" est court : ne doit pas matcher "epicéa"
    assert count_occurrences("epicea fournit", ["ipc"]) == 0
def test_tag_direction_ambiguous():
    # un titre qui contient hausse ET baisse → AMBIGUOUS (=2)
    df = make_df(["inflation en hausse mais baisse possible"])
    out = tag_dataframe(df, concept=["inflation"], up=["hausse"], down=["baisse"])
    assert out.loc[0, "direction"] == 2
def test_tag_no_concept_returns_flat():
    df = make_df(["actualité diverse"])
    out = tag_dataframe(df, concept=["inflation"], up=["hausse"], down=["baisse"])
    assert out.loc[0, "is_match"] == 0 and out.loc[0, "direction"] == 0

# test_aggregation.py
def test_aggregate_monthly_basic_counts():
    df = tagged_fixture(...)
    stats = aggregate_by_period(df, "Mensuelle")
    assert stats.loc[stats.period_start == "2020-01-01", "matched_titles"].iat[0] == 3
def test_aggregate_frequency_zero_when_no_titles():
    # période vide → frequency NaN, pas crash

# test_dictionaries.py
def test_normalize_payload_drops_invalid_keys()
def test_normalize_payload_accepts_legacy_list_form()  # raw_theme: list (compat)

# test_api_themes.py (TestClient FastAPI)
def test_create_theme_returns_201_and_persists()
def test_create_existing_theme_returns_409()
def test_delete_last_theme_returns_400()
def test_update_theme_invalidates_analysis_cache()

# test_api_analysis.py
def test_analysis_with_unknown_theme_returns_404()
def test_analysis_filter_by_channel_excludes_others()
def test_export_csv_streams_with_correct_separator()
```

Cible : **20 tests unitaires + 10 tests API**, couverture 80%+ sur `ina_core`. Pytest + pytest-cov + httpx (TestClient).

---

## 13. Quick wins immédiats (avant migration)

Triés par ratio impact/effort. **Aucun ne nécessite de refactor profond, tous applicables sur `app.py` actuel.**

| # | Quick win | Fichier:lignes | Effort | Impact |
|---|---|---|---|---|
| 1 | Charger les dicos HF **une seule fois par session** : aujourd'hui, `load_dictionaries_from_hf` est appelé à chaque rerun où `"dictionaries" not in st.session_state` est faux — sauf que c'est OK ; **mais** la sauvegarde HF est bloquante. Passer `save_dictionaries_to_hf` en `threading.Thread(daemon=True)` ou afficher un spinner. | [app.py:782-786,822-825,891-893,932-935,949-953](app.py) | 30 min | UI plus fluide, créations/maj non bloquantes |
| 2 | Ajouter un **fallback local** dans `load_clean_parquets_from_hf` : si HF échoue, lire `data/clean/CURRENT/*_clean.parquet`. Réduit la fragilité. | [app.py:210-253](app.py#L210-L253) | 1 h | App reste utilisable hors-ligne |
| 3 | Réduire la **copie de DataFrame** dans `add_tagging_columns_hier` : aujourd'hui `df_base.copy()` est passé puis muté. Préférer retourner un nouveau df ou n'ajouter que les colonnes nécessaires côté caller. | [app.py:998](app.py#L998) | 30 min | -30 % mémoire par entrée de cache |
| 4 | Augmenter la limite du `_tagged_theme_cache` à 4 ou la rendre paramétrable (variable d'env) : 2 c'est trop peu si l'utilisateur jongle entre 3 thèmes. | [app.py:1008-1009](app.py#L1008-L1009) | 5 min | UX |
| 5 | Sécuriser les boutons « Toutes / Aucune » qui modifient `_ch_sel` puis ne `rerun()` pas : ajouter un `st.rerun()` explicite après `st.session_state["_ch_sel"] = …`. | [app.py:1076-1079](app.py#L1076-L1079) | 5 min | Cohérence d'état |
| 6 | Logger la **taille mémoire** du `_tagged_theme_cache` à chaque insertion (`df.memory_usage(deep=True).sum()`) pour diagnostiquer en prod. | [app.py:996-1009](app.py#L996-L1009) | 15 min | Observabilité |
| 7 | Centraliser la lecture de `HF_TOKEN` : aujourd'hui `st.secrets.get("HF_TOKEN", None)` est appelé à 3 endroits différents. Un seul `_hf_token = st.secrets.get("HF_TOKEN")` en haut de fichier. | [app.py:215,544,547](app.py) | 5 min | Lisibilité, prépare extraction core |
| 8 | Ajouter un **timeout court (10 s) + retry x2** sur les requêtes HF parquet, plutôt que `timeout=300` séquentiel. | [app.py:224](app.py#L224) | 30 min | Robustesse, fail-fast |
| 9 | **Dédupliquer `normalize_text`** : 6 redéfinitions identiques dans `app.py` + `scripts/clean_data.py` + `scripts/build_gold_sample*.py` + `scripts/predict_corpus*.py` + `scripts/train_classifier*.py`. Une seule source dans `ina_core/text.py` (anticipe Phase 1). | partout | 1 h | Cohérence garantie pipeline ↔ app |
| 10 | Ajouter un message clair quand le tagging produit **0 match** : actuellement on voit juste les KPIs à 0. Avertir si `concept_norm` ne matche aucun titre — possible erreur de dico. | après [app.py:1011](app.py#L1011) | 15 min | UX diagnostic |

---

## 14. Risques et décisions à prendre

### Risques
- **Concurrence d'écriture sur `dictionaries.json`** : last-write-wins. Si plusieurs analystes éditent en parallèle en prod, on perdra des modifs. Mitigation v1 : verrou `asyncio.Lock` côté API ; long terme : versioning Git du fichier ou migration DB.
- **HF rate-limiting / panne** : aujourd'hui = app down. Décision : exiger fallback local Phase 1.
- **Mémoire serveur** : multipliée par le nombre de thèmes activement consultés × users si on garde le pattern actuel. Décision : cache process unique côté API, plus de session_state.
- **Perte du contexte ML** : le pipeline `scripts/` consomme la même normalisation que l'app. Si on extrait `ina_core/text.py` mais qu'on ne met pas à jour les scripts, on aura **deux normalisations divergentes**. Décision : Phase 1 inclut le branchement des scripts sur `ina_core` (1 h).
- **Authentification frontend** : si l'app est publique, écrire dans les dicos doit être protégé. Décision : Bearer token simple v1 ; OAuth si besoin plus tard.
- **Migration des utilisateurs Streamlit** : risque de friction. Décision : garder Streamlit en parallèle pendant Phase 4-6.

### Décisions à arbitrer
1. **Hugging Face en prod : source canonique ou miroir ?** Recommandation : **miroir** lecture publique pour partage, mais la source de vérité est le `dictionaries.json` du backend versionné Git. Ça résout 80 % des risques de concurrence.
2. **Frontend : recharts vs Plotly.js ?** Recommandation : recharts (plus simple, rendu rapide, suffit pour ces charts). Plotly.js seulement si on veut conserver le rangeselector + rangeslider (utile mais pas vital).
3. **Auth utilisateur ?** Si l'app reste interne (1-3 personnes) : Bearer token statique. Si public : auth OAuth GitHub à prévoir Phase 7.
4. **Conserver le mode `intensity` (occurrences brutes) ?** Le code existe ([app.py:702-712](app.py#L702-L712)) mais le métier semble préférer `binary`. À confirmer avec l'utilisateur avant la migration frontend (ne pas porter ce qui ne sert pas).

---

## 15. Première checklist d'implémentation

À cocher dans l'ordre, **sans tout casser**. Estimation totale Phase 0–4 : **5–7 jours de travail concentré**.

- [ ] **P0.1** Tag git `pre-migration`, push.
- [ ] **P0.2** Backup de `dictionaries.json` et `data/clean/CURRENT` hors du dépôt.
- [ ] **P1.1** Créer `backend/ina_core/` + `pyproject.toml` (package éditable).
- [ ] **P1.2** Déplacer `normalize_text` → `ina_core/text.py`. Mettre à jour app.py + 5 scripts.
- [ ] **P1.3** Déplacer dictionnaires : `clean_term_list`, `normalize_theme_dictionary`, `normalize_dictionaries_payload`, `clone_dictionaries`, `empty_theme_dictionary` → `ina_core/dictionaries.py`.
- [ ] **P1.4** Déplacer tagging : `prepare_keywords`, `count_occurrences`, `add_tagging_columns_hier` (renommer en `tag_dataframe`, **toujours retourner une copie**) → `ina_core/tagging.py`. Constantes `DIRECTION_*` exposées.
- [ ] **P1.5** Déplacer agrégation : `periodize`, `aggregate_by_period`, `build_descriptive_table`, `build_top_channels` → `ina_core/aggregation.py`.
- [ ] **P1.6** Déplacer couverture : `build_channel_stats`, `build_decade_distribution` → `ina_core/coverage.py`.
- [ ] **P1.7** Lancer `streamlit run app.py`, refaire un cycle complet de test manuel sur thème `chomage`. Comparer KPIs visuellement avec le tag pre-migration.
- [ ] **P2.1** Installer `pytest`, `pytest-cov`, `pyarrow`. Créer `tests/fixtures/`.
- [ ] **P2.2** Écrire les ~20 tests listés en section 12.
- [ ] **P2.3** `pytest -q` vert. Coverage > 80% sur `ina_core/`.
- [ ] **P3.1** Quick wins #2 (fallback local), #8 (retry HF), #9 (dédup `normalize_text` déjà fait en P1.2).
- [ ] **P3.2** `DataStore` (charge HF ou local, expose `df_base`), `DictRepository` (Local + HF), `Cache` mémoire process.
- [ ] **P4.1** `pip install fastapi[standard] pydantic-settings`. Squelette `ina_api/`.
- [ ] **P4.2** Routes `/health`, `/metadata`, `/themes` (GET list + GET one).
- [ ] **P4.3** Routes `/themes` POST/PUT/DELETE avec invalidation du cache.
- [ ] **P4.4** Route `/analysis` (la plus importante) ; vérifier parité bit-à-bit avec Streamlit sur 3 thèmes.
- [ ] **P4.5** Route `/export/csv` streaming.
- [ ] **P4.6** Route `/channels`, `/coverage`, `/data/profile`.
- [ ] **P4.7** Tests TestClient.
- [ ] **P4.8** Démarrer `uvicorn` en parallèle de Streamlit pendant 1 semaine, comparer.

À ce stade, **rien n'a été cassé**, `app.py` tourne toujours, et on a un backend prêt à recevoir un frontend.

---

**Note** : aucun fichier n'a été modifié pendant cet audit. Les chemins, lignes et fonctions cités sont vérifiés directement dans le code observé.
