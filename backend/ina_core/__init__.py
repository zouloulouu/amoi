"""ina_core: pure business logic for data_ina thematic analysis."""
from ina_core.text import normalize_text
from ina_core.dictionaries import (
    clean_term_list,
    clone_dictionaries,
    empty_theme_dictionary,
    normalize_dictionaries_payload,
    normalize_theme_dictionary,
)
from ina_core.tagging import (
    DIRECTION_AMBIGUOUS,
    DIRECTION_DOWN,
    DIRECTION_FLAT,
    DIRECTION_UP,
    count_occurrences,
    prepare_keywords,
    tag_dataframe,
)
from ina_core.aggregation import (
    aggregate_by_period,
    build_descriptive_table,
    build_top_channels,
    periodize,
)
from ina_core.coverage import (
    build_channel_stats,
    build_decade_distribution,
)

__version__ = "0.1.0"

__all__ = [
    "normalize_text",
    "clean_term_list",
    "clone_dictionaries",
    "empty_theme_dictionary",
    "normalize_dictionaries_payload",
    "normalize_theme_dictionary",
    "DIRECTION_AMBIGUOUS",
    "DIRECTION_DOWN",
    "DIRECTION_FLAT",
    "DIRECTION_UP",
    "count_occurrences",
    "prepare_keywords",
    "tag_dataframe",
    "aggregate_by_period",
    "build_descriptive_table",
    "build_top_channels",
    "periodize",
    "build_channel_stats",
    "build_decade_distribution",
]
