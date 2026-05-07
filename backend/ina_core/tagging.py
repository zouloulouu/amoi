"""Pure tagging logic: keyword preparation, occurrence counting, dataframe tagging."""
from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from ina_core.text import normalize_text

# Direction signal — 4 states
DIRECTION_UP = 1         # at least one UP term, no DOWN term
DIRECTION_DOWN = -1      # at least one DOWN term, no UP term
DIRECTION_FLAT = 0       # no UP/DOWN term (or title not matched)
DIRECTION_AMBIGUOUS = 2  # both UP and DOWN terms in the same title


def prepare_keywords(keywords: List[str]) -> List[str]:
    """Normalize, dedupe, and sort a list of keyword strings."""
    normalized = [normalize_text(k) for k in keywords if str(k).strip()]
    return sorted(set(k for k in normalized if k))


def count_occurrences(text_norm: str, keywords_norm: List[str]) -> int:
    """Count raw keyword occurrences in a normalized text.

    Short alphabetic keywords (≤4 chars) use word-boundary matching to avoid
    false positives like "ipc" matching "epicea".
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


def tag_dataframe(
    df: pd.DataFrame,
    title_col: str,
    concept_norm: List[str],
    up_norm: List[str],
    down_norm: List[str],
    title_norm_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a NEW DataFrame with tagging columns added.

    Adds: occ_concept, occ_context (=0), occ_up, occ_down, is_concept,
    is_context (=0), is_match_broad, is_match_strict, is_match, direction.

    Direction is one of DIRECTION_FLAT|UP|DOWN|AMBIGUOUS.
    The input DataFrame is NOT mutated.
    """
    df = df.copy()

    if title_norm_col and title_norm_col in df.columns:
        titles_norm = df[title_norm_col].fillna("").astype(str)
    else:
        titles_norm = df[title_col].fillna("").astype(str).map(normalize_text)

    df["occ_concept"] = titles_norm.map(lambda x: count_occurrences(x, concept_norm))
    df["occ_context"] = 0
    df["occ_up"] = titles_norm.map(lambda x: count_occurrences(x, up_norm))
    df["occ_down"] = titles_norm.map(lambda x: count_occurrences(x, down_norm))

    df["is_concept"] = (df["occ_concept"] > 0).astype("int8")
    df["is_context"] = 0
    df["is_match_broad"] = df["is_concept"].astype("int8")
    df["is_match_strict"] = df["is_match_broad"].astype("int8")
    df["is_match"] = df["is_match_broad"].astype("int8")

    has_up = df["occ_up"] > 0
    has_down = df["occ_down"] > 0
    matched = df["is_match"] == 1

    direction = pd.Series(DIRECTION_FLAT, index=df.index, dtype="int8")
    direction.loc[matched & has_up & ~has_down] = DIRECTION_UP
    direction.loc[matched & has_down & ~has_up] = DIRECTION_DOWN
    direction.loc[matched & has_up & has_down] = DIRECTION_AMBIGUOUS
    df["direction"] = direction

    return df
