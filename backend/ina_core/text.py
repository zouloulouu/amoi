"""Text normalization utilities."""
from __future__ import annotations

import unicodedata


def normalize_text(value) -> str:
    """Lowercase, strip whitespace, and remove diacritics.

    Returns "" for None and NaN. Other non-string values are coerced via str().
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN check
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = value.strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text
