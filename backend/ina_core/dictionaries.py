"""Theme dictionary normalization helpers (no I/O)."""
from __future__ import annotations

import json
from typing import Dict, List


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
