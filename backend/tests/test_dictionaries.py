"""Tests for ina_core.dictionaries — pure normalization helpers."""
from __future__ import annotations

from ina_core import (
    clean_term_list,
    clone_dictionaries,
    empty_theme_dictionary,
    normalize_dictionaries_payload,
    normalize_theme_dictionary,
)


def test_empty_theme_dictionary_has_four_buckets():
    out = empty_theme_dictionary()
    assert set(out.keys()) == {"concept", "context", "up", "down"}
    assert all(v == [] for v in out.values())


def test_clean_term_list_strips_and_filters_empty():
    assert clean_term_list(["  inflation  ", "", "  ", "ipc"]) == ["inflation", "ipc"]


def test_clean_term_list_returns_empty_for_non_list():
    assert clean_term_list("not a list") == []
    assert clean_term_list(None) == []
    assert clean_term_list({"concept": ["x"]}) == []


def test_normalize_theme_dictionary_accepts_legacy_list_form():
    """Backwards compat: a bare list is treated as concept-only."""
    out = normalize_theme_dictionary(["inflation", "ipc"])
    assert out == {"concept": ["inflation", "ipc"], "context": [], "up": [], "down": []}


def test_normalize_theme_dictionary_accepts_dict_form():
    raw = {"concept": ["a"], "up": ["hausse"], "down": ["baisse"], "context": []}
    out = normalize_theme_dictionary(raw)
    assert out["concept"] == ["a"]
    assert out["up"] == ["hausse"]
    assert out["down"] == ["baisse"]


def test_normalize_theme_dictionary_returns_empty_for_invalid():
    assert normalize_theme_dictionary(None) == empty_theme_dictionary()
    assert normalize_theme_dictionary(42) == empty_theme_dictionary()


def test_normalize_payload_drops_invalid_keys():
    raw = {
        "valid": {"concept": ["a"]},
        "": {"concept": ["b"]},          # empty key dropped
        "  ": {"concept": ["c"]},        # whitespace key dropped
        42: {"concept": ["d"]},          # non-str key dropped
    }
    out = normalize_dictionaries_payload(raw)
    assert set(out.keys()) == {"valid"}


def test_normalize_payload_returns_empty_for_non_dict():
    assert normalize_dictionaries_payload([]) == {}
    assert normalize_dictionaries_payload(None) == {}


def test_clone_dictionaries_is_deep_copy():
    src = {"theme": {"concept": ["inflation"]}}
    cloned = clone_dictionaries(src)
    cloned["theme"]["concept"].append("ipc")
    assert src["theme"]["concept"] == ["inflation"], "clone leaked into source"
