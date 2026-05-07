"""Tests for ina_core.text — string normalization."""
from __future__ import annotations

import math

from ina_core import normalize_text


def test_strips_accents():
    assert normalize_text("Inflation à la HAUSSE") == "inflation a la hausse"


def test_handles_chomage_with_circumflex():
    assert normalize_text("CHÔMAGE") == "chomage"


def test_returns_empty_string_for_none():
    assert normalize_text(None) == ""


def test_returns_empty_string_for_nan():
    assert normalize_text(math.nan) == ""


def test_coerces_int_to_str():
    assert normalize_text(42) == "42"


def test_coerces_bool():
    assert normalize_text(True) == "true"


def test_strips_whitespace():
    assert normalize_text("   inflation   ") == "inflation"


def test_empty_string_stays_empty():
    assert normalize_text("") == ""
