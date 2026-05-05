"""Spot tests for name and key normalization."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.match.normalize import normalize_ein, normalize_name, normalize_state, normalize_uei


def test_name_basic_normalization():
    assert normalize_name("The American Red Cross, Inc.") == "american red cross"
    assert normalize_name("Univ. of California Regents") == "university california regents"
    # Multi-word abbreviation expanded before single-word.
    assert normalize_name("Massachusetts Med Ctr") == "massachusetts medical center"


def test_name_handles_unicode_and_none():
    assert normalize_name(None) == ""
    # Diacritics are stripped (ASCII fold). "de" is Spanish, not in the stopword list.
    assert normalize_name("Académia de Niños") == "academia de ninos"


def test_state_and_keys():
    assert normalize_state("ca") == "CA"
    assert normalize_state(" Ny ") == "NY"
    assert normalize_state(None) == ""
    assert normalize_ein("12-3456789") == "123456789"
    assert normalize_ein("3456789") == "003456789"
    assert normalize_uei(" abcdef123 ") == "ABCDEF123"
