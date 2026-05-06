"""Smoke tests for the QA workbook generator (scripts/build_qa_workbooks.py).

The workbook generator is the artifact handed to non-technical clients, so
correctness of the URLs and formatting helpers matters. The full workbook
build is exercised manually; these tests cover the URL helper and any
small pure functions.
"""
from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# scripts/ is not a package; import the file directly.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "qa_workbooks", ROOT / "scripts" / "build_qa_workbooks.py"
)
qa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa)


def test_usaspending_search_url_with_name():
    """Recipient-name URL goes to USAspending keyword_search with proper encoding."""
    url = qa.usaspending_search_url("CLIMATE UNITED FUND", None)
    assert url.startswith("https://www.usaspending.gov/keyword_search/")
    # Spaces must be percent-encoded.
    assert "CLIMATE%20UNITED%20FUND" in url
    # Verify the URL parses.
    parsed = urllib.parse.urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "www.usaspending.gov"


def test_usaspending_search_url_falls_back_to_uei():
    """Empty/None name falls back to UEI as the keyword."""
    url = qa.usaspending_search_url(None, "ABC123XYZ")
    assert url == "https://www.usaspending.gov/keyword_search/ABC123XYZ"

    url2 = qa.usaspending_search_url("", "ABC123XYZ")
    assert url2 == "https://www.usaspending.gov/keyword_search/ABC123XYZ"


def test_usaspending_search_url_special_characters_encoded():
    """Names with special characters must be percent-encoded so the URL is valid."""
    url = qa.usaspending_search_url("Foo & Bar, Inc.", None)
    # & must be encoded as %26, comma as %2C
    assert "Foo%20%26%20Bar%2C%20Inc." in url
    # No raw ampersand or unencoded comma.
    keyword_part = url.rsplit("/", 1)[1]
    assert "&" not in keyword_part
    assert "," not in keyword_part


def test_usaspending_search_url_empty_inputs():
    """Both inputs empty returns empty string, not a malformed URL."""
    assert qa.usaspending_search_url(None, None) == ""
    assert qa.usaspending_search_url("", "") == ""
    assert qa.usaspending_search_url("   ", "   ") == ""


def test_state_name_lookup():
    """State-code lookup expands two-letter codes; passes through unknown."""
    assert qa.state_name("NY") == "New York"
    assert qa.state_name("ca") == "California"  # case-insensitive
    assert qa.state_name(" tx ") == "Texas"  # whitespace trim
    assert qa.state_name("DC") == "District of Columbia"
    assert qa.state_name("ZZ") == "ZZ"  # unknown passes through
    assert qa.state_name(None) == ""
    assert qa.state_name("") == ""


def test_translate_exclusion_reason_plain_english():
    """Codes turn into human-readable exclusion phrases."""
    code_to_label = {
        "Q": "For-Profit Organization (Other than Small Business)",
        "A": "State Government",
        "X": "Other",
    }
    assert qa.translate_exclusion_reason("no_M_tag", code_to_label) == \
        "No 501(c)(3) tag (M) was found on any of this recipient's transactions."

    single = qa.translate_exclusion_reason("disqualifying_cotag:Q", code_to_label)
    assert "For-Profit Organization" in single
    assert "mutually exclusive with 501(c)(3) status" in single

    multi = qa.translate_exclusion_reason("disqualifying_cotag:A+Q", code_to_label)
    assert "State Government" in multi
    assert "For-Profit Organization" in multi


def test_fmt_business_types():
    """Business-types-set string renders as comma-separated readable list."""
    assert qa.fmt_business_types("M") == "M"
    assert qa.fmt_business_types("MX") == "M, X"  # Sorted by character.
    assert qa.fmt_business_types("MEK") == "E, K, M"  # alphabetic ordering
    assert qa.fmt_business_types("") == ""
    assert qa.fmt_business_types(None) == ""
