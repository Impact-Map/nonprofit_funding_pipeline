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
# Register in sys.modules BEFORE exec so decorators like @dataclass can
# resolve cls.__module__ via sys.modules.get() during class creation.
sys.modules["qa_workbooks"] = qa
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


# ---------------------------------------------------------------------------
# --source flag: config and metadata correctness
# ---------------------------------------------------------------------------

def test_source_configs_have_expected_shape():
    """Both source configs must expose the fields the builders read."""
    light = qa.SOURCES["lightweight"]()
    bmf = qa.SOURCES["bmf"]()
    for cfg in (light, bmf):
        assert cfg.name in ("lightweight", "bmf")
        assert cfg.label
        assert cfg.workbook_title
        assert cfg.methodology_desc
        assert cfg.out_subdir
        assert cfg.membership_column
        assert cfg.membership_label
        assert isinstance(cfg.extra_caveats, list)
        assert isinstance(cfg.include_reconciliation, bool)


def test_reconciliation_only_on_bmf_source():
    """The reconciliation sheet is a BMF-only artifact; the lightweight
    view has nothing meaningful to compare against."""
    assert qa.lightweight_config().include_reconciliation is False
    assert qa.bmf_config().include_reconciliation is True


def test_lightweight_and_bmf_use_different_output_dirs():
    """The two workbooks must not collide — they're delivered separately."""
    light = qa.lightweight_config()
    bmf = qa.bmf_config()
    assert light.out_subdir != bmf.out_subdir


def test_match_tier_label():
    """Numeric match_tier renders as plain-English phrase for the workbook."""
    assert qa._match_tier_label(1) == "Tier 1 (direct EIN)"
    assert qa._match_tier_label(2) == "Tier 2 (deterministic name+state)"
    assert qa._match_tier_label(3) == "Tier 3 (probabilistic name+state)"
    assert qa._match_tier_label(4) == "Tier 4 (manual override)"
    assert qa._match_tier_label(5) == "Tier 5 (unmatched)"
    # Robust to NaN and unexpected values (workbook must not crash).
    import math
    assert qa._match_tier_label(math.nan) == ""
    assert qa._match_tier_label(None) == ""
    assert qa._match_tier_label("not a number") == ""


def test_reconciliation_returns_empty_when_bmf_parquet_missing(tmp_path, monkeypatch):
    """sheet_reconciliation should silently return empty rather than crash
    when the BMF-backed analytic table doesn't exist. Same for lightweight."""
    # Repoint the config module the qa module captured at import time.
    fake_processed = tmp_path / "processed"
    fake_processed.mkdir()
    monkeypatch.setattr(qa.config, "PROCESSED", fake_processed)
    out = qa.sheet_reconciliation()
    assert out.empty


def test_bmf_config_carries_bmf_specific_caveats():
    """The BMF caveats sheet must explain the tier system and NCCS NTEE."""
    bmf = qa.bmf_config()
    joined = " ".join(row[1] for row in bmf.extra_caveats)
    # Not testing exact wording — testing that key concepts are present.
    assert "Tier" in joined
    assert "NTEE" in joined or "NCCS" in joined
    assert "IRS BMF" in joined


def test_lightweight_caveats_mention_bmf_alternative():
    """A client reading the lightweight workbook should know a more robust
    BMF-verified version exists."""
    light = qa.lightweight_config()
    joined = " ".join(row[1] for row in light.extra_caveats)
    assert "BMF" in joined
