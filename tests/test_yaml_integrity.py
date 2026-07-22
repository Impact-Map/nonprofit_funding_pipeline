"""Reference-data integrity tests.

Every classification rule and panel decision is driven by YAML files under
reference_lists/. A typo or schema drift in one of those files silently
changes results, so these tests assert structural invariants on every YAML
we ship.

Specifically:
  - business_types_lightweight.yaml: required_code is set, no code is in
    both hard exclusions and soft codes, all codes are single uppercase
    letters from A-Z.
  - hospital_name_patterns.yaml / educational_name_patterns.yaml: every
    regex pattern compiles successfully (re.compile doesn't raise).
  - intl_subagencies.yaml: every subagency entry has a non-empty canonical
    field.
  - intl_listings.yaml: prefix_includes is a list of strings.
  - intl_subcategory_rules.yaml: each named sub-category has at least one
    keyword/listing/prefix specified.
  - covid_programs.yaml: listings is a list.
  - ntee_categories.yaml: B / E / Q ranges have valid start/end strings.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def _load(name: str):
    return yaml.safe_load((config.REFERENCE_LISTS / name).read_text(encoding="utf-8"))


def test_business_types_lightweight_yaml():
    doc = _load("business_types_lightweight.yaml")
    assert doc["required_code"] == "M", "M is the required code per methodology"
    excluded = set((doc.get("excluded_codes") or {}).keys())
    soft = set((doc.get("soft_codes") or {}).keys())
    permitted = set((doc.get("permitted_cotags") or {}).keys())

    # No code may appear in both excluded and soft.
    assert excluded.isdisjoint(soft), \
        f"Codes in both excluded and soft: {excluded & soft}"

    # Every code must be a single uppercase letter A-Z.
    for code in excluded | soft | permitted | {doc["required_code"]}:
        assert isinstance(code, str) and len(code) == 1 and "A" <= code <= "Z", \
            f"Bad code: {code!r}"

    # Sanity: hard list should include the obvious anti-501(c)(3) tags.
    for required_excl in ("A", "Q", "P", "N", "H", "O"):
        assert required_excl in excluded, f"{required_excl} should be excluded"


def test_hospital_name_patterns_compile():
    doc = _load("hospital_name_patterns.yaml")
    assert "patterns" in doc
    for entry in doc["patterns"]:
        assert "label" in entry and "regex" in entry, f"missing fields in {entry}"
        re.compile(entry["regex"])  # raises re.error on a bad regex
    for entry in doc.get("exclusions", []) or []:
        assert "label" in entry and "regex" in entry
        re.compile(entry["regex"])


def test_educational_name_patterns_compile():
    doc = _load("educational_name_patterns.yaml")
    assert "patterns" in doc
    for entry in doc["patterns"] + (doc.get("research_patterns") or []):
        assert "label" in entry and "regex" in entry
        re.compile(entry["regex"])
    # research_agency_keywords must exist and be non-empty for the
    # research-pattern rule to actually fire.
    assert doc.get("research_agency_keywords"), \
        "research_agency_keywords cannot be empty - the research pattern requires it"


def test_intl_subagencies_yaml():
    doc = _load("intl_subagencies.yaml")
    subs = doc.get("subagencies", [])
    assert subs, "intl_subagencies has zero entries"
    for entry in subs:
        canonical = (entry.get("canonical") or "").strip()
        assert canonical, f"Empty canonical in entry: {entry}"


def test_intl_listings_yaml():
    doc = _load("intl_listings.yaml")
    prefixes = doc.get("prefix_includes", [])
    assert isinstance(prefixes, list) and prefixes, \
        "prefix_includes must be a non-empty list"
    for p in prefixes:
        assert isinstance(p, str) and p.endswith("."), \
            f"Bad listing prefix {p!r} (should look like '19.')"


def test_intl_subcategory_rules_yaml():
    doc = _load("intl_subcategory_rules.yaml")
    expected_subcats = {"humanitarian", "global_health", "governance_economic",
                        "security", "education_research"}
    assert expected_subcats <= set(doc.keys()), \
        f"Missing sub-categories: {expected_subcats - set(doc.keys())}"
    for cat in expected_subcats:
        block = doc[cat]
        # Each sub-category must have at least one signal source defined.
        signals = (
            (block.get("subagency_keywords") or [])
            + (block.get("listings") or [])
            + (block.get("listing_prefixes") or [])
            + (block.get("program_keywords") or [])
        )
        assert signals, f"Sub-category {cat!r} has no signals defined"


def test_covid_programs_yaml():
    doc = _load("covid_programs.yaml")
    listings = doc.get("listings") or []
    assert isinstance(listings, list) and len(listings) > 0
    for code in listings:
        assert isinstance(code, str), f"Non-string listing: {code!r}"


def test_ntee_categories_yaml():
    doc = _load("ntee_categories.yaml")
    for major in ("educational", "hospital"):
        ranges = doc[major].get("ranges") or []
        assert ranges, f"{major} has no ranges"
        for r in ranges:
            assert isinstance(r["start"], str) and isinstance(r["end"], str)
            assert r["start"] <= r["end"], f"Bad range {r}"
