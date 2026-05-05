"""Recipient-match tier exercises.

The original test suite covered classifier hierarchy and name normalization,
but did not run a synthetic frame through `build_recipient_match` itself.
Tier 2 (deterministic name+state) wasn't exercised, so a column-drop bug in
the post-join reset_index call survived initial review.

These tests use minimal hand-crafted frames where each row's expected tier
is obvious by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.match.recipient_match import build_recipient_match


def _txn(rows):
    return pd.DataFrame(rows)


def _bmf(rows):
    df = pd.DataFrame(rows)
    df["SUBSECTION"] = "03"
    return df


def test_tier1_direct_ein_match():
    txn = _txn([
        {"recipient_uei": "U1", "recipient_name": "Hospital A",
         "recipient_state_code": "NY", "recipient_ein": "11-1111111",
         "federal_action_obligation": "100"},
    ])
    bmf = _bmf([
        {"EIN": "111111111", "NAME": "Hospital A", "STATE": "NY",
         "NTEE_CD": "E22", "FOUNDATION": "15"},
    ])
    match, stats, _ = build_recipient_match(txn, bmf)
    assert stats.tier1_ein == 1
    assert stats.tier5_unresolved == 0
    assert match.iloc[0]["irs_ein"] == "111111111"
    assert match.iloc[0]["bmf_ntee"] == "E22"


def test_tier2_deterministic_name_state_match():
    """Regression: previously failed with KeyError on STATE_norm because the
    post-join reset_index dropped the index columns."""
    txn = _txn([
        {"recipient_uei": "U1", "recipient_name": "American Heart Association",
         "recipient_state_code": "TX",
         "federal_action_obligation": "5000"},
    ])
    bmf = _bmf([
        {"EIN": "222222222", "NAME": "American Heart Association",
         "STATE": "TX", "NTEE_CD": "G40", "FOUNDATION": "15"},
        # Decoy in different state - must not match.
        {"EIN": "333333333", "NAME": "American Heart Association",
         "STATE": "FL", "NTEE_CD": "G40", "FOUNDATION": "15"},
    ])
    match, stats, _ = build_recipient_match(txn, bmf)
    assert stats.tier2_det_name == 1
    assert stats.tier1_ein == 0
    row = match.iloc[0]
    assert row["match_tier"] == 2
    assert row["irs_ein"] == "222222222"
    assert row["bmf_state"] == "TX"
    assert row["bmf_ntee"] == "G40"


def test_tier3_probabilistic_name_state_match():
    """Word reorder is the textbook Tier 3 case: literal strings differ so
    Tier 2 (exact name+state) won't fire, but token_set_ratio scores 100."""
    txn = _txn([
        {"recipient_uei": "U1", "recipient_name": "Heart Association American",
         "recipient_state_code": "TX", "federal_action_obligation": "5000"},
    ])
    bmf = _bmf([
        {"EIN": "222222222", "NAME": "American Heart Association",
         "STATE": "TX", "NTEE_CD": "G40", "FOUNDATION": "15"},
    ])
    match, stats, _ = build_recipient_match(txn, bmf)
    assert stats.tier2_det_name == 0  # literal strings differ
    assert stats.tier3_prob_name == 1
    assert match.iloc[0]["irs_ein"] == "222222222"


def test_tier5_unresolved_review_queue():
    txn = _txn([
        {"recipient_uei": "U1", "recipient_name": "Some Org Nobody Knows",
         "recipient_state_code": "ZZ",
         "federal_action_obligation": "12345"},
    ])
    bmf = _bmf([
        {"EIN": "999999999", "NAME": "Different Org", "STATE": "CA",
         "NTEE_CD": "B50", "FOUNDATION": "15"},
    ])
    match, stats, review = build_recipient_match(txn, bmf)
    assert stats.tier5_unresolved == 1
    assert match.iloc[0]["match_tier"] == 5
    assert len(review) == 1
    assert float(review.iloc[0]["total_obligated"]) == 12345.0
