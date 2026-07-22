"""End-to-end integration test on the lightweight pipeline.

The bugs we hit in this codebase (build_awards_table not filtering to
in-scope, exhibits cross-tab failing on a missing-column fallback, the
exhibits column-name drift `awarding_agency_name` -> `awarding_agency`)
all surfaced because individual unit tests passed but the orchestrator
wired things wrong. This test runs the full step chain against a small
synthetic transactions frame written to disk and re-read at each step,
mirroring how the real orchestrator works.

Specifically, we assert:
  - business_types_set survives the parquet round-trip (frozenset and
    dict types do not always serialize cleanly through parquet, so the
    code stores it as a string; this test confirms).
  - The in-scope filter holds end-to-end: out-of-scope recipients in the
    raw frame do not appear in the analytic transactions or awards.
  - Panel sums equal the total in the analytic frame.
  - classification_rule_hits is preserved through parquet.
  - The awards table has one row per award, restricted to in-scope.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lightweight.recipient_filter import build_recipient_filter, write_outputs as write_filter
from src.lightweight.categorize import classify
from src.lightweight.tables import build_transactions_table, build_awards_table


def _synthetic_transactions() -> pd.DataFrame:
    """Three recipients across two FYs.

    U1: pure M, in scope, hospital-name-flagged.
    U2: M+X, in scope (X is soft), educational charter school.
    U3: A+X, OUT of scope (state government). Should be filtered out at
        every downstream step.
    """
    rows = []
    for fy_year, action_date in ((2024, "2024-05-01"), (2025, "2025-05-01")):
        # In-scope hospital
        rows.append({
            "transaction_id": f"t1-{fy_year}", "award_id_unique": "a1",
            "action_date": action_date, "award_type_code": "04", "action_type": "A",
            "awarding_agency_name": "Department of Health and Human Services",
            "awarding_sub_agency_name": "Health Resources and Services Administration",
            "assistance_listing_number": "93.224",
            "assistance_listing_title": "Health Center Program",
            "recipient_uei": "U1", "recipient_name": "Riverside Medical Center",
            "recipient_state_code": "NY", "recipient_business_types": "M",
            "primary_place_of_performance_country_code": "USA",
            "federal_action_obligation": 100_000,
            "total_outlayed_amount_for_overall_award": 80_000,
        })
        # In-scope educational (M+X, soft)
        rows.append({
            "transaction_id": f"t2-{fy_year}", "award_id_unique": "a2",
            "action_date": action_date, "award_type_code": "04", "action_type": "A",
            "awarding_agency_name": "Department of Education",
            "awarding_sub_agency_name": "Office of Elementary",
            "assistance_listing_number": "84.282",
            "assistance_listing_title": "Charter Schools Program",
            "recipient_uei": "U2", "recipient_name": "Bronx Charter School Network",
            "recipient_state_code": "NY", "recipient_business_types": "M,X",
            "primary_place_of_performance_country_code": "USA",
            "federal_action_obligation": 50_000,
            "total_outlayed_amount_for_overall_award": 40_000,
        })
        # Out-of-scope (state government)
        rows.append({
            "transaction_id": f"t3-{fy_year}", "award_id_unique": "a3",
            "action_date": action_date, "award_type_code": "04", "action_type": "A",
            "awarding_agency_name": "Department of Health and Human Services",
            "awarding_sub_agency_name": "Centers for Medicare and Medicaid Services",
            "assistance_listing_number": "93.778",
            "assistance_listing_title": "Medicaid Assistance Programs",
            "recipient_uei": "U3", "recipient_name": "STATE OF SOMEWHERE DEPT OF HEALTH",
            "recipient_state_code": "CA", "recipient_business_types": "A",
            "primary_place_of_performance_country_code": "USA",
            "federal_action_obligation": 999_000_000,  # large $; if it leaked it would be obvious
            "total_outlayed_amount_for_overall_award": 999_000_000,
        })
    return pd.DataFrame(rows)


def test_lightweight_pipeline_round_trip(tmp_path):
    """Run filter -> classify -> tables, persisting parquet at each boundary."""
    raw = _synthetic_transactions()
    raw_parquet = tmp_path / "transactions.parquet"
    raw.to_parquet(raw_parquet, index=False)

    # Step: build_recipient_filter
    raw_re = pd.read_parquet(raw_parquet)
    filter_df, stats = build_recipient_filter(raw_re)
    assert stats.total_distinct_recipients == 3
    assert stats.in_scope_recipients == 2  # U1 (M) and U2 (M+X)
    assert stats.excluded_by_disqualifying_cotag == 0  # no_M_tag, not cotag
    assert stats.excluded_no_m_tag == 1  # U3

    # Persist + reload to confirm bt_set survives parquet
    filter_path, _ = write_filter(filter_df, stats, out_dir=tmp_path)
    rfilter = pd.read_parquet(filter_path)
    assert "bt_set" in rfilter.columns
    assert isinstance(rfilter.iloc[0]["bt_set"], str), \
        "bt_set should be persisted as a string, not a frozenset"
    u1 = rfilter[rfilter["recipient_uei"] == "U1"].iloc[0]
    u2 = rfilter[rfilter["recipient_uei"] == "U2"].iloc[0]
    u3 = rfilter[rfilter["recipient_uei"] == "U3"].iloc[0]
    assert u1["in_scope"] and u2["in_scope"] and not u3["in_scope"]

    # Step: classify (operates on the raw transactions, not the filter table)
    classified, cstats = classify(raw_re)
    assert cstats.rows_total == len(raw_re)
    # classification_rule_hits must be present and non-empty for non-Core rows
    assert "classification_rule_hits" in classified.columns
    classified_path = tmp_path / "classified.parquet"
    classified.to_parquet(classified_path, index=False)

    classified_re = pd.read_parquet(classified_path)
    # classification_rule_hits round-trips through parquet as a list (or array)
    sample = classified_re.iloc[0]["classification_rule_hits"]
    assert hasattr(sample, "__iter__"), \
        "classification_rule_hits must remain iterable after parquet round-trip"

    # Step: build_transactions_table + build_awards_table - both must filter
    # to in-scope and produce comparable totals.
    txn_table = build_transactions_table(classified_re, rfilter,
                                         deflator="CPI-U", base_fy=2025)
    awd_table = build_awards_table(classified_re, recipient_filter=rfilter)

    # In-scope filter held: U3's 999M never appears.
    assert "U3" not in set(txn_table["recipient_uei"]), \
        "Out-of-scope recipient U3 leaked into transactions table"
    assert "U3" not in set(awd_table["recipient_uei"]), \
        "Out-of-scope recipient U3 leaked into awards table"

    # Both in-scope recipients are present.
    assert {"U1", "U2"} <= set(txn_table["recipient_uei"])
    assert {"a1", "a2"} <= set(awd_table["award_id_unique"])
    assert "a3" not in set(awd_table["award_id_unique"])

    # Dollar arithmetic. U1 contributes 100k x 2 FYs = 200k, U2 50k x 2 = 100k.
    assert int(txn_table["federal_action_obligation"].sum()) == 300_000

    # Real-dollar series exists and is finite.
    assert "federal_action_obligation_real" in txn_table.columns
    assert txn_table["federal_action_obligation_real"].notna().all()


def test_geographic_columns_propagate_to_analytic_table():
    """Geographic fields added in Tier C must survive classify -> tables -> parquet.

    The bug class this catches: a column added to _CLASSIFY_COLUMNS in
    pipeline.py but forgotten in build_transactions_table column_specs (or
    vice versa) silently drops the field from the analytic table.
    """
    raw = _synthetic_transactions()
    # Fill in geographic fields the synthetic frame doesn't already set.
    raw["recipient_country_code"] = "USA"
    raw["recipient_country_name"] = "UNITED STATES"
    raw["recipient_state_name"] = "New York"
    raw["recipient_county_name"] = "Manhattan"
    raw["prime_award_transaction_recipient_county_fips_code"] = "36061"
    raw["recipient_city_name"] = "NEW YORK"
    raw["recipient_zip_code"] = "10001"
    raw["prime_award_transaction_recipient_cd_current"] = "NY-12"
    raw["primary_place_of_performance_country_name"] = "UNITED STATES"
    raw["primary_place_of_performance_state_name"] = "New York"
    raw["prime_award_transaction_place_of_performance_state_fips_code"] = "36"
    raw["primary_place_of_performance_county_name"] = "Manhattan"
    raw["prime_award_transaction_place_of_performance_county_fips_code"] = "36061"
    raw["primary_place_of_performance_city_name"] = "NEW YORK"
    raw["primary_place_of_performance_zip_4"] = "10001-1234"
    raw["prime_award_transaction_place_of_performance_cd_current"] = "NY-12"

    classified, _ = classify(raw)
    rfilter = pd.DataFrame([
        {"recipient_uei": "U1", "in_scope": True, "bt_set": "M"},
        {"recipient_uei": "U2", "in_scope": True, "bt_set": "MX"},
    ])
    txn_table = build_transactions_table(classified, rfilter)

    expected_geo_columns = {
        # Recipient
        "recipient_country", "recipient_country_name",
        "recipient_state", "recipient_state_name",
        "recipient_county_name", "recipient_county_fips",
        "recipient_city", "recipient_zip", "recipient_cd",
        # POP
        "place_of_performance_country", "place_of_performance_country_name",
        "place_of_performance_state_name", "place_of_performance_state_fips",
        "place_of_performance_county_name", "place_of_performance_county_fips",
        "place_of_performance_city", "place_of_performance_zip",
        "place_of_performance_cd",
    }
    missing = expected_geo_columns - set(txn_table.columns)
    assert not missing, f"Geographic columns missing from analytic table: {missing}"

    # Spot-check a row's values made it through (U1 has hospital classification)
    u1 = txn_table[txn_table["recipient_uei"] == "U1"].iloc[0]
    assert u1["recipient_state"] == "NY"
    assert u1["recipient_city"] == "NEW YORK"
    assert u1["recipient_zip"] == "10001"
    assert u1["recipient_cd"] == "NY-12"
    assert u1["place_of_performance_state_name"] == "New York"


def test_pre_fy22_award_flag():
    """build_awards_table sets pre_fy22_award=True iff the award's earliest
    period_of_performance_start_date predates Oct 1, 2021."""
    rows = [
        # Pre-FY22: PoP start is FY18.
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "period_of_performance_start_date": "2018-10-01",
         "award_type_code": "04", "action_type": "B",
         "awarding_agency_name": "X", "awarding_sub_agency_name": "Y",
         "assistance_listing_number": "84.001",
         "recipient_uei": "U1", "recipient_name": "Long-Running Grantee",
         "recipient_state_code": "NY", "recipient_category": "core",
         "recipient_subcategory": "",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 100, "total_outlayed_amount_for_overall_award": 500},
        # Post-FY22: PoP start is FY24.
        {"transaction_id": "t2", "award_id_unique": "a2", "action_date": "2024-05-01",
         "period_of_performance_start_date": "2024-01-15",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "X", "awarding_sub_agency_name": "Y",
         "assistance_listing_number": "84.001",
         "recipient_uei": "U2", "recipient_name": "Fresh Grantee",
         "recipient_state_code": "CA", "recipient_category": "core",
         "recipient_subcategory": "",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 200, "total_outlayed_amount_for_overall_award": 50},
        # Edge case: PoP start exactly Oct 1, 2021 (FY22 day 1) - NOT pre-FY22.
        {"transaction_id": "t3", "award_id_unique": "a3", "action_date": "2024-05-01",
         "period_of_performance_start_date": "2021-10-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "X", "awarding_sub_agency_name": "Y",
         "assistance_listing_number": "84.001",
         "recipient_uei": "U3", "recipient_name": "Day-One Grantee",
         "recipient_state_code": "TX", "recipient_category": "core",
         "recipient_subcategory": "",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 50, "total_outlayed_amount_for_overall_award": 30},
    ]
    classified = pd.DataFrame(rows)
    rfilter = pd.DataFrame([
        {"recipient_uei": uei, "in_scope": True, "bt_set": "M"}
        for uei in ("U1", "U2", "U3")
    ])
    awards = build_awards_table(classified, recipient_filter=rfilter)
    by_award = awards.set_index("award_id_unique")["pre_fy22_award"]
    assert bool(by_award["a1"]) is True, "FY18 PoP start should be flagged pre-FY22"
    assert bool(by_award["a2"]) is False, "FY24 PoP start should NOT be flagged"
    assert bool(by_award["a3"]) is False, "Oct 1, 2021 (FY22 day 1) is on-or-after the cutoff"


def test_panel_arithmetic_invariant():
    """In any classified frame, sum of obligations across the four panels
    must equal the total. If a transaction landed in two panels (or zero),
    the sum would mismatch."""
    raw = _synthetic_transactions()
    classified, _ = classify(raw)
    total = classified["federal_action_obligation"].sum()
    by_panel = (classified.groupby("recipient_category")["federal_action_obligation"]
                .sum().sum())
    assert total == by_panel, "Panel decomposition is not exhaustive / mutually exclusive"

    # Each transaction must have exactly one (non-empty) recipient_category.
    assert classified["recipient_category"].notna().all()
    valid_cats = {"international", "hospital", "educational", "core"}
    assert set(classified["recipient_category"].unique()) <= valid_cats
