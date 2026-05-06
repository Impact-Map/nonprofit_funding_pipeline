"""Phase 1 lightweight: M-rule and heuristic carve-out tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lightweight.recipient_filter import (
    parse_business_types,
    aggregate_business_types_per_recipient,
    apply_filter,
)
from src.lightweight.categorize import classify


def test_parse_business_types():
    assert parse_business_types("M") == frozenset({"M"})
    assert parse_business_types("M,X") == frozenset({"M", "X"})
    assert parse_business_types("M, X") == frozenset({"M", "X"})
    assert parse_business_types("M;X") == frozenset({"M", "X"})
    assert parse_business_types("MNX") == frozenset({"M", "N", "X"})
    assert parse_business_types("") == frozenset()
    assert parse_business_types(None) == frozenset()
    assert parse_business_types("  ") == frozenset()
    # Lowercase normalized.
    assert parse_business_types("m,x") == frozenset({"M", "X"})


def test_m_rule_includes_m_only():
    txn = pd.DataFrame([
        {"recipient_uei": "U1", "recipient_name": "Pure M Org",
         "recipient_state_code": "NY", "business_types_code": "M"},
    ])
    per = aggregate_business_types_per_recipient(txn)
    out, stats = apply_filter(per)
    assert stats.in_scope_recipients == 1
    assert out.iloc[0]["in_scope"] is True or bool(out.iloc[0]["in_scope"]) is True
    assert "M_only" in stats.cotag_distribution


def test_m_rule_tolerates_X_per_v2_soft_codes():
    """Rule v2: X = 'Other' is soft (informational), tolerated with M.

    Real example: ALIVIO MEDICAL CENTER (501c3 FQHC) was tagged M+X by
    different agencies and the v1 strict rule wrongly excluded it. v2
    tolerates the M+X combination.
    """
    txn = pd.DataFrame([
        {"recipient_uei": "U1", "recipient_name": "Alivio Medical Center",
         "recipient_state_code": "IL", "business_types_code": "M"},
        {"recipient_uei": "U1", "recipient_name": "Alivio Medical Center",
         "recipient_state_code": "IL", "business_types_code": "X"},
    ])
    per = aggregate_business_types_per_recipient(txn)
    out, stats = apply_filter(per)
    assert stats.in_scope_recipients == 1


def test_m_rule_still_excludes_hard_cotags():
    """Hard exclusions (Q, A, etc.) still disqualify, even when X also present."""
    txn = pd.DataFrame([
        # M+Q -> excluded (Q is hard)
        {"recipient_uei": "U1", "recipient_name": "Borderline For-Profit",
         "recipient_state_code": "NY", "business_types_code": "M,Q"},
        # M+W -> excluded (W is hard - non-US)
        {"recipient_uei": "U2", "recipient_name": "Foreign Org",
         "recipient_state_code": "", "business_types_code": "M,W"},
        # M+Q+X -> still excluded (Q is hard, X being soft doesn't save it)
        {"recipient_uei": "U3", "recipient_name": "Mixed",
         "recipient_state_code": "CA", "business_types_code": "M,Q,X"},
    ])
    per = aggregate_business_types_per_recipient(txn)
    _, stats = apply_filter(per)
    assert stats.in_scope_recipients == 0
    assert stats.excluded_by_disqualifying_cotag == 3


def test_m_rule_allows_non_excluded_cotags_E_K():
    txn = pd.DataFrame([
        {"recipient_uei": "U1", "recipient_name": "Regional Nonprofit",
         "recipient_state_code": "CA", "business_types_code": "M,E"},
        {"recipient_uei": "U2", "recipient_name": "Tribal Org",
         "recipient_state_code": "AZ", "business_types_code": "M,K"},
    ])
    per = aggregate_business_types_per_recipient(txn)
    out, stats = apply_filter(per)
    assert stats.in_scope_recipients == 2


def test_m_rule_excludes_S_V_T_U_per_user_decision():
    """User confirmed: S and V join T and U on the exclusion list."""
    txn = pd.DataFrame([
        {"recipient_uei": "U1", "recipient_name": "HSI",
         "recipient_state_code": "NM", "business_types_code": "M,S"},
        {"recipient_uei": "U2", "recipient_name": "HBCU",
         "recipient_state_code": "GA", "business_types_code": "M,T"},
        {"recipient_uei": "U3", "recipient_name": "TCCU",
         "recipient_state_code": "ND", "business_types_code": "M,U"},
        {"recipient_uei": "U4", "recipient_name": "ANNHSI",
         "recipient_state_code": "AK", "business_types_code": "M,V"},
    ])
    per = aggregate_business_types_per_recipient(txn)
    _, stats = apply_filter(per)
    assert stats.in_scope_recipients == 0
    assert stats.excluded_by_disqualifying_cotag == 4


def test_m_rule_no_M_excluded():
    txn = pd.DataFrame([
        {"recipient_uei": "U1", "recipient_name": "Govt Agency",
         "recipient_state_code": "NY", "business_types_code": "A"},
        {"recipient_uei": "U2", "recipient_name": "Non-501c3 Nonprofit",
         "recipient_state_code": "CA", "business_types_code": "N"},
    ])
    per = aggregate_business_types_per_recipient(txn)
    _, stats = apply_filter(per)
    assert stats.in_scope_recipients == 0
    assert stats.excluded_no_m_tag == 2


def test_classify_international_by_pop_country():
    # Note: this assumes the txn has already been filtered to in-scope
    # M-recipients before classify is called.
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of State",
         "awarding_sub_agency_name": "Bureau of Population, Refugees, and Migration",
         "assistance_listing_number": "19.518",
         "recipient_uei": "U1", "recipient_name": "Save The Children US",
         "recipient_state_code": "CT", "primary_place_of_performance_country_code": "KEN",
         "federal_action_obligation": 1_000_000},
    ])
    out, stats = classify(txn)
    assert stats.rows_international == 1
    assert "intl_pop" in out.iloc[0]["classification_rule_hits"]


def test_classify_hospital_by_name():
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of Health and Human Services",
         "awarding_sub_agency_name": "Centers for Disease Control",
         "assistance_listing_number": "93.000",
         "recipient_uei": "U1", "recipient_name": "Big City General Hospital",
         "recipient_state_code": "NY", "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 200_000},
    ])
    out, stats = classify(txn)
    assert stats.rows_hospital == 1
    assert "name_hospital" in out.iloc[0]["classification_rule_hits"]


def test_classify_hospital_excluded_by_foundation():
    """Hospital Foundation should NOT match — the exclusion regex vetoes."""
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of Health and Human Services",
         "awarding_sub_agency_name": "Centers for Disease Control",
         "assistance_listing_number": "93.000",
         "recipient_uei": "U1", "recipient_name": "Mercy Hospital Foundation",
         "recipient_state_code": "MO", "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 100_000},
    ])
    _, stats = classify(txn)
    assert stats.rows_hospital == 0


def test_classify_educational_charter_school():
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of Education",
         "awarding_sub_agency_name": "Office of Elementary",
         "assistance_listing_number": "84.282",
         "recipient_uei": "U1", "recipient_name": "Bronx Charter School Network",
         "recipient_state_code": "NY", "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 500_000},
    ])
    _, stats = classify(txn)
    assert stats.rows_educational == 1


def test_build_awards_table_filters_to_in_scope():
    """Regression: build_awards_table must filter to in-scope recipients.

    Originally build_awards_table did NOT filter, producing an awards table
    that contained the entire FY22-FY25 award population (not just 501(c)(3)
    recipients) - inflating Exhibit 2 outlay totals by ~40x.
    """
    from src.lightweight.tables import build_awards_table

    classified = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "X", "awarding_sub_agency_name": "Y",
         "assistance_listing_number": "84.001",
         "recipient_uei": "U1", "recipient_name": "In-Scope Org",
         "recipient_state_code": "NY", "recipient_category": "core",
         "recipient_subcategory": "",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 100, "total_outlayed_amount_for_overall_award": 80},
        {"transaction_id": "t2", "award_id_unique": "a2", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "X", "awarding_sub_agency_name": "Y",
         "assistance_listing_number": "84.001",
         "recipient_uei": "U2", "recipient_name": "Out-of-Scope Org",
         "recipient_state_code": "CA", "recipient_category": "core",
         "recipient_subcategory": "",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 999, "total_outlayed_amount_for_overall_award": 999},
    ])
    rfilter = pd.DataFrame([
        {"recipient_uei": "U1", "in_scope": True},
        {"recipient_uei": "U2", "in_scope": False},
    ])
    awards = build_awards_table(classified, recipient_filter=rfilter)
    assert len(awards) == 1
    assert awards.iloc[0]["recipient_uei"] == "U1"
    assert int(awards.iloc[0]["sum_obligation"]) == 100


def test_classify_priority_intl_over_hospital():
    """Hospital name with foreign POP -> International wins per priority hierarchy."""
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of State",
         "awarding_sub_agency_name": "Bureau of Population, Refugees, and Migration",
         "assistance_listing_number": "19.518",
         "recipient_uei": "U1", "recipient_name": "International Medical Center",
         "recipient_state_code": "DC", "primary_place_of_performance_country_code": "KEN",
         "federal_action_obligation": 1_000_000},
    ])
    _, stats = classify(txn)
    assert stats.rows_international == 1
    assert stats.rows_hospital == 0


# ---------------------------------------------------------------------------
# Test 4: parse_business_types edge cases
# ---------------------------------------------------------------------------

def test_parse_business_types_separator_variants():
    assert parse_business_types("M;X|E") == frozenset({"M", "X", "E"})
    assert parse_business_types("M\tX\nE") == frozenset({"M", "X", "E"})
    assert parse_business_types("  M  ,  X  ") == frozenset({"M", "X"})


def test_parse_business_types_concatenated_no_separator():
    assert parse_business_types("MNX") == frozenset({"M", "N", "X"})
    assert parse_business_types("mxe") == frozenset({"M", "X", "E"})


def test_parse_business_types_ignores_non_letters():
    assert parse_business_types("M,1,X") == frozenset({"M", "X"})
    assert parse_business_types("M, ?, X") == frozenset({"M", "X"})


def test_parse_business_types_empty_inputs():
    assert parse_business_types("") == frozenset()
    assert parse_business_types(None) == frozenset()
    assert parse_business_types("   ") == frozenset()
    assert parse_business_types(",,,") == frozenset()


# ---------------------------------------------------------------------------
# Test 5: FY boundary
# ---------------------------------------------------------------------------

def test_fy_boundary_sept_30_vs_oct_1():
    """Federal FY runs Oct 1 - Sep 30. Check the day-of-boundary correctness."""
    from src.analytic.tables import _fy_from_action_date

    s = pd.Series(["2021-10-01", "2022-09-30", "2022-10-01",
                   "2024-09-30", "2024-10-01", "2025-09-30"])
    fy = _fy_from_action_date(s)
    assert list(fy) == [2022, 2022, 2023, 2024, 2025, 2025]


# ---------------------------------------------------------------------------
# Test 6 + 7: International OR semantics, each rule fires alone
# ---------------------------------------------------------------------------

def test_international_or_fires_on_pop_alone():
    """POP non-USA -> International, even if subagency and listing are domestic."""
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of Education",
         "awarding_sub_agency_name": "Office of Postsecondary Education",
         "assistance_listing_number": "84.063",
         "recipient_uei": "U1", "recipient_name": "Some Org",
         "recipient_state_code": "CT", "primary_place_of_performance_country_code": "KEN",
         "federal_action_obligation": 100_000},
    ])
    _, stats = classify(txn)
    assert stats.rows_international == 1


def test_international_or_fires_on_subagency_alone():
    """USAID subagency -> International, even with domestic POP and domestic-looking listing."""
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Agency for International Development",
         "awarding_sub_agency_name": "United States Agency for International Development",
         "assistance_listing_number": "84.001",  # not a 19./98./85. prefix
         "recipient_uei": "U1", "recipient_name": "Some Org",
         "recipient_state_code": "CT", "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 100_000},
    ])
    _, stats = classify(txn)
    assert stats.rows_international == 1


def test_international_or_fires_on_listing_prefix_alone():
    """Listing prefix 19. (State Dept) -> International, even with domestic agency."""
    txn = pd.DataFrame([
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of Defense",
         "awarding_sub_agency_name": "U.S. Army",
         "assistance_listing_number": "19.518",
         "recipient_uei": "U1", "recipient_name": "Some Org",
         "recipient_state_code": "CT", "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 100_000},
    ])
    _, stats = classify(txn)
    assert stats.rows_international == 1


# ---------------------------------------------------------------------------
# Test 8: Hospital exclusion vetoes (multiple patterns + case insensitivity)
# ---------------------------------------------------------------------------

def test_hospital_exclusion_vetoes_multiple_patterns():
    """Foundation/Association/Auxiliary suffixes must veto a hospital match."""
    cases = [
        ("Mercy Hospital", True),
        ("MERCY HOSPITAL", True),  # case insensitivity
        ("mercy hospital", True),
        ("Mercy Hospital Foundation", False),  # foundation veto
        ("Hospital Foundation of New York", False),  # foundation veto
        ("HOSPITAL ASSOCIATION OF AMERICA", False),  # association veto
        ("New York Hospital Auxiliary", False),  # auxiliary veto
        ("Childrens Hospital", True),
        ("Children's Hospital", True),  # apostrophe variant
    ]
    rows = []
    for i, (name, _expected_hospital) in enumerate(cases):
        rows.append({
            "transaction_id": f"t{i}", "award_id_unique": f"a{i}",
            "action_date": "2024-05-01", "award_type_code": "04",
            "action_type": "A", "awarding_agency_name": "HHS",
            "awarding_sub_agency_name": "CDC", "assistance_listing_number": "93.000",
            "recipient_uei": f"U{i}", "recipient_name": name,
            "recipient_state_code": "NY", "primary_place_of_performance_country_code": "USA",
            "federal_action_obligation": 100_000,
        })
    classified, _ = classify(pd.DataFrame(rows))
    for i, (name, expected) in enumerate(cases):
        actual = classified.iloc[i]["recipient_category"] == "hospital"
        assert actual == expected, (
            f"Hospital flag for {name!r}: expected hospital={expected}, got hospital={actual}"
        )


# ---------------------------------------------------------------------------
# Test 9: Reconciliation graceful degradation
# ---------------------------------------------------------------------------

def test_reconciliation_handles_missing_bmf_outputs(tmp_path, monkeypatch):
    """When the BMF-backed analytic table doesn't exist, the reconciliation
    exhibit produces a placeholder CSV with a clear note instead of crashing."""
    from src.lightweight import reconcile
    from src import config as cfg

    # Point processed/ at a temp dir with only the lightweight parquet present.
    monkeypatch.setattr(cfg, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(reconcile.config, "PROCESSED", tmp_path / "processed")
    (tmp_path / "processed").mkdir()
    (tmp_path / "exhibits").mkdir()

    # Minimal lightweight parquet so the function gets past the first guard.
    lightweight = pd.DataFrame([
        {"fy": 2024, "transaction_id": "t1", "recipient_uei": "U1",
         "recipient_category": "core",
         "federal_action_obligation": 100.0,
         "federal_action_obligation_real": 105.0},
    ])
    lightweight.to_parquet(tmp_path / "processed" / "assistance_txn_501c3_lightweight.parquet",
                           index=False)
    # No BMF parquet in this directory.

    out = reconcile.produce(out_dir=tmp_path / "exhibits")
    assert out.exists()
    df = pd.read_csv(out)
    assert "note" in df.columns or len(df) > 0  # placeholder note or actual data
