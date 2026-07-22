"""End-to-end classifier test on a synthetic frame.

Exercises the priority hierarchy (international > hospital > educational > core)
and the Q-series dollar-floor rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify.categorize import classify


def _txn_frame() -> pd.DataFrame:
    return pd.DataFrame([
        # Domestic university - educational
        {
            "transaction_id": "t1", "award_id_unique": "a1", "action_date": "2023-05-01",
            "award_type_code": "04", "action_type": "A",
            "awarding_agency_name": "Department of Education",
            "awarding_sub_agency_name": "Office of Postsecondary Education",
            "assistance_listing_number": "84.063",
            "recipient_uei": "U01", "recipient_name": "Wisconsin State University",
            "recipient_state_code": "WI", "recipient_business_types": "06",
            "primary_place_of_performance_country_code": "USA",
            "federal_action_obligation": 100000,
        },
        # Domestic hospital - hospital
        {
            "transaction_id": "t2", "award_id_unique": "a2", "action_date": "2023-05-01",
            "award_type_code": "04", "action_type": "A",
            "awarding_agency_name": "Department of Health and Human Services",
            "awarding_sub_agency_name": "Health Resources and Services Administration",
            "assistance_listing_number": "93.224",
            "recipient_uei": "U02", "recipient_name": "Big City General Hospital",
            "recipient_state_code": "NY", "recipient_business_types": "26",
            "primary_place_of_performance_country_code": "USA",
            "federal_action_obligation": 200000,
        },
        # International by place of performance - international (overrides educational)
        {
            "transaction_id": "t3", "award_id_unique": "a3", "action_date": "2023-05-01",
            "award_type_code": "05", "action_type": "A",
            "awarding_agency_name": "Department of State",
            "awarding_sub_agency_name": "Bureau of Population, Refugees, and Migration",
            "assistance_listing_number": "19.518",
            "recipient_uei": "U03", "recipient_name": "Save The Children US",
            "recipient_state_code": "CT", "recipient_business_types": "M",
            "primary_place_of_performance_country_code": "KEN",
            "federal_action_obligation": 1000000,
        },
        # Q-series small dollar - falls through to core (below $100k floor)
        {
            "transaction_id": "t4", "award_id_unique": "a4", "action_date": "2023-05-01",
            "award_type_code": "04", "action_type": "A",
            "awarding_agency_name": "Department of Justice",
            "awarding_sub_agency_name": "Office of Justice Programs",
            "assistance_listing_number": "16.580",
            "recipient_uei": "U04", "recipient_name": "Tiny Foreign Affairs Org",
            "recipient_state_code": "DC", "recipient_business_types": "M",
            "primary_place_of_performance_country_code": "USA",
            "federal_action_obligation": 50000,
        },
    ])


def _match_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"recipient_uei": "U01", "irs_ein": "111111111", "match_tier": 1, "bmf_ntee": "B50",
         "bmf_foundation": "15", "bmf_name": "Wisconsin State Univ"},
        {"recipient_uei": "U02", "irs_ein": "222222222", "match_tier": 1, "bmf_ntee": "E22",
         "bmf_foundation": "15", "bmf_name": "Big City General Hospital"},
        {"recipient_uei": "U03", "irs_ein": "333333333", "match_tier": 1, "bmf_ntee": "Q33",
         "bmf_foundation": "15", "bmf_name": "Save The Children US"},
        {"recipient_uei": "U04", "irs_ein": "444444444", "match_tier": 1, "bmf_ntee": "Q99",
         "bmf_foundation": "15", "bmf_name": "Tiny Foreign Affairs Org"},
    ])


def test_priority_hierarchy_and_q_floor():
    txn = _txn_frame()
    rm = _match_frame()
    out, stats = classify(txn, rm)
    by_id = out.set_index("transaction_id")["recipient_category"].to_dict()
    assert by_id["t1"] == "educational"
    assert by_id["t2"] == "hospital"
    assert by_id["t3"] == "international"   # POP foreign overrides educational/hospital
    assert by_id["t4"] == "core"            # Q-series but $50k < $100k floor
    assert stats.rows_total == 4


def test_edu_biz_rule_accepts_letter_codes_from_award_data_archive():
    """The Award Data Archive uses letter business_types_code (H, O, T, U,
    S, V, G), not the numeric SF-424 codes (06, 11, 12, 23) that the
    bulk_download API uses. Both encodings must fire the educational rule.

    Regression: Johns Hopkins ($4.7B, tagged mostly 'O') and UNC Chapel
    Hill ($2.9B, tagged mostly 'H') were entirely mis-classified as
    Topline_Core because the rule only checked numeric codes.
    """
    import pandas as pd
    from src.classify.categorize import classify

    txn = pd.DataFrame([
        # Letter-code Private IHE (Johns Hopkins / Duke / any private university)
        {"transaction_id": "t1", "award_id_unique": "a1", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "NIH", "awarding_sub_agency_name": "NIAID",
         "assistance_listing_number": "93.855",
         "recipient_uei": "U1", "recipient_name": "THE PRIVATE UNIVERSITY",
         "recipient_state_code": "MD", "recipient_business_types": "O",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 500_000},
        # Letter-code Public IHE (UNC / Michigan / any state university)
        {"transaction_id": "t2", "award_id_unique": "a2", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "NSF", "awarding_sub_agency_name": "NSF",
         "assistance_listing_number": "47.049",
         "recipient_uei": "U2", "recipient_name": "STATE UNIVERSITY",
         "recipient_state_code": "NC", "recipient_business_types": "H",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 400_000},
        # Concatenated letter codes ("OMX" — private IHE + 501c3 + Other)
        {"transaction_id": "t3", "award_id_unique": "a3", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "DOE", "awarding_sub_agency_name": "DOE",
         "assistance_listing_number": "81.049",
         "recipient_uei": "U3", "recipient_name": "MIXED TAG UNIVERSITY",
         "recipient_state_code": "MA", "recipient_business_types": "OMX",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 300_000},
        # Legacy numeric code ("06" = State IHE from bulk_download API)
        {"transaction_id": "t4", "award_id_unique": "a4", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "DOE", "awarding_sub_agency_name": "DOE",
         "assistance_listing_number": "81.049",
         "recipient_uei": "U4", "recipient_name": "LEGACY-NUMERIC UNIVERSITY",
         "recipient_state_code": "WI", "recipient_business_types": "06",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 200_000},
        # Independent school district (letter G)
        {"transaction_id": "t5", "award_id_unique": "a5", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "Department of Education", "awarding_sub_agency_name": "OESE",
         "assistance_listing_number": "84.001",
         "recipient_uei": "U5", "recipient_name": "UNIFIED SCHOOL DISTRICT",
         "recipient_state_code": "CA", "recipient_business_types": "G",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 100_000},
        # NEGATIVE case: only M+X (no educational letter) - must NOT be educational
        {"transaction_id": "t6", "award_id_unique": "a6", "action_date": "2024-05-01",
         "award_type_code": "04", "action_type": "A",
         "awarding_agency_name": "HHS", "awarding_sub_agency_name": "SAMHSA",
         "assistance_listing_number": "93.958",
         "recipient_uei": "U6", "recipient_name": "SOME 501C3",
         "recipient_state_code": "NY", "recipient_business_types": "MX",
         "primary_place_of_performance_country_code": "USA",
         "federal_action_obligation": 50_000},
    ])
    # Minimal match table so bmf_ntee is empty (forcing the biz rule to be
    # the deciding signal, mirroring the JHU / UNC real case).
    match = pd.DataFrame([
        {"recipient_uei": f"U{i}", "irs_ein": f"{i:09d}", "match_tier": 2,
         "bmf_ntee": "", "bmf_foundation": "15", "bmf_name": ""}
        for i in range(1, 7)
    ])
    out, stats = classify(txn, match)
    by_id = out.set_index("transaction_id")["recipient_category"].to_dict()
    assert by_id["t1"] == "educational", "Private IHE (O) should classify as educational"
    assert by_id["t2"] == "educational", "Public IHE (H) should classify as educational"
    assert by_id["t3"] == "educational", "OMX concatenated code should include O and classify educational"
    assert by_id["t4"] == "educational", "Legacy numeric code 06 should still work"
    assert by_id["t5"] == "educational", "Independent school district (G) should classify educational"
    assert by_id["t6"] == "core", "M+X (no educational letter) must NOT classify educational"
