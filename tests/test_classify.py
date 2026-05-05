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
