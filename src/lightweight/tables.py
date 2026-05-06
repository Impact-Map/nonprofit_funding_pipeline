"""Phase 1 lightweight analytic-table assembly.

Mirrors src/analytic/tables.py but adapted for the lightweight schema:
  - No bmf_ntee_primary, no bmf_foundation, no irs_ein.
  - business_types_set replaces NTEE-derived recipient typing.
  - in_scope filter is `match_tier` style on the recipient_filter table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config
from ..analytic.tables import (
    CPI_U_FY, GDP_DEFLATOR_FY, _fy_from_action_date, _real_dollars,
)
from ..match.normalize import normalize_uei

LOG = logging.getLogger(__name__)


@dataclass
class AnalyticTablePaths:
    transactions: Path
    awards: Path


def build_transactions_table(transactions_classified: pd.DataFrame,
                             recipient_filter: pd.DataFrame,
                             deflator: str = "CPI-U",
                             base_fy: int = 2025) -> pd.DataFrame:
    """Project the lightweight classified transactions to the methodology
    Section 11 (lightweight) schema."""
    df = transactions_classified.copy()
    if "fy" not in df.columns:
        df["fy"] = _fy_from_action_date(df["action_date"])

    df["recipient_uei"] = df["recipient_uei"].map(normalize_uei)

    # Filter to in-scope recipients only.
    rf = recipient_filter[["recipient_uei", "in_scope", "bt_set"]].copy()
    rf["recipient_uei"] = rf["recipient_uei"].map(normalize_uei)
    df = df.merge(rf, on="recipient_uei", how="left")
    in_scope = df["in_scope"].fillna(False)
    LOG.info("Transactions in lightweight 501(c)(3) scope: %d / %d",
             int(in_scope.sum()), len(df))
    df = df.loc[in_scope].copy()

    df["federal_action_obligation"] = pd.to_numeric(
        df.get("federal_action_obligation", 0), errors="coerce"
    ).fillna(0.0)
    deflator_table = CPI_U_FY if deflator.upper() == "CPI-U" else GDP_DEFLATOR_FY
    df["federal_action_obligation_real"] = _real_dollars(
        df["federal_action_obligation"], df["fy"], deflator_table, base_fy
    )

    column_specs: list[tuple[str, str | None]] = [
        # Identity / dates
        ("fy", None),
        ("action_date", None),
        ("award_id_unique", None),
        ("transaction_id", None),
        ("award_type_code", None),
        ("action_type", None),
        # Awarding
        ("awarding_agency_name", "awarding_agency"),
        ("awarding_sub_agency_name", "awarding_subagency"),
        ("assistance_listing_number", None),
        # Recipient identity
        ("recipient_uei", None),
        ("recipient_name", None),
        ("bt_set", "business_types_set"),
        # Recipient geography (for mapping)
        ("recipient_country_code", "recipient_country"),
        ("recipient_country_name", None),
        ("recipient_state_code", "recipient_state"),
        ("recipient_state_name", None),
        ("recipient_county_name", None),
        ("prime_award_transaction_recipient_county_fips_code", "recipient_county_fips"),
        ("recipient_city_name", "recipient_city"),
        ("recipient_zip_code", "recipient_zip"),
        ("prime_award_transaction_recipient_cd_current", "recipient_cd"),
        # Classification
        ("recipient_category", None),
        ("recipient_subcategory", None),
        ("classification_rule_hits", None),
        # Place of performance geography (for International maps + state cuts)
        ("primary_place_of_performance_country_code", "place_of_performance_country"),
        ("primary_place_of_performance_country_name", "place_of_performance_country_name"),
        ("primary_place_of_performance_state_name", "place_of_performance_state_name"),
        ("prime_award_transaction_place_of_performance_state_fips_code", "place_of_performance_state_fips"),
        ("primary_place_of_performance_county_name", "place_of_performance_county_name"),
        ("prime_award_transaction_place_of_performance_county_fips_code", "place_of_performance_county_fips"),
        ("primary_place_of_performance_city_name", "place_of_performance_city"),
        ("primary_place_of_performance_zip_4", "place_of_performance_zip"),
        ("prime_award_transaction_place_of_performance_cd_current", "place_of_performance_cd"),
        # Money
        ("federal_action_obligation", None),
        ("federal_action_obligation_real", None),
        ("covid_flag", None),
    ]
    out = pd.DataFrame(index=df.index)
    for src, dest in column_specs:
        if src in df.columns:
            out[dest or src] = df[src]
        elif (dest or src) not in out.columns:
            out[dest or src] = pd.Series([pd.NA] * len(df), index=df.index)

    if "snapshot_date" not in out.columns:
        out["snapshot_date"] = pd.NaT
    if "classification_rules_version" not in out.columns:
        out["classification_rules_version"] = ""

    return out


def build_awards_table(transactions_classified: pd.DataFrame,
                       recipient_filter: pd.DataFrame | None = None) -> pd.DataFrame:
    """Award-level table with first_action_date and cumulative outlay.

    Filters to in-scope recipients (M-with-exclusions per the recipient_filter
    table) before aggregating, mirroring build_transactions_table. Without
    this filter the awards table reflects the entire FY22-FY25 award
    population rather than just lightweight-501(c)(3) recipients - producing
    wildly inflated outlay totals in Exhibit 2.
    """
    df = transactions_classified.copy()
    if recipient_filter is not None:
        rf = recipient_filter[recipient_filter["in_scope"]]
        in_scope_uei = set(rf["recipient_uei"].map(normalize_uei))
        df["recipient_uei"] = df["recipient_uei"].map(normalize_uei)
        df = df[df["recipient_uei"].isin(in_scope_uei)].copy()
        LOG.info("Awards (lightweight): %d transactions in M-with-exclusions scope", len(df))
    if "award_id_unique" not in df.columns:
        df["award_id_unique"] = df.get("generated_unique_award_id", df.index.astype(str))

    df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
    df["recipient_uei"] = df["recipient_uei"].map(normalize_uei)
    df["federal_action_obligation"] = pd.to_numeric(
        df.get("federal_action_obligation", 0), errors="coerce"
    ).fillna(0.0)

    grp = df.groupby("award_id_unique", dropna=False)
    awards = grp.agg(
        first_action_date=("action_date", "min"),
        last_action_date=("action_date", "max"),
        recipient_uei=("recipient_uei", "first"),
        recipient_name=("recipient_name", "first"),
        awarding_agency_name=("awarding_agency_name", "first"),
        awarding_sub_agency_name=("awarding_sub_agency_name", "first"),
        assistance_listing_number=("assistance_listing_number", "first"),
        award_type_code=("award_type_code", "first"),
        recipient_category=("recipient_category", "first"),
        recipient_subcategory=("recipient_subcategory", "first"),
        sum_obligation=("federal_action_obligation", "sum"),
    ).reset_index()
    awards["vintage_fy"] = _fy_from_action_date(awards["first_action_date"])

    if "total_outlayed_amount_for_overall_award" in df.columns:
        outlay = (
            df.dropna(subset=["total_outlayed_amount_for_overall_award"])
            .assign(_outlay=lambda d: pd.to_numeric(
                d["total_outlayed_amount_for_overall_award"], errors="coerce"
            ))
            .groupby("award_id_unique")["_outlay"].max()
            .rename("cumulative_outlay")
        )
        awards = awards.merge(outlay, on="award_id_unique", how="left")
    else:
        awards["cumulative_outlay"] = float("nan")

    return awards


def write_outputs(transactions: pd.DataFrame, awards: pd.DataFrame,
                  out_dir: Path | None = None) -> AnalyticTablePaths:
    out_dir = out_dir or config.PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    txn_path = out_dir / "assistance_txn_501c3_lightweight.parquet"
    awd_path = out_dir / "assistance_awards_501c3_lightweight.parquet"
    # bt_set may have been a frozenset on input; normalize to sorted str.
    if "business_types_set" in transactions.columns:
        transactions = transactions.copy()
        transactions["business_types_set"] = transactions["business_types_set"].apply(
            lambda s: "".join(sorted(s)) if not isinstance(s, str) and s is not None else s
        )
    transactions.to_parquet(txn_path, index=False)
    awards.to_parquet(awd_path, index=False)
    return AnalyticTablePaths(transactions=txn_path, awards=awd_path)
