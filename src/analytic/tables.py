"""Analytic-table assembly (methodology Step 6).

Combines the parquet outputs from earlier steps into the two persistent
analytic tables described in Section 11:

  - assistance_txn_501c3: one row per modification, restricted to 501(c)(3)
    recipients (match_tier in 1..4), classified into a panel, with the audit
    trail and the columns required by every downstream exhibit.

  - assistance_awards_501c3: one row per prime award, with first_action_date
    and cumulative outlay.

Real-dollar adjustments use a CPI-U deflator table (default) or the GDP
price deflator (sensitivity), both pinned to FY 2025.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..match.normalize import normalize_uei

LOG = logging.getLogger(__name__)


# Annual-average CPI-U (CUUR0000SA0). Update on each run from BLS.
# Values are FY-average (Oct prior year - Sep FY year). Indices below are
# defensible defaults until the run pulls fresh BLS data.
CPI_U_FY: dict[int, float] = {
    2022: 287.504,   # FY22 average
    2023: 301.836,   # FY23 average
    2024: 313.689,   # FY24 average
    2025: 322.420,   # FY25 average (preliminary)
}

GDP_DEFLATOR_FY: dict[int, float] = {
    2022: 117.612,
    2023: 122.273,
    2024: 125.482,
    2025: 128.367,
}


@dataclass
class AnalyticTablePaths:
    transactions: Path
    awards: Path


def _fy_from_action_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    fy = (d.dt.year + (d.dt.month >= 10).astype(int)).astype("Int64")
    return fy


def _real_dollars(nominal: pd.Series, fy: pd.Series, deflator: dict[int, float], base_fy: int) -> pd.Series:
    base = deflator[base_fy]
    factor = fy.map(lambda y: base / deflator.get(int(y), float("nan")) if pd.notna(y) else float("nan"))
    return nominal * factor.astype(float)


def build_transactions_table(transactions_classified: pd.DataFrame,
                             match_table: pd.DataFrame,
                             deflator: str = "CPI-U",
                             base_fy: int = 2025) -> pd.DataFrame:
    """Project the classified transactions to the schema in Section 11."""
    df = transactions_classified.copy()

    if "fy" not in df.columns:
        df["fy"] = _fy_from_action_date(df["action_date"])

    # Normalize keys
    df["recipient_uei"] = df["recipient_uei"].map(normalize_uei)

    # Restrict to recipients that resolved as 501(c)(3) (tiers 1-4).
    in_scope = df["match_tier"].notna() & (df["match_tier"] < 5)
    LOG.info("Transactions in 501(c)(3) scope: %d / %d", int(in_scope.sum()), len(df))
    df = df.loc[in_scope].copy()

    # Coerce numerics.
    df["federal_action_obligation"] = pd.to_numeric(
        df.get("federal_action_obligation", 0), errors="coerce"
    ).fillna(0.0)

    # Real-dollar series.
    deflator_table = CPI_U_FY if deflator.upper() == "CPI-U" else GDP_DEFLATOR_FY
    df["federal_action_obligation_real"] = _real_dollars(
        df["federal_action_obligation"], df["fy"], deflator_table, base_fy
    )

    # Columns from Section 11.
    column_specs: list[tuple[str, str | None]] = [
        ("fy", None),
        ("action_date", None),
        ("award_id_unique", None),
        ("transaction_id", None),
        ("award_type_code", None),
        ("action_type", None),
        ("awarding_agency_name", "awarding_agency"),
        ("awarding_sub_agency_name", "awarding_subagency"),
        ("assistance_listing_number", None),
        ("recipient_uei", None),
        ("irs_ein", "recipient_ein"),
        ("recipient_state_code", "recipient_state"),
        ("match_tier", None),
        ("bmf_foundation", None),
        ("bmf_ntee", "bmf_ntee_primary"),
        ("recipient_category", None),
        ("recipient_subcategory", None),
        ("classification_rule_hits", None),
        ("primary_place_of_performance_country_code", "place_of_performance_country"),
        ("ipeds_match", None),
        ("aha_match", None),
        ("hrsa_uds_match", None),
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

    # Constants
    out["bmf_subsection"] = "03"

    # Snapshot fields are filled in by the orchestrator from the manifest;
    # leave placeholders here so the schema is complete.
    if "snapshot_date" not in out.columns:
        out["snapshot_date"] = pd.NaT
    if "bmf_release_date" not in out.columns:
        out["bmf_release_date"] = pd.NaT
    if "classification_rules_version" not in out.columns:
        out["classification_rules_version"] = ""

    # Recipient state fallback
    if "recipient_state" in out.columns and out["recipient_state"].isna().all():
        if "recipient_state_name" in df.columns:
            out["recipient_state"] = df["recipient_state_name"]

    return out


def build_awards_table(transactions_classified: pd.DataFrame) -> pd.DataFrame:
    """Award-level table: first_action_date and cumulative outlay per Section 6/Step 6."""
    df = transactions_classified.copy()
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
        match_tier=("match_tier", "min"),  # best tier on the award
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
    txn_path = out_dir / "assistance_txn_501c3.parquet"
    awd_path = out_dir / "assistance_awards_501c3.parquet"
    transactions.to_parquet(txn_path, index=False)
    awards.to_parquet(awd_path, index=False)
    return AnalyticTablePaths(transactions=txn_path, awards=awd_path)
