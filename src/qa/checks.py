"""QA checks (methodology Step 8 + Section 5.3 / Section 8 caveats).

Three groups of checks:

1. Reconciliation against USAspending Spending Explorer aggregates
   (FY22-FY24; FY25 is in-flight). Difference >2% is escalated.

2. Top-25-recipient plausibility list per FY: produces a small CSV that a
   human reviewer is expected to sanity-check against Form 990s.

3. Diagnostic counters: row counts, dollar totals by panel, match-tier mix,
   COVID-program contribution, FY25 outlay-lag sensitivity.

Spending Explorer reference figures are pinned in `spending_explorer_ref.yaml`
under reference_lists/. The yaml is empty by default and is populated by the
operator from the public dashboards on each run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import config

LOG = logging.getLogger(__name__)


@dataclass
class QAReport:
    panel_summary: pd.DataFrame
    match_tier_mix: pd.DataFrame
    covid_contribution: pd.DataFrame
    fy25_outlay_floor: pd.DataFrame
    top25_recipients_per_fy: pd.DataFrame
    spending_explorer_recon: pd.DataFrame


def _load_spending_explorer_ref() -> pd.DataFrame:
    p = config.REFERENCE_LISTS / "spending_explorer_ref.yaml"
    if not p.exists():
        return pd.DataFrame(columns=["fy", "obligations_reference"])
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = doc.get("totals", [])
    return pd.DataFrame(rows)


def panel_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    g = transactions.groupby(["fy", "recipient_category"], dropna=False, observed=False)
    return g.agg(
        nominal=("federal_action_obligation", "sum"),
        real_fy25=("federal_action_obligation_real", "sum"),
        rows=("transaction_id", "count"),
        unique_recipients=("recipient_uei", lambda s: s.nunique()),
    ).reset_index()


def match_tier_mix(match_table: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    j = transactions.merge(match_table[["recipient_uei", "match_tier"]],
                           on="recipient_uei", how="left", suffixes=("", "_dup"))
    if "match_tier_dup" in j.columns:
        j["match_tier"] = j["match_tier"].fillna(j["match_tier_dup"])
    g = j.groupby(["fy", "match_tier"], dropna=False, observed=False)
    out = g.agg(
        rows=("transaction_id", "count"),
        nominal=("federal_action_obligation", "sum"),
    ).reset_index()
    return out


def covid_contribution(transactions: pd.DataFrame) -> pd.DataFrame:
    g = transactions.groupby(["fy", "recipient_category", "covid_flag"],
                             dropna=False, observed=False)
    out = g["federal_action_obligation"].sum().reset_index()
    return out


def fy25_outlay_floor(awards: pd.DataFrame) -> pd.DataFrame:
    if "cumulative_outlay" not in awards.columns:
        return pd.DataFrame(columns=["vintage_fy", "cumulative_outlay", "n_awards"])
    return awards.groupby("vintage_fy", dropna=False).agg(
        cumulative_outlay=("cumulative_outlay", "sum"),
        n_awards=("award_id_unique", "count"),
    ).reset_index()


def top25_recipients_per_fy(transactions: pd.DataFrame) -> pd.DataFrame:
    g = transactions.groupby(["fy", "recipient_uei", "recipient_category"], dropna=False, observed=False)
    obl = g["federal_action_obligation"].sum().reset_index()
    obl["rank"] = obl.groupby("fy")["federal_action_obligation"].rank(method="dense", ascending=False)
    return obl[obl["rank"] <= 25].sort_values(["fy", "rank"]).reset_index(drop=True)


def spending_explorer_recon(transactions: pd.DataFrame) -> pd.DataFrame:
    ref = _load_spending_explorer_ref()
    obs = transactions.groupby("fy", dropna=False)["federal_action_obligation"].sum().reset_index()
    obs = obs.rename(columns={"federal_action_obligation": "obligations_observed"})
    if ref.empty:
        obs["obligations_reference"] = pd.NA
        obs["pct_diff"] = pd.NA
        obs["over_2pct_flag"] = pd.NA
        return obs
    rep = obs.merge(ref, on="fy", how="left")
    rep["pct_diff"] = (rep["obligations_observed"] - rep["obligations_reference"]) / rep["obligations_reference"]
    rep["over_2pct_flag"] = rep["pct_diff"].abs() > 0.02
    return rep


def run(transactions: pd.DataFrame, awards: pd.DataFrame, match_table: pd.DataFrame,
        out_dir: Path | None = None) -> QAReport:
    out_dir = out_dir or (config.EXHIBITS / "qa")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = QAReport(
        panel_summary=panel_summary(transactions),
        match_tier_mix=match_tier_mix(match_table, transactions),
        covid_contribution=covid_contribution(transactions),
        fy25_outlay_floor=fy25_outlay_floor(awards),
        top25_recipients_per_fy=top25_recipients_per_fy(transactions),
        spending_explorer_recon=spending_explorer_recon(transactions),
    )
    rep.panel_summary.to_csv(out_dir / "qa_panel_summary.csv", index=False)
    rep.match_tier_mix.to_csv(out_dir / "qa_match_tier_mix.csv", index=False)
    rep.covid_contribution.to_csv(out_dir / "qa_covid_contribution.csv", index=False)
    rep.fy25_outlay_floor.to_csv(out_dir / "qa_fy25_outlay_floor.csv", index=False)
    rep.top25_recipients_per_fy.to_csv(out_dir / "qa_top25_recipients_per_fy.csv", index=False)
    rep.spending_explorer_recon.to_csv(out_dir / "qa_spending_explorer_recon.csv", index=False)

    LOG.info("QA outputs written to %s", out_dir)
    return rep
