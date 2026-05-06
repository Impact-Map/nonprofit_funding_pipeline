"""Aggregations and exhibits (methodology Step 7).

One function per exhibit (1-14) plus a `produce_all` orchestrator that runs
the full set against the analytic transactions table and writes CSV/parquet
outputs to /exhibits.

Each exhibit is produced four times (Topline/Core, Educational, Hospital,
International) and once as a combined Total. Sub-category exhibits run once
per panel where applicable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config

LOG = logging.getLogger(__name__)

PANELS = ("core", "educational", "hospital", "international")
PANEL_LABEL = {
    "core": "Topline_Core",
    "educational": "Educational",
    "hospital": "Hospital",
    "international": "International",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panel_view(txn: pd.DataFrame, panel: str | None) -> pd.DataFrame:
    return txn if panel is None else txn[txn["recipient_category"] == panel]


def _write(df: pd.DataFrame, exhibit_name: str, panel: str | None,
           out_dir: Path) -> Path:
    safe = exhibit_name.replace(" ", "_").lower()
    panel_tag = PANEL_LABEL.get(panel, "Total") if panel else "Total"
    path = out_dir / f"{safe}__{panel_tag}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Exhibit 1: total obligations and unique-recipient counts by FY
# ---------------------------------------------------------------------------


def exhibit_1_obligations_recipients_by_fy(txn: pd.DataFrame) -> pd.DataFrame:
    g = txn.groupby("fy", dropna=False)
    out = g.agg(
        nominal_obligations=("federal_action_obligation", "sum"),
        real_obligations_fy25=("federal_action_obligation_real", "sum"),
        unique_recipients=("recipient_uei", lambda s: s.nunique()),
        transaction_count=("transaction_id", "count"),
    ).reset_index()
    return out


# Exhibit 2: cumulative outlays by FY of award vintage ----------------------

def exhibit_2_outlays_by_vintage_fy(awards: pd.DataFrame) -> pd.DataFrame:
    if "cumulative_outlay" not in awards.columns:
        return pd.DataFrame(columns=["vintage_fy", "cumulative_outlay", "n_awards"])
    g = awards.groupby("vintage_fy", dropna=False)
    return g.agg(
        cumulative_outlay=("cumulative_outlay", "sum"),
        n_awards=("award_id_unique", "count"),
    ).reset_index()


# Exhibit 3 / 4: mix shift by award_type and action_type --------------------

def exhibit_3_mix_by_award_type(txn: pd.DataFrame) -> pd.DataFrame:
    return _pivot_share(txn, "fy", "award_type_code")


def exhibit_4_mix_by_action_type(txn: pd.DataFrame) -> pd.DataFrame:
    return _pivot_share(txn, "fy", "action_type")


def _pivot_share(txn: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    g = txn.groupby([index, columns], dropna=False, observed=False)["federal_action_obligation"].sum()
    pivot = g.unstack(fill_value=0.0)
    totals = pivot.sum(axis=1).replace(0, np.nan)
    share = pivot.div(totals, axis=0).fillna(0.0)
    share.columns = [f"{c}_share" for c in share.columns]
    return pivot.join(share).reset_index()


# Exhibit 5 / 6: top 25 funding agencies / CFDA programs --------------------

def exhibit_5_top_agencies(txn: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    # Column name in the analytic table is `awarding_agency` per Section 11
    # schema (stripped of the `_name` suffix during build_transactions_table).
    return _top_n_with_rank(txn, "awarding_agency", top_n)


def exhibit_6_top_listings(txn: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    return _top_n_with_rank(txn, "assistance_listing_number", top_n)


def _top_n_with_rank(txn: pd.DataFrame, dim: str, top_n: int) -> pd.DataFrame:
    g = txn.groupby(["fy", dim], dropna=False, observed=False)["federal_action_obligation"].sum().reset_index()
    g = g.dropna(subset=[dim])
    g["rank"] = g.groupby("fy")["federal_action_obligation"].rank(method="dense", ascending=False)
    return g[g["rank"] <= top_n].sort_values(["fy", "rank"]).reset_index(drop=True)


# Exhibit 7: state-level concentration --------------------------------------

def exhibit_7_state_concentration(txn: pd.DataFrame) -> pd.DataFrame:
    if "recipient_state" not in txn.columns:
        return pd.DataFrame()
    g = txn.groupby(["fy", "recipient_state"], dropna=False, observed=False)["federal_action_obligation"].sum().reset_index()
    return g.sort_values(["fy", "federal_action_obligation"], ascending=[True, False]).reset_index(drop=True)


# Exhibit 8: distribution of award size by FY -------------------------------

def exhibit_8_award_size_distribution(awards: pd.DataFrame) -> pd.DataFrame:
    g = awards.groupby("vintage_fy", dropna=False)["sum_obligation"]
    quantiles = g.quantile([0.25, 0.5, 0.75, 0.9, 0.99]).unstack()
    quantiles.columns = [f"P{int(q*100)}" for q in quantiles.columns]
    out = quantiles.reset_index()
    counts = g.agg(["count", "mean", "sum"]).reset_index()
    return out.merge(counts, on="vintage_fy")


# Exhibit 9: COVID contribution to YoY change -------------------------------

def exhibit_9_covid_contribution(txn: pd.DataFrame) -> pd.DataFrame:
    g = txn.groupby(["fy", "covid_flag"], dropna=False, observed=False)["federal_action_obligation"].sum()
    pivot = g.unstack(fill_value=0.0).reset_index()
    pivot.columns = ["fy"] + [f"covid={c}" for c in pivot.columns[1:]]
    return pivot


# Exhibit 10: shift-share decomposition of FY25 vs FY22 ---------------------

def exhibit_10_shift_share_fy25_vs_fy22(txn: pd.DataFrame,
                                        dim_a: str = "awarding_agency",
                                        dim_b: str = "assistance_listing_number") -> pd.DataFrame:
    """Within / between / interaction decomposition by (agency, program).

    Total change = sum_{a,b} (s22_ab + s25_ab) * (v25_ab - v22_ab)/2 etc.
    Implemented as the standard within/mix/interaction formulation:
      DT = sum_g (s_g_22 * dV_g) + sum_g (V_g_22 * ds_g) + interaction
    where g is the (agency, program) cell.
    """
    f = txn[txn["fy"].isin([2022, 2025])].copy()
    if f.empty:
        return pd.DataFrame()
    cell = (
        f.groupby(["fy", dim_a, dim_b], dropna=False, observed=False)["federal_action_obligation"]
        .sum().reset_index()
    )
    p = cell.pivot_table(
        index=[dim_a, dim_b], columns="fy", values="federal_action_obligation", fill_value=0
    ).reset_index()
    if 2022 not in p.columns or 2025 not in p.columns:
        return pd.DataFrame()
    p = p.rename(columns={2022: "v22", 2025: "v25"})
    p["delta"] = p["v25"] - p["v22"]
    total22 = p["v22"].sum() or np.nan
    total25 = p["v25"].sum() or np.nan
    p["share22"] = p["v22"] / total22
    p["share25"] = p["v25"] / total25
    # within-cell change (mean share weighted)
    p["within"] = ((p["share22"] + p["share25"]) / 2.0) * (p["v25"] - p["v22"])
    # mix shift component
    p["mix"] = ((p["v22"] + p["v25"]) / 2.0) * (p["share25"] - p["share22"])
    # residual interaction
    p["residual"] = p["delta"] - p["within"] - p["mix"]
    return p.sort_values("delta", ascending=False).reset_index(drop=True)


# Exhibit 11: Educational sub-category breakdown ----------------------------

def exhibit_11_educational_subcats(txn: pd.DataFrame) -> pd.DataFrame:
    sub = txn[txn["recipient_category"] == "educational"]
    g = sub.groupby(["fy", "recipient_subcategory"], dropna=False, observed=False)["federal_action_obligation"].sum().reset_index()
    return g


# Exhibit 12: Hospital sub-category breakdown -------------------------------

def exhibit_12_hospital_subcats(txn: pd.DataFrame) -> pd.DataFrame:
    sub = txn[txn["recipient_category"] == "hospital"]
    g = sub.groupby(["fy", "recipient_subcategory"], dropna=False, observed=False)["federal_action_obligation"].sum().reset_index()
    return g


# Exhibit 13: International sub-category breakdown --------------------------

def exhibit_13_international_subcats(txn: pd.DataFrame) -> pd.DataFrame:
    sub = txn[txn["recipient_category"] == "international"]
    g = sub.groupby(["fy", "recipient_subcategory"], dropna=False, observed=False)["federal_action_obligation"].sum().reset_index()
    return g


def exhibit_13b_international_country(txn: pd.DataFrame) -> pd.DataFrame:
    sub = txn[txn["recipient_category"] == "international"]
    if "place_of_performance_country" not in sub.columns:
        return pd.DataFrame()
    return sub.groupby(["fy", "place_of_performance_country"], dropna=False, observed=False)["federal_action_obligation"].sum().reset_index()


def exhibit_13c_international_us_vs_foreign_prime(txn: pd.DataFrame) -> pd.DataFrame:
    sub = txn[txn["recipient_category"] == "international"].copy()
    if sub.empty:
        return pd.DataFrame()
    state = sub.get("recipient_state", pd.Series([""] * len(sub)))
    sub["prime_origin"] = np.where(
        state.fillna("").astype(str).str.len().between(1, 2), "us_prime", "foreign_prime"
    )
    return sub.groupby(["fy", "prime_origin"], dropna=False, observed=False)["federal_action_obligation"].sum().reset_index()


# Exhibit 14: Cross-tab appendix --------------------------------------------

def exhibit_14_cross_tabs(txn: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # 14.1 - Recipient type within International (NTEE-derived bucketing).
    # The lightweight schema doesn't carry bmf_ntee_primary; in that case we
    # emit only an "other" bucket. The fallback Series must reuse the input
    # frame's index so boolean alignment in bucket.loc[...] works.
    intl = txn[txn["recipient_category"] == "international"].copy()
    if not intl.empty:
        if "bmf_ntee_primary" in intl.columns:
            ntee = intl["bmf_ntee_primary"].fillna("").astype(str).str.upper()
        else:
            ntee = pd.Series("", index=intl.index)
        bucket = pd.Series("other", index=intl.index)
        bucket.loc[ntee.str.startswith("B")] = "educational"
        bucket.loc[ntee.str.startswith("E")] = "hospital"
        bucket.loc[ntee.str.startswith("Q")] = "intl_ngo"
        bucket.loc[ntee.str.startswith("X")] = "faith_based"
        intl = intl.assign(_bucket=bucket)
        out["intl_recipient_type_mix"] = (
            intl.groupby(["fy", "_bucket"], dropna=False, observed=False)["federal_action_obligation"]
            .sum().reset_index().rename(columns={"_bucket": "recipient_type"})
        )

    # 14.2 - Funding character within Educational/Hospital panels (intl-flag share).
    for panel in ("educational", "hospital"):
        sub = txn[txn["recipient_category"] == panel].copy()
        if sub.empty:
            continue
        if "place_of_performance_country" in sub.columns:
            sub["intl_flag"] = (
                sub["place_of_performance_country"].fillna("").astype(str).str.upper().ne("USA")
                & sub["place_of_performance_country"].fillna("").astype(str).ne("")
            )
        else:
            sub["intl_flag"] = False
        out[f"{panel}_intl_share"] = (
            sub.groupby(["fy", "intl_flag"], dropna=False, observed=False)["federal_action_obligation"]
            .sum().reset_index()
        )

    # 14.3 - Recipient-type-only parallel view (ignore intl flag, classify on
    # recipient type alone using NTEE prefix and business types). Skipped
    # entirely when NTEE is unavailable (lightweight schema) - methodology
    # Section 7.1 of the lightweight document drops this cross-tab.
    if "bmf_ntee_primary" in txn.columns:
        base = txn.copy()
        ntee = base["bmf_ntee_primary"].fillna("").astype(str).str.upper()
        rec_type = pd.Series("core", index=base.index)
        rec_type.loc[ntee.str.startswith("B")] = "educational"
        rec_type.loc[(ntee >= "E20") & (ntee <= "E32")] = "hospital"
        rec_type.loc[ntee.str.startswith("Q")] = "intl_ngo"
        out["recipient_type_only"] = (
            base.assign(_rt=rec_type)
            .groupby(["fy", "_rt"], dropna=False, observed=False)["federal_action_obligation"]
            .sum().reset_index().rename(columns={"_rt": "recipient_type"})
        )

    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ExhibitArtifact:
    name: str
    panel: str | None
    path: Path


def produce_all(transactions: pd.DataFrame,
                awards: pd.DataFrame,
                out_dir: Path | None = None) -> list[ExhibitArtifact]:
    out_dir = out_dir or config.EXHIBITS
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ExhibitArtifact] = []

    # Per-panel exhibits 1, 3, 4, 5, 6, 7, 9
    panels: tuple[str | None, ...] = (None, *PANELS)
    for panel in panels:
        view = _panel_view(transactions, panel)
        if view.empty:
            continue
        artifacts.append(ExhibitArtifact("exhibit_01_obligations_recipients", panel,
                                         _write(exhibit_1_obligations_recipients_by_fy(view), "exhibit_01_obligations_recipients", panel, out_dir)))
        artifacts.append(ExhibitArtifact("exhibit_03_award_type_mix", panel,
                                         _write(exhibit_3_mix_by_award_type(view), "exhibit_03_award_type_mix", panel, out_dir)))
        artifacts.append(ExhibitArtifact("exhibit_04_action_type_mix", panel,
                                         _write(exhibit_4_mix_by_action_type(view), "exhibit_04_action_type_mix", panel, out_dir)))
        artifacts.append(ExhibitArtifact("exhibit_05_top_agencies", panel,
                                         _write(exhibit_5_top_agencies(view), "exhibit_05_top_agencies", panel, out_dir)))
        artifacts.append(ExhibitArtifact("exhibit_06_top_listings", panel,
                                         _write(exhibit_6_top_listings(view), "exhibit_06_top_listings", panel, out_dir)))
        artifacts.append(ExhibitArtifact("exhibit_07_state_concentration", panel,
                                         _write(exhibit_7_state_concentration(view), "exhibit_07_state_concentration", panel, out_dir)))
        artifacts.append(ExhibitArtifact("exhibit_09_covid_contribution", panel,
                                         _write(exhibit_9_covid_contribution(view), "exhibit_09_covid_contribution", panel, out_dir)))

    # Awards-driven exhibits 2 and 8 - per panel
    for panel in panels:
        a_view = awards if panel is None else awards[awards["recipient_category"] == panel]
        if a_view.empty:
            continue
        artifacts.append(ExhibitArtifact("exhibit_02_outlays_by_vintage", panel,
                                         _write(exhibit_2_outlays_by_vintage_fy(a_view), "exhibit_02_outlays_by_vintage", panel, out_dir)))
        artifacts.append(ExhibitArtifact("exhibit_08_award_size_distribution", panel,
                                         _write(exhibit_8_award_size_distribution(a_view), "exhibit_08_award_size_distribution", panel, out_dir)))

    # Single-shot exhibits
    artifacts.append(ExhibitArtifact("exhibit_10_shift_share_fy25_fy22", None,
                                     _write(exhibit_10_shift_share_fy25_vs_fy22(transactions), "exhibit_10_shift_share", None, out_dir)))
    artifacts.append(ExhibitArtifact("exhibit_11_educational_subcats", "educational",
                                     _write(exhibit_11_educational_subcats(transactions), "exhibit_11_educational_subcats", "educational", out_dir)))
    artifacts.append(ExhibitArtifact("exhibit_12_hospital_subcats", "hospital",
                                     _write(exhibit_12_hospital_subcats(transactions), "exhibit_12_hospital_subcats", "hospital", out_dir)))
    artifacts.append(ExhibitArtifact("exhibit_13_international_subcats", "international",
                                     _write(exhibit_13_international_subcats(transactions), "exhibit_13_international_subcats", "international", out_dir)))
    artifacts.append(ExhibitArtifact("exhibit_13b_international_country", "international",
                                     _write(exhibit_13b_international_country(transactions), "exhibit_13b_international_country", "international", out_dir)))
    artifacts.append(ExhibitArtifact("exhibit_13c_us_vs_foreign_prime", "international",
                                     _write(exhibit_13c_international_us_vs_foreign_prime(transactions), "exhibit_13c_us_vs_foreign_prime", "international", out_dir)))

    cross_tabs = exhibit_14_cross_tabs(transactions)
    for name, df in cross_tabs.items():
        artifacts.append(ExhibitArtifact(f"exhibit_14_{name}", None,
                                         _write(df, f"exhibit_14_{name}", None, out_dir)))

    LOG.info("Produced %d exhibit artifacts", len(artifacts))
    return artifacts
