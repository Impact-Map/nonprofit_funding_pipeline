"""Five-tier recipient matching (methodology Section 5).

Inputs:
  - distinct USAspending recipients seen in FY22-FY25 transactions
  - SAM entity extract (UEI -> EIN), if available
  - IRS BMF current release (filtered to SUBSECTION='03')
  - IRS BMF historical snapshots (Sep 2022..Sep 2025) for point-in-time eligibility

Output: a `recipient_match` parquet with one row per distinct USAspending
recipient. Each row records the resolved IRS EIN (if any), the match tier
(1=EIN, 2=det name+state, 3=prob name+state, 4=manual, 5=unresolved), the
match score, and the BMF NTEE/foundation fields needed by the classifier.

Tier 4 (manual) is implemented as a YAML override file at
reference_lists/manual_match_overrides.yaml; absent -> only top 200
unmatched are flagged for review.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from rapidfuzz import fuzz, process

from .. import config
from .normalize import normalize_ein, normalize_name, normalize_state, normalize_uei

LOG = logging.getLogger(__name__)


@dataclass
class MatchStats:
    total_usaspending_recipients: int
    tier1_ein: int
    tier2_det_name: int
    tier3_prob_name: int
    tier4_manual: int
    tier5_unresolved: int


def _distinct_recipients(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build the distinct recipient frame referenced as `usaspending_recipients`."""
    cols = [c for c in [
        "recipient_uei",
        "recipient_duns",
        "recipient_name",
        "recipient_state_code",
        "recipient_state",
        "recipient_ein",
    ] if c in transactions.columns]
    if "recipient_uei" not in transactions.columns:
        raise KeyError("transactions table missing recipient_uei")
    df = transactions[cols].copy()
    state_col = "recipient_state_code" if "recipient_state_code" in df.columns else "recipient_state"
    if state_col != "recipient_state":
        df = df.rename(columns={state_col: "recipient_state"})
    df = df.drop_duplicates(subset=["recipient_uei"]).reset_index(drop=True)
    df["recipient_uei"] = df["recipient_uei"].map(normalize_uei)
    df["recipient_state_norm"] = df["recipient_state"].map(normalize_state)
    df["recipient_name_norm"] = df["recipient_name"].map(normalize_name)
    if "recipient_ein" in df.columns:
        df["recipient_ein_norm"] = df["recipient_ein"].map(normalize_ein)
    else:
        df["recipient_ein_norm"] = ""
    return df


def _prepare_bmf(bmf: pd.DataFrame) -> pd.DataFrame:
    df = bmf.copy()
    df["EIN_norm"] = df["EIN"].map(normalize_ein)
    df["NAME_norm"] = df["NAME"].map(normalize_name)
    df["STATE_norm"] = df["STATE"].map(normalize_state)
    return df


def _prepare_sam(sam: pd.DataFrame | None) -> pd.DataFrame | None:
    if sam is None:
        return None
    df = sam.copy()
    if "UEI" not in df.columns or "TIN" not in df.columns:
        LOG.warning("SAM extract missing UEI/TIN; backfill disabled")
        return None
    df["UEI_norm"] = df["UEI"].map(normalize_uei)
    df["TIN_norm"] = df["TIN"].map(normalize_ein)
    df = df[df["UEI_norm"] != ""].drop_duplicates("UEI_norm")
    return df[["UEI_norm", "TIN_norm"]].rename(columns={"TIN_norm": "ein_from_sam"})


def _load_manual_overrides() -> pd.DataFrame:
    """Optional YAML file listing curated UEI -> EIN overrides (Tier 4)."""
    path = config.REFERENCE_LISTS / "manual_match_overrides.yaml"
    if not path.exists():
        return pd.DataFrame(columns=["recipient_uei", "irs_ein", "reviewer", "reviewed_at", "note"])
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = doc.get("overrides", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["recipient_uei"] = df["recipient_uei"].map(normalize_uei)
    df["irs_ein"] = df["irs_ein"].map(normalize_ein)
    return df


def build_recipient_match(transactions: pd.DataFrame,
                          bmf: pd.DataFrame,
                          sam: Optional[pd.DataFrame] = None,
                          tier3_threshold: float = 0.94,
                          tier4_top_n: int = config.TIER4_MANUAL_REVIEW_TOP_N
                          ) -> tuple[pd.DataFrame, MatchStats, pd.DataFrame]:
    """Run the five-tier match.

    Returns (match_table, stats, tier4_review_queue).
    """
    recipients = _distinct_recipients(transactions)
    bmf_p = _prepare_bmf(bmf)
    sam_p = _prepare_sam(sam)

    # ----- Tier 1: deterministic EIN -----
    out = recipients.copy()
    out["irs_ein"] = ""
    out["match_tier"] = 5
    out["match_score"] = 0.0
    out["bmf_name"] = ""
    out["bmf_state"] = ""
    out["bmf_ntee"] = ""
    out["bmf_foundation"] = ""

    # 1a. Direct EIN exposed on USAspending side
    direct_ein = out["recipient_ein_norm"].fillna("")
    bmf_by_ein = bmf_p.set_index("EIN_norm")
    have_direct = direct_ein.ne("") & direct_ein.isin(bmf_by_ein.index)
    out.loc[have_direct, "irs_ein"] = direct_ein[have_direct].values
    _join_bmf(out, bmf_by_ein, mask=have_direct, ein_col="recipient_ein_norm")
    out.loc[have_direct, "match_tier"] = 1
    out.loc[have_direct, "match_score"] = 1.0
    LOG.info("Tier 1 (direct EIN): %d", int(have_direct.sum()))

    # 1b. UEI -> SAM TIN -> BMF EIN
    if sam_p is not None:
        unresolved = out["match_tier"] == 5
        if unresolved.any():
            joined = out.loc[unresolved, ["recipient_uei"]].merge(
                sam_p, left_on="recipient_uei", right_on="UEI_norm", how="left"
            )
            sam_ein = joined["ein_from_sam"].fillna("")
            sam_hit = sam_ein.ne("") & sam_ein.isin(bmf_by_ein.index)
            tier1b_idx = out.loc[unresolved].index[sam_hit.values]
            out.loc[tier1b_idx, "irs_ein"] = sam_ein.values[sam_hit.values]
            for col_src, col_dst in [
                ("NAME", "bmf_name"), ("STATE_norm", "bmf_state"),
                ("NTEE_CD", "bmf_ntee"), ("FOUNDATION", "bmf_foundation"),
            ]:
                if col_src in bmf_by_ein.columns:
                    out.loc[tier1b_idx, col_dst] = (
                        bmf_by_ein.loc[out.loc[tier1b_idx, "irs_ein"].values, col_src].values
                    )
            out.loc[tier1b_idx, "match_tier"] = 1
            out.loc[tier1b_idx, "match_score"] = 1.0
            LOG.info("Tier 1 (UEI->SAM->EIN): %d", len(tier1b_idx))

    # ----- Tier 2: deterministic name+state -----
    bmf_by_name_state = bmf_p.set_index(["NAME_norm", "STATE_norm"])
    if not bmf_by_name_state.index.is_unique:
        # Collapse duplicates: keep the first row per (name, state). Document by
        # adding a count so the manual reviewer can see when ambiguity exists.
        bmf_by_name_state = bmf_p.drop_duplicates(["NAME_norm", "STATE_norm"]).set_index(
            ["NAME_norm", "STATE_norm"]
        )
    unresolved = out["match_tier"] == 5
    if unresolved.any():
        keys = list(zip(
            out.loc[unresolved, "recipient_name_norm"].values,
            out.loc[unresolved, "recipient_state_norm"].values,
        ))
        # Membership check needs a set since MultiIndex .isin is column-wise.
        bmf_keys = set(map(tuple, bmf_by_name_state.index.tolist()))
        hit_mask = pd.Series(
            [(k[0] != "" and k[1] != "" and k in bmf_keys) for k in keys],
            index=out.loc[unresolved].index,
        )
        tier2_idx = hit_mask.index[hit_mask.values]
        if len(tier2_idx):
            sub = out.loc[tier2_idx, ["recipient_name_norm", "recipient_state_norm"]]
            joined = bmf_by_name_state.loc[
                list(zip(sub["recipient_name_norm"], sub["recipient_state_norm"]))
            ].reset_index(drop=True)
            out.loc[tier2_idx, "irs_ein"] = joined["EIN_norm"].values
            out.loc[tier2_idx, "bmf_name"] = joined.get("NAME", pd.Series([""] * len(joined))).values
            out.loc[tier2_idx, "bmf_state"] = joined["STATE_norm"].values
            if "NTEE_CD" in joined.columns:
                out.loc[tier2_idx, "bmf_ntee"] = joined["NTEE_CD"].fillna("").values
            if "FOUNDATION" in joined.columns:
                out.loc[tier2_idx, "bmf_foundation"] = joined["FOUNDATION"].fillna("").values
            out.loc[tier2_idx, "match_tier"] = 2
            out.loc[tier2_idx, "match_score"] = 1.0
            LOG.info("Tier 2 (det name+state): %d", len(tier2_idx))

    # ----- Tier 3: probabilistic name+state via Jaro-Winkler -----
    # Build per-state BMF candidate lists once.
    unresolved = out["match_tier"] == 5
    if unresolved.any():
        bmf_by_state: dict[str, pd.DataFrame] = {
            s: g.reset_index(drop=True)
            for s, g in bmf_p.groupby("STATE_norm")
            if isinstance(s, str) and s
        }
        score_threshold_pct = tier3_threshold * 100  # rapidfuzz returns 0-100
        tier3_assigned = 0
        for idx in out.index[unresolved]:
            state = out.at[idx, "recipient_state_norm"]
            name = out.at[idx, "recipient_name_norm"]
            if not state or not name or state not in bmf_by_state:
                continue
            cands = bmf_by_state[state]
            if cands.empty:
                continue
            best = process.extractOne(
                name, cands["NAME_norm"].tolist(), scorer=fuzz.token_set_ratio,
                score_cutoff=score_threshold_pct,
            )
            if best is None:
                continue
            best_name, score, ridx = best
            row = cands.iloc[ridx]
            out.at[idx, "irs_ein"] = row["EIN_norm"]
            out.at[idx, "bmf_name"] = row.get("NAME", "")
            out.at[idx, "bmf_state"] = row["STATE_norm"]
            out.at[idx, "bmf_ntee"] = row.get("NTEE_CD", "") or ""
            out.at[idx, "bmf_foundation"] = row.get("FOUNDATION", "") or ""
            out.at[idx, "match_tier"] = 3
            out.at[idx, "match_score"] = score / 100.0
            tier3_assigned += 1
        LOG.info("Tier 3 (prob name+state, threshold=%.2f): %d", tier3_threshold, tier3_assigned)

    # ----- Tier 4: manual override file -----
    overrides = _load_manual_overrides()
    if not overrides.empty:
        ovr = out.merge(overrides, on="recipient_uei", how="left", suffixes=("", "_ovr"))
        manual_hit = ovr["irs_ein_ovr"].fillna("").ne("")
        if manual_hit.any():
            new_ein = ovr.loc[manual_hit, "irs_ein_ovr"].values
            target_idx = ovr.loc[manual_hit].index
            out.loc[target_idx, "irs_ein"] = new_ein
            # Pull BMF fields for the override EINs.
            ein_to_row = bmf_by_ein.loc[bmf_by_ein.index.intersection(set(new_ein))]
            ein_lookup = ein_to_row.to_dict(orient="index")
            for ein, ridx in zip(new_ein, target_idx):
                row = ein_lookup.get(ein, {})
                out.at[ridx, "bmf_name"] = row.get("NAME", "")
                out.at[ridx, "bmf_state"] = row.get("STATE_norm", "")
                out.at[ridx, "bmf_ntee"] = row.get("NTEE_CD", "") or ""
                out.at[ridx, "bmf_foundation"] = row.get("FOUNDATION", "") or ""
            out.loc[target_idx, "match_tier"] = 4
            out.loc[target_idx, "match_score"] = 1.0
            LOG.info("Tier 4 (manual override): %d", int(manual_hit.sum()))

    # ----- Tier 5: unresolved -> review queue (top N by obligated dollars) -----
    if "federal_action_obligation" in transactions.columns:
        obligated = (
            transactions.assign(
                _obl=lambda d: pd.to_numeric(d["federal_action_obligation"], errors="coerce").fillna(0)
            )
            .groupby("recipient_uei", as_index=False)["_obl"].sum()
            .rename(columns={"_obl": "total_obligated"})
        )
        obligated["recipient_uei"] = obligated["recipient_uei"].map(normalize_uei)
    else:
        obligated = pd.DataFrame({"recipient_uei": [], "total_obligated": []})

    review_queue = (
        out[out["match_tier"] == 5][["recipient_uei", "recipient_name", "recipient_state"]]
        .merge(obligated, on="recipient_uei", how="left")
        .fillna({"total_obligated": 0.0})
        .sort_values("total_obligated", ascending=False)
        .head(tier4_top_n)
        .reset_index(drop=True)
    )

    stats = MatchStats(
        total_usaspending_recipients=len(out),
        tier1_ein=int((out["match_tier"] == 1).sum()),
        tier2_det_name=int((out["match_tier"] == 2).sum()),
        tier3_prob_name=int((out["match_tier"] == 3).sum()),
        tier4_manual=int((out["match_tier"] == 4).sum()),
        tier5_unresolved=int((out["match_tier"] == 5).sum()),
    )

    keep_cols = [
        "recipient_uei", "recipient_name", "recipient_state",
        "irs_ein", "match_tier", "match_score",
        "bmf_name", "bmf_state", "bmf_ntee", "bmf_foundation",
    ]
    return out[keep_cols].copy(), stats, review_queue


def _join_bmf(out: pd.DataFrame, bmf_by_ein: pd.DataFrame, mask: pd.Series, ein_col: str) -> None:
    """In-place: pull BMF NAME/STATE/NTEE/FOUNDATION onto rows where mask is True."""
    if not mask.any():
        return
    eins = out.loc[mask, ein_col].values
    sub = bmf_by_ein.loc[bmf_by_ein.index.intersection(set(eins))]
    lookup = sub.to_dict(orient="index")
    for idx, ein in zip(out.loc[mask].index, eins):
        row = lookup.get(ein, {})
        out.at[idx, "bmf_name"] = row.get("NAME", "")
        out.at[idx, "bmf_state"] = row.get("STATE_norm", "")
        out.at[idx, "bmf_ntee"] = row.get("NTEE_CD", "") or ""
        out.at[idx, "bmf_foundation"] = row.get("FOUNDATION", "") or ""


def write_outputs(match_df: pd.DataFrame,
                  stats: MatchStats,
                  review_queue: pd.DataFrame,
                  out_dir: Path | None = None) -> tuple[Path, Path, Path]:
    out_dir = out_dir or config.PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    match_path = out_dir / "recipient_match.parquet"
    review_path = out_dir / "tier4_review_queue.parquet"
    stats_path = out_dir / "match_stats.json"
    match_df.to_parquet(match_path, index=False)
    review_queue.to_parquet(review_path, index=False)
    import json
    stats_path.write_text(json.dumps(stats.__dict__, indent=2))
    return match_path, review_path, stats_path


def coverage_report(match_df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Dollar-weighted match-rate by FY x agency, per Section 5.3."""
    txn = transactions.copy()
    txn["recipient_uei"] = txn["recipient_uei"].map(normalize_uei)
    txn["federal_action_obligation"] = pd.to_numeric(
        txn["federal_action_obligation"], errors="coerce"
    ).fillna(0.0)
    if "fy" not in txn.columns and "action_date" in txn.columns:
        txn["fy"] = _fy_from_action_date(txn["action_date"])
    j = txn.merge(match_df[["recipient_uei", "match_tier"]], on="recipient_uei", how="left")
    j["match_tier"] = j["match_tier"].fillna(5).astype(int)
    j["matched_dollars"] = j["federal_action_obligation"].where(j["match_tier"] < 5, 0.0)
    grp = j.groupby(["fy", "awarding_agency_name"], dropna=False, observed=False)
    rep = grp.agg(
        total_dollars=("federal_action_obligation", "sum"),
        matched_dollars=("matched_dollars", "sum"),
    ).reset_index()
    rep["match_rate"] = rep["matched_dollars"] / rep["total_dollars"].replace(0, pd.NA)
    rep["flag_under_90pct"] = rep["match_rate"] < 0.90
    return rep


def _fy_from_action_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    return ((d.dt.month >= 10).astype("Int64") + d.dt.year).astype("Int64")
