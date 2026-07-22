"""Recipient identification under the M-with-exclusions rule.

Replaces the BMF-backed recipient_match step in the full methodology with a
single rule applied at the recipient grain (Section 3.2 of the lightweight
methodology):

  in_scope iff 'M' in business_types_set AND business_types_set is disjoint
  from the exclusion list.

The "business_types_set" for a recipient is the union of every
business_types_code value the recipient appears under across the FY22-FY25
transaction pool. This handles agencies that under-tag: if Agency A reports
the recipient as 'M' on one transaction and Agency B reports the same
recipient as 'M, X' on another, the union is {M, X} and the recipient is
disqualified (X is excluded).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .. import config
from ..match.normalize import normalize_uei
from ..refdata import load_reference_lists

LOG = logging.getLogger(__name__)


@dataclass
class FilterStats:
    total_distinct_recipients: int
    in_scope_recipients: int
    excluded_no_m_tag: int
    excluded_by_disqualifying_cotag: int
    cotag_distribution: dict[str, int]   # {'M only': N, 'M+E': N, 'M+K': N, ...}
    exclusion_reasons: dict[str, int]    # excluded code -> count


_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z]+")


def parse_business_types(raw: str | None) -> frozenset[str]:
    """Parse a business_types_code string into a set of single-letter codes.

    USAspending exposes business types as a single string field with the
    codes embedded in various formats: bare letters ("MNX"), comma-separated
    ("M,N,X"), comma+space ("M, N"), or mixed with descriptive text. The
    parser is permissive: split on any non-letter, uppercase, keep tokens
    that are exactly one A-Z character.
    """
    if not raw or not isinstance(raw, str):
        return frozenset()
    parts = _TOKEN_SPLIT_RE.split(raw.strip())
    out: set[str] = set()
    for p in parts:
        p = p.strip().upper()
        if len(p) == 1 and "A" <= p <= "Z":
            out.add(p)
        elif len(p) > 1:
            # Some sources concatenate codes without separators ("MNX"); split.
            for c in p:
                if c.isascii() and "A" <= c <= "Z":
                    out.add(c)
    return frozenset(out)


def aggregate_business_types_per_recipient(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build a per-recipient (UEI) view: name, state, union of business-types
    codes seen across all transactions."""
    if "recipient_uei" not in transactions.columns:
        raise KeyError("transactions table missing recipient_uei")
    if "business_types_code" not in transactions.columns:
        # Some bulk-download outputs use a different name; tolerate.
        for alt in ("recipient_business_types", "business_types_description"):
            if alt in transactions.columns:
                transactions = transactions.rename(columns={alt: "business_types_code"})
                break
        else:
            raise KeyError("transactions table missing business_types_code")

    df = transactions[
        ["recipient_uei", "recipient_name", "recipient_state_code", "business_types_code"]
    ].copy()
    df["recipient_uei"] = df["recipient_uei"].map(normalize_uei)
    df = df[df["recipient_uei"] != ""]
    df["bt_set"] = df["business_types_code"].map(parse_business_types)

    # Group by UEI; take first non-null name/state, union the bt sets.
    def _union(sets: Iterable[frozenset[str]]) -> frozenset[str]:
        out: set[str] = set()
        for s in sets:
            out |= s
        return frozenset(out)

    grouped = df.groupby("recipient_uei", as_index=False).agg(
        recipient_name=("recipient_name", "first"),
        recipient_state=("recipient_state_code", "first"),
        bt_set=("bt_set", _union),
    )
    return grouped


def apply_filter(per_recipient: pd.DataFrame) -> tuple[pd.DataFrame, FilterStats]:
    """Apply the M-with-exclusions rule. Returns (filter_table, stats).

    filter_table has one row per UEI with columns:
      recipient_uei, recipient_name, recipient_state, bt_set,
      in_scope (bool), exclusion_reason (str or '').
    """
    refs = load_reference_lists()
    rule = refs.business_types_lightweight
    required = rule["required_code"]
    excluded = set(rule["excluded_codes"].keys())
    # Soft codes (e.g., X = "Other") are tolerated when co-occurring with M.
    # They appear in the recipient's bt_set for transparency but do not
    # disqualify. See methodology Section 3.2 v2 note.
    soft = set((rule.get("soft_codes") or {}).keys())
    # Defensive: any code that ended up in both lists is treated as soft.
    excluded -= soft

    out = per_recipient.copy()

    has_m = out["bt_set"].apply(lambda s: required in s)
    disqualifying = out["bt_set"].apply(lambda s: bool(s & excluded))

    out["in_scope"] = has_m & ~disqualifying

    def _reason(s: frozenset[str]) -> str:
        if required not in s:
            return "no_M_tag"
        bad = sorted(s & excluded)
        if bad:
            return "disqualifying_cotag:" + "+".join(bad)
        return ""

    out["exclusion_reason"] = out["bt_set"].apply(_reason)

    # Stats
    total = len(out)
    in_scope = int(out["in_scope"].sum())
    no_m = int((~has_m).sum())
    bad_cotag = int((has_m & disqualifying).sum())

    cotag_dist: dict[str, int] = {}
    for s in out.loc[out["in_scope"], "bt_set"]:
        extras = sorted(c for c in s if c != required)
        key = "M_only" if not extras else "M+" + "+".join(extras)
        cotag_dist[key] = cotag_dist.get(key, 0) + 1

    excl_reasons: dict[str, int] = {}
    for r in out.loc[~out["in_scope"], "exclusion_reason"]:
        excl_reasons[r] = excl_reasons.get(r, 0) + 1

    stats = FilterStats(
        total_distinct_recipients=total,
        in_scope_recipients=in_scope,
        excluded_no_m_tag=no_m,
        excluded_by_disqualifying_cotag=bad_cotag,
        cotag_distribution=cotag_dist,
        exclusion_reasons=excl_reasons,
    )
    return out, stats


def build_recipient_filter(transactions: pd.DataFrame
                           ) -> tuple[pd.DataFrame, FilterStats]:
    LOG.info("Aggregating business-types per recipient (UEI grain)")
    per_rec = aggregate_business_types_per_recipient(transactions)
    LOG.info("Distinct recipients seen in transactions: %d", len(per_rec))
    table, stats = apply_filter(per_rec)
    LOG.info(
        "M-rule: in-scope=%d  no_M=%d  disqualifying_cotag=%d",
        stats.in_scope_recipients, stats.excluded_no_m_tag,
        stats.excluded_by_disqualifying_cotag,
    )
    return table, stats


def write_outputs(filter_df: pd.DataFrame, stats: FilterStats,
                  out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or config.PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "recipient_filter_lightweight.parquet"
    stats_json = out_dir / "filter_stats_lightweight.json"
    # Persist bt_set as a sorted joined string for parquet round-trip stability.
    out = filter_df.copy()
    out["bt_set"] = out["bt_set"].apply(lambda s: "".join(sorted(s)))
    out.to_parquet(parquet, index=False)
    import json
    stats_dict = {
        **stats.__dict__,
        "cotag_distribution": dict(sorted(
            stats.cotag_distribution.items(),
            key=lambda kv: -kv[1],
        )),
        "exclusion_reasons": dict(sorted(
            stats.exclusion_reasons.items(),
            key=lambda kv: -kv[1],
        )),
    }
    stats_json.write_text(json.dumps(stats_dict, indent=2))
    return parquet, stats_json
