"""Phase 1 lightweight panel classification (heuristic carve-outs).

Implements methodology Section 4 with USAspending-only signals:
  - International:  same transaction-level rules as full methodology Section
                    4.4, less the NTEE Q-series rule (no NTEE access).
  - Hospital:       recipient-name regex + curated assistance-listing flags
                    (Section 4.3).
  - Educational:    recipient-name regex (charter, academy, education
                    foundation/fund) + research-nonprofit pattern when paired
                    with NSF/NIH/HHS/ED awarding (Section 4.4).
  - Core:           residual.

Priority order: International > Hospital > Educational > Core.

Each transaction carries a `classification_rule_hits` audit array listing
every rule that fired for it (regardless of which one won the priority
hierarchy), per the same audit-trail convention as the full methodology.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..refdata import load_reference_lists
from ..classify.categorize import _norm_agency, _intl_subagency_set, _intl_listing_predicate, _intl_subcategory

LOG = logging.getLogger(__name__)


@dataclass
class CategoryStats:
    rows_total: int
    rows_international: int
    rows_hospital: int
    rows_educational: int
    rows_core: int


def _compile_patterns(rule_block: list[dict]) -> list[tuple[str, re.Pattern]]:
    """Compile a list of {label, regex} patterns to (label, regex) pairs."""
    out = []
    for entry in rule_block or []:
        label = entry.get("label", "")
        pat = entry.get("regex", "")
        if pat:
            out.append((label, re.compile(pat, re.IGNORECASE)))
    return out


def _name_flag(name_series: pd.Series,
               include_patterns: list[tuple[str, re.Pattern]],
               exclude_patterns: list[tuple[str, re.Pattern]]
               ) -> tuple[pd.Series, pd.Series]:
    """Return (hit_mask, hit_labels_per_row).

    A row matches if any include pattern fires AND no exclude pattern fires.
    `hit_labels_per_row` is a list of fired label strings for the audit trail.
    """
    s = name_series.fillna("").astype(str)
    fired = pd.Series([[] for _ in range(len(s))], index=s.index, dtype=object)
    inc_mask = pd.Series(False, index=s.index)
    for label, pat in include_patterns:
        m = s.str.contains(pat)
        inc_mask |= m
        fired.loc[m] = fired.loc[m].apply(lambda lst, lab=label: lst + [lab])

    exc_mask = pd.Series(False, index=s.index)
    for label, pat in exclude_patterns:
        m = s.str.contains(pat)
        exc_mask |= m
        fired.loc[m] = fired.loc[m].apply(lambda lst, lab=label: lst + [lab])

    return inc_mask & ~exc_mask, fired


def classify(transactions: pd.DataFrame,
             priority: tuple[str, ...] = ("international", "hospital", "educational", "core"),
             ) -> tuple[pd.DataFrame, CategoryStats]:
    """Apply the heuristic panel rules.

    `transactions` must already be filtered to the M-with-exclusions in-scope
    set (call recipient_filter.build_recipient_filter and join first).
    """
    refs = load_reference_lists()
    df = transactions.copy()
    if "federal_action_obligation" in df.columns:
        df["federal_action_obligation"] = pd.to_numeric(
            df["federal_action_obligation"], errors="coerce"
        ).fillna(0.0)

    # ---------------- International rules (Section 4.2) ----------------
    pop_country = df.get("primary_place_of_performance_country_code",
                         df.get("place_of_performance_country_code"))
    if pop_country is None:
        pop_country = pd.Series([""] * len(df), index=df.index)
    pop_country = pop_country.fillna("").astype(str).str.upper()
    df["_rule_intl_pop"] = (
        (pop_country != "") & (pop_country != "USA") & (pop_country != "UNITED STATES")
    )

    intl_subagency_set = _intl_subagency_set(refs)
    sub_agency_norm = (
        df.get("awarding_sub_agency_name", pd.Series([""] * len(df), index=df.index))
        .fillna("").astype(str).map(_norm_agency)
    )
    df["_rule_intl_subagency"] = sub_agency_norm.isin(intl_subagency_set)

    listing = df.get("assistance_listing_number",
                     pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    df["_rule_intl_listing"] = _intl_listing_predicate(refs)(listing)

    # ---------------- Hospital rules (Section 4.3) ----------------
    name = df.get("recipient_name", pd.Series([""] * len(df), index=df.index))
    hosp_inc = _compile_patterns(refs.hospital_name_patterns.get("patterns", []))
    hosp_exc = _compile_patterns(refs.hospital_name_patterns.get("exclusions", []))
    name_hit, name_labels = _name_flag(name, hosp_inc, hosp_exc)
    df["_rule_hosp_name"] = name_hit
    df["_rule_hosp_name_labels"] = name_labels

    # FQHC: HRSA + curated listings
    hosp_listings = refs.hospital_listings or {}
    fqhc = hosp_listings.get("fqhc", {})
    fqhc_keyword = (fqhc.get("subagency_keyword") or "").casefold()
    fqhc_listings = set(fqhc.get("listings") or [])
    sub_agency_low = (
        df.get("awarding_sub_agency_name", pd.Series([""] * len(df), index=df.index))
        .fillna("").astype(str).str.casefold()
    )
    df["_rule_hosp_fqhc"] = (
        sub_agency_low.str.contains(fqhc_keyword, regex=False) & listing.isin(fqhc_listings)
    ) if fqhc_keyword else pd.Series(False, index=df.index)

    # Research hospital: NIH sub-agency + 93.3xx listing + name-flag (require
    # name_hit so we don't sweep up NIH grants to universities).
    research = hosp_listings.get("research", {})
    research_keyword = (research.get("subagency_keyword") or "").casefold()
    research_prefix = research.get("listings_prefix") or ""
    df["_rule_hosp_research"] = (
        sub_agency_low.str.contains(research_keyword, regex=False)
        & listing.str.startswith(research_prefix)
        & name_hit
    ) if research_keyword and research_prefix else pd.Series(False, index=df.index)

    general_listings = set((hosp_listings.get("general", {}) or {}).get("listings") or [])
    df["_rule_hosp_general"] = listing.isin(general_listings)

    # ---------------- Educational rules (Section 4.4) ----------------
    edu_inc = _compile_patterns(refs.educational_name_patterns.get("patterns", []))
    edu_exc = _compile_patterns(refs.educational_name_patterns.get("exclusions", []))
    edu_name_hit, edu_name_labels = _name_flag(name, edu_inc, edu_exc)
    df["_rule_edu_name"] = edu_name_hit
    df["_rule_edu_name_labels"] = edu_name_labels

    research_inc = _compile_patterns(refs.educational_name_patterns.get("research_patterns", []))
    research_agency_kw = [
        k.casefold() for k in refs.educational_name_patterns.get("research_agency_keywords", []) or []
    ]
    awarding_agency = (
        df.get("awarding_agency_name", pd.Series([""] * len(df), index=df.index))
        .fillna("").astype(str).str.casefold()
    )
    is_research_agency = pd.Series(False, index=df.index)
    for kw in research_agency_kw:
        is_research_agency |= awarding_agency.str.contains(kw, regex=False)

    research_name_hit, research_name_labels = _name_flag(name, research_inc, edu_exc)
    df["_rule_edu_research"] = research_name_hit & is_research_agency
    df["_rule_edu_research_labels"] = research_name_labels.where(
        df["_rule_edu_research"], pd.Series([[] for _ in range(len(df))], index=df.index)
    )

    # ---------------- Compose category by priority ----------------
    is_intl = (
        df["_rule_intl_pop"] | df["_rule_intl_subagency"] | df["_rule_intl_listing"]
    )
    is_hospital = (
        df["_rule_hosp_name"]
        | df["_rule_hosp_fqhc"]
        | df["_rule_hosp_research"]
        | df["_rule_hosp_general"]
    )
    is_educational = df["_rule_edu_name"] | df["_rule_edu_research"]

    cat_to_mask = {
        "international": is_intl,
        "hospital": is_hospital,
        "educational": is_educational,
        "core": pd.Series(True, index=df.index),
    }
    category = pd.Series("core", index=df.index, dtype=object)
    for cat in reversed(priority):
        category[cat_to_mask[cat]] = cat
    df["recipient_category"] = category

    # ---------------- Subcategories ----------------
    intl_rules = refs.intl_subcategory_rules
    df["recipient_subcategory"] = ""
    intl_mask = df["recipient_category"] == "international"
    if intl_mask.any():
        df.loc[intl_mask, "recipient_subcategory"] = df.loc[intl_mask].apply(
            lambda r: _intl_subcategory(r, intl_rules), axis=1
        )

    # Hospital sub-cuts: prefer FQHC > research > name-flagged > other
    def _hosp_sub(row):
        if row["_rule_hosp_fqhc"]:
            return "fqhc_clinic"
        if row["_rule_hosp_research"]:
            return "research_hospital"
        if row["_rule_hosp_name"]:
            return "general_or_specialty"
        if row["_rule_hosp_general"]:
            return "other_hospital_grant"
        return "other_hospital"
    hosp_mask = df["recipient_category"] == "hospital"
    if hosp_mask.any():
        df.loc[hosp_mask, "recipient_subcategory"] = df.loc[hosp_mask].apply(_hosp_sub, axis=1)

    # Educational: collapsed in lightweight; just one sub-cut by which rule fired.
    def _edu_sub(row):
        if row["_rule_edu_name"]:
            return "education_org"
        if row["_rule_edu_research"]:
            return "research_nonprofit"
        return "other_educational"
    edu_mask = df["recipient_category"] == "educational"
    if edu_mask.any():
        df.loc[edu_mask, "recipient_subcategory"] = df.loc[edu_mask].apply(_edu_sub, axis=1)

    # ---------------- Audit trail ----------------
    rule_cols = [c for c in df.columns if c.startswith("_rule_") and not c.endswith("_labels")]
    label_cols = {
        "_rule_hosp_name": "_rule_hosp_name_labels",
        "_rule_edu_name": "_rule_edu_name_labels",
        "_rule_edu_research": "_rule_edu_research_labels",
    }

    def _audit(row):
        hits = []
        for c in rule_cols:
            if not row[c]:
                continue
            base = c.replace("_rule_", "")
            if c in label_cols:
                lbls = row[label_cols[c]] or []
                hits.extend(lbls if lbls else [base])
            else:
                hits.append(base)
        return hits

    df["classification_rule_hits"] = df.apply(_audit, axis=1)

    # ---------------- COVID flag (same as full methodology) ----------------
    covid_listings = set(refs.covid_programs.get("listings", []) or [])
    covid_kw = [k.casefold() for k in (refs.covid_programs.get("program_name_keywords", []) or [])]
    program_text = (
        df.get("assistance_listing_title", pd.Series([""] * len(df), index=df.index))
        .fillna("").astype(str).str.casefold()
    )
    covid_kw_hit = pd.Series(False, index=df.index)
    for kw in covid_kw:
        covid_kw_hit |= program_text.str.contains(kw, regex=False)
    df["covid_flag"] = listing.isin(covid_listings) | covid_kw_hit

    # Drop intermediate rule columns from persisted output.
    drop_cols = list(rule_cols) + list(label_cols.values())
    df = df.drop(columns=drop_cols, errors="ignore")

    stats = CategoryStats(
        rows_total=len(df),
        rows_international=int((df["recipient_category"] == "international").sum()),
        rows_hospital=int((df["recipient_category"] == "hospital").sum()),
        rows_educational=int((df["recipient_category"] == "educational").sum()),
        rows_core=int((df["recipient_category"] == "core").sum()),
    )
    return df, stats
