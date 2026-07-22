"""Category classification with the priority hierarchy (Section 4).

Default priority: International > Hospital > Educational > Core. Each
transaction is assigned to exactly one panel. The priority order is
configurable through `RunConfig.classification_priority`.

Each transaction also carries `classification_rule_hits`: an array of every
rule that *would have fired* for it, regardless of which one won. This is the
audit trail required by the methodology.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .. import config
from ..refdata import load_reference_lists

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-rule predicate builders. Each returns a boolean Series aligned with the
# input frame and the rule_id that should be appended to rule_hits when True.
# ---------------------------------------------------------------------------


def _intl_subagency_set(refs) -> set[str]:
    s: set[str] = set()
    for entry in refs.intl_subagencies.get("subagencies", []):
        s.add(_norm_agency(entry["canonical"]))
        for a in entry.get("aliases", []) or []:
            s.add(_norm_agency(a))
    return s


def _intl_listing_predicate(refs):
    listings = refs.intl_listings
    prefixes = tuple(listings.get("prefix_includes", []) or [])
    explicit = set(listings.get("explicit_includes", []) or [])
    excludes = set(listings.get("explicit_excludes", []) or [])

    def fn(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str)
        prefix_hit = pd.Series(False, index=s.index)
        for p in prefixes:
            prefix_hit |= s.str.startswith(p)
        explicit_hit = s.isin(explicit)
        return (prefix_hit | explicit_hit) & ~s.isin(excludes)

    return fn


def _norm_agency(name: str) -> str:
    return " ".join(str(name).split()).strip().casefold()


def _ntee_in_range(ntee: pd.Series, start: str, end: str) -> pd.Series:
    """Range check on NTEE primary code (string compare works lexicographically)."""
    n = ntee.fillna("").astype(str).str.upper()
    return (n >= start) & (n <= end)


def _ntee_starts_with(ntee: pd.Series, prefix: str) -> pd.Series:
    return ntee.fillna("").astype(str).str.upper().str.startswith(prefix)


# ---------------------------------------------------------------------------
# Subcategory dispatch for International (Section 4.4 sub-cuts).
# ---------------------------------------------------------------------------


def _intl_subcategory(row: pd.Series, rules: dict) -> str:
    sub_kw = lambda k: (rules.get(k, {}) or {}).get("subagency_keywords", []) or []
    listings_in = lambda k: set((rules.get(k, {}) or {}).get("listings", []) or [])
    listing_pref = lambda k: tuple((rules.get(k, {}) or {}).get("listing_prefixes", []) or [])
    program_kw = lambda k: (rules.get(k, {}) or {}).get("program_keywords", []) or []

    subagency = (str(row.get("awarding_sub_agency_name", "")) or "").casefold()
    listing = (str(row.get("assistance_listing_number", "")) or "")
    program = (str(row.get("program_activity_name", "")) or
               str(row.get("assistance_listing_title", "")) or "").casefold()

    for cat in ("humanitarian", "global_health", "governance_economic", "security", "education_research"):
        if any(k.casefold() in subagency for k in sub_kw(cat)):
            return cat
        if listing in listings_in(cat):
            return cat
        if any(listing.startswith(p) for p in listing_pref(cat)):
            return cat
        if any(k.casefold() in program for k in program_kw(cat)):
            return cat
    return "other"


# ---------------------------------------------------------------------------
# Educational subcategory dispatch (Section 4.2).
# ---------------------------------------------------------------------------


def _safe_upper(v) -> str:
    """Coerce a possibly-NaN / possibly-None cell to an uppercase string.
    row.get(col) can return NaN (float) when the column exists but the cell
    is null; `NaN or ""` returns NaN, not "", so the naive `.upper()` fails."""
    if v is None:
        return ""
    if isinstance(v, float):
        # NaN check without importing math
        if v != v:
            return ""
    s = str(v).strip()
    return s.upper() if s.lower() != "nan" else ""


def _educational_subcategory(row: pd.Series) -> str:
    """Sub-cut labels within the Educational panel.

    NTEE B-series canonical assignments (from the NCCS classification manual):
      B20-B29  elementary & secondary (K-12)
      B30-B39  vocational / technical
      B40      Higher Education (general / unspecified 4-year)
      B41      Community / Junior College (2-year)
      B42      Undergraduate College (4-year)
      B43      University / Technological Institute (4-year, doctoral)
      B44-B49  Higher-ed variants (rare; treat as 4-year)
      B50-B59  Graduate & Professional Schools (4-year+)
      B60-B99  Adult ed, libraries, student services, education NEC

    Public vs private for 4-year is decided by business_types code (H = state
    IHE, O = private IHE) if present; otherwise falls back to a name heuristic.
    """
    ntee = _safe_upper(row.get("bmf_ntee"))
    biz_codes = _safe_upper(row.get("recipient_business_types"))
    is_public = "06" in biz_codes or "H" in biz_codes
    is_private = "11" in biz_codes or "O" in biz_codes

    if ntee:
        # 2-year colleges (B41 only) — B40 and B42+ are 4-year.
        if ntee == "B41":
            return "public_2yr" if is_public else "private_2yr"
        # 4-year colleges and universities
        if ntee == "B40" or ("B42" <= ntee <= "B59"):
            if is_public:
                return "public_4yr"
            if is_private:
                return "private_4yr"
            return "higher_ed_4yr"
        if "B20" <= ntee <= "B29":
            return "k12"
        if "B30" <= ntee <= "B39":
            return "vocational"
        if "B60" <= ntee <= "B99":
            return "other_education"

    # Business-types fallback (used when NTEE is missing, which is common for
    # large universities per NCCS coverage gaps).
    if is_public:
        return "public_4yr"
    if is_private:
        return "private_4yr"
    if "12" in biz_codes or "G" in biz_codes:
        return "k12"
    if "T" in biz_codes:
        return "hbcu"
    if "U" in biz_codes:
        return "tccu"
    if "S" in biz_codes:
        return "hispanic_serving"
    if "V" in biz_codes:
        return "alaska_native_hawaiian_serving"
    return "other_education"


# ---------------------------------------------------------------------------
# Hospital subcategory dispatch (Section 4.3).
# ---------------------------------------------------------------------------


def _hospital_subcategory(row: pd.Series) -> str:
    ntee = _safe_upper(row.get("bmf_ntee"))
    if ntee:
        if "E20" <= ntee <= "E22":
            return "general_hospital"
        if "E22" < ntee <= "E25":
            return "specialty_hospital"
        if "E30" <= ntee <= "E32":
            return "fqhc_clinic"
    if row.get("hrsa_uds_match"):
        return "fqhc_clinic"
    if row.get("aha_match"):
        return "general_hospital"
    return "other_hospital"


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


@dataclass
class CategoryStats:
    rows_total: int
    rows_international: int
    rows_hospital: int
    rows_educational: int
    rows_core: int


def classify(transactions: pd.DataFrame,
             recipient_match: pd.DataFrame,
             aha_eins: Iterable[str] | None = None,
             hrsa_eins: Iterable[str] | None = None,
             ipeds_eins: Iterable[str] | None = None,
             nces_school_district_names: Iterable[str] | None = None,
             priority: tuple[str, ...] = ("international", "hospital", "educational", "core"),
             intl_q_dollar_floor: float = config.INTL_Q_SERIES_DOLLAR_FLOOR,
             ) -> tuple[pd.DataFrame, CategoryStats]:
    """Apply category classification.

    Parameters mirror Section 4. The supplemental EIN sets (AHA, HRSA UDS,
    IPEDS) are optional; pass empty iterables when the underlying lists are
    not yet provisioned. The classifier degrades gracefully: NTEE-only rules
    still fire.
    """
    refs = load_reference_lists()

    df = transactions.copy()
    if "federal_action_obligation" in df.columns:
        df["federal_action_obligation"] = pd.to_numeric(
            df["federal_action_obligation"], errors="coerce"
        ).fillna(0.0)

    # Bring matched BMF NTEE/foundation onto the transaction row.
    rm = recipient_match[["recipient_uei", "irs_ein", "match_tier",
                          "bmf_ntee", "bmf_foundation", "bmf_name"]].copy()
    df = df.merge(rm, on="recipient_uei", how="left")

    aha_set = set(map(str, aha_eins or []))
    hrsa_set = set(map(str, hrsa_eins or []))
    ipeds_set = set(map(str, ipeds_eins or []))
    nces_names = set(n.casefold().strip() for n in (nces_school_district_names or []) if n)

    df["aha_match"] = df["irs_ein"].isin(aha_set)
    df["hrsa_uds_match"] = df["irs_ein"].isin(hrsa_set)
    df["ipeds_match"] = df["irs_ein"].isin(ipeds_set)

    name_low = df.get("recipient_name", pd.Series([""] * len(df))).fillna("").str.casefold()
    df["nces_ccd_match"] = name_low.isin(nces_names) if nces_names else False

    # ---------- Build rule_hit columns ----------
    intl_subagency_set = _intl_subagency_set(refs)
    intl_listing_pred = _intl_listing_predicate(refs)
    biz_codes_intl = (
        df.get("recipient_business_types", pd.Series([""] * len(df)))
          .fillna("").astype(str)
    )

    pop_country = df.get("primary_place_of_performance_country_code",
                         df.get("place_of_performance_country_code"))
    if pop_country is None:
        pop_country = pd.Series([""] * len(df))
    pop_country = pop_country.fillna("").astype(str).str.upper()

    df["_rule_intl_pop"] = (pop_country != "") & (pop_country != "USA") & (pop_country != "UNITED STATES")

    sub_agency_norm = (
        df.get("awarding_sub_agency_name", pd.Series([""] * len(df)))
          .fillna("").astype(str).map(_norm_agency)
    )
    df["_rule_intl_subagency"] = sub_agency_norm.isin(intl_subagency_set)

    listing = df.get("assistance_listing_number", pd.Series([""] * len(df))).fillna("").astype(str)
    df["_rule_intl_listing"] = intl_listing_pred(listing)

    obligation = df.get("federal_action_obligation", pd.Series([0.0] * len(df)))
    df["_rule_intl_q_series"] = (
        _ntee_starts_with(df["bmf_ntee"], "Q") & (obligation > intl_q_dollar_floor)
    )

    # Hospital rules
    df["_rule_hosp_ntee_e20_e25"] = _ntee_in_range(df["bmf_ntee"], "E20", "E25")
    df["_rule_hosp_ntee_e30_e32"] = _ntee_in_range(df["bmf_ntee"], "E30", "E32")
    df["_rule_hosp_aha"] = df["aha_match"]
    df["_rule_hosp_hrsa"] = df["hrsa_uds_match"]
    biz = df.get("recipient_business_types", pd.Series([""] * len(df))).fillna("").astype(str)
    # business_types_code appears in two encodings depending on acquisition
    # path: numeric SF-424 codes (bulk_download API) and letter codes
    # (Award Data Archive Public Profile format). Match both. There is no
    # dedicated hospital letter code in SF-424; numeric 26 is the only signal.
    df["_rule_hosp_biz"] = biz.str.contains(r"\b26\b", regex=True)

    # Educational rules. business_types signals for education (both encodings):
    #   Numeric (bulk_download): 06 State IHE, 11 Private IHE, 12 School
    #                            District, 23 Other Educational
    #   Letter (Award Archive):  H State/Public IHE, O Private IHE,
    #                            G Independent School District,
    #                            T HBCU, U TCCU, S Hispanic-Serving,
    #                            V Alaska Native / Native Hawaiian-Serving
    df["_rule_edu_ntee_b"] = _ntee_starts_with(df["bmf_ntee"], "B")
    df["_rule_edu_ipeds"] = df["ipeds_match"]
    df["_rule_edu_nces_ccd"] = df["nces_ccd_match"]
    numeric_edu_pattern = r"\b(?:06|11|12|23)\b"
    # Letter codes: check that the letter is present as a distinct token in
    # the concatenated business_types string. Letters can appear either as
    # single-letter runs ("O", "OMX", "O,M,X") or space/comma-separated.
    # A word-boundary check on the concatenated string catches the standalone
    # cases; a set-membership check on the parsed set catches the concatenated
    # ones like "OMX".
    letter_edu_codes = ("H", "O", "T", "U", "S", "V", "G")
    def _has_edu_letter(s):
        upper = s.upper()
        return any(c in upper for c in letter_edu_codes)
    letter_edu_hit = biz.map(_has_edu_letter).fillna(False)
    df["_rule_edu_biz"] = biz.str.contains(numeric_edu_pattern, regex=True) | letter_edu_hit

    # ---------- Compose category from rule columns ----------
    is_intl = (
        df["_rule_intl_pop"] | df["_rule_intl_subagency"]
        | df["_rule_intl_listing"] | df["_rule_intl_q_series"]
    )
    is_hospital = (
        df["_rule_hosp_ntee_e20_e25"] | df["_rule_hosp_ntee_e30_e32"]
        | df["_rule_hosp_aha"] | df["_rule_hosp_hrsa"] | df["_rule_hosp_biz"]
    )
    is_educational = (
        df["_rule_edu_ntee_b"] | df["_rule_edu_ipeds"]
        | df["_rule_edu_nces_ccd"] | df["_rule_edu_biz"]
    )

    category = pd.Series("core", index=df.index, dtype=object)
    # Apply in reverse priority so earlier wins overwrite later.
    cat_to_mask = {
        "international": is_intl,
        "hospital": is_hospital,
        "educational": is_educational,
        "core": pd.Series(True, index=df.index),
    }
    for cat in reversed(priority):
        category[cat_to_mask[cat]] = cat
    df["recipient_category"] = category

    # ---------- Subcategories ----------
    intl_rules = refs.intl_subcategory_rules
    intl_mask = df["recipient_category"] == "international"
    df["recipient_subcategory"] = ""
    if intl_mask.any():
        df.loc[intl_mask, "recipient_subcategory"] = df.loc[intl_mask].apply(
            lambda r: _intl_subcategory(r, intl_rules), axis=1
        )

    hosp_mask = df["recipient_category"] == "hospital"
    if hosp_mask.any():
        df.loc[hosp_mask, "recipient_subcategory"] = df.loc[hosp_mask].apply(
            _hospital_subcategory, axis=1
        )

    edu_mask = df["recipient_category"] == "educational"
    if edu_mask.any():
        df.loc[edu_mask, "recipient_subcategory"] = df.loc[edu_mask].apply(
            _educational_subcategory, axis=1
        )

    # ---------- Audit trail: list of every rule that fired ----------
    rule_cols = [c for c in df.columns if c.startswith("_rule_")]
    df["classification_rule_hits"] = (
        df[rule_cols]
        .astype(bool)
        .apply(lambda row: [c.replace("_rule_", "") for c in rule_cols if row[c]], axis=1)
    )

    # COVID flag (methodology 7.2 robustness)
    covid_listings = set(refs.covid_programs.get("listings", []) or [])
    covid_kw = [k.casefold() for k in (refs.covid_programs.get("program_name_keywords", []) or [])]
    listing_str = listing
    program_text = (
        df.get("assistance_listing_title", pd.Series([""] * len(df)))
          .fillna("").astype(str).str.casefold()
    )
    covid_kw_hit = pd.Series(False, index=df.index)
    for kw in covid_kw:
        covid_kw_hit |= program_text.str.contains(kw, regex=False)
    df["covid_flag"] = listing_str.isin(covid_listings) | covid_kw_hit

    # Drop the temporary rule columns from the persisted output.
    df = df.drop(columns=rule_cols)

    stats = CategoryStats(
        rows_total=len(df),
        rows_international=int((df["recipient_category"] == "international").sum()),
        rows_hospital=int((df["recipient_category"] == "hospital").sum()),
        rows_educational=int((df["recipient_category"] == "educational").sum()),
        rows_core=int((df["recipient_category"] == "core").sum()),
    )
    return df, stats
