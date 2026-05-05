"""Project paths, fixed parameters, and configuration constants.

Centralizes the directory layout from Section 10.1 of the methodology and the
fixed analytic parameters (FY window, in-scope award type codes, match-tier
thresholds, default deflator). Anything that can vary between runs lives here
so it shows up cleanly in the run manifest.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(os.environ.get("USASP_PROJECT_ROOT", Path(__file__).resolve().parent.parent))

RAW = PROJECT_ROOT / "raw"
INTERIM = PROJECT_ROOT / "interim"
REFERENCE = PROJECT_ROOT / "reference"
REFERENCE_LISTS = PROJECT_ROOT / "reference_lists"
PROCESSED = PROJECT_ROOT / "processed"
EXHIBITS = PROJECT_ROOT / "exhibits"
MANIFESTS = PROJECT_ROOT / "manifests"

RAW_USASPENDING = RAW / "usaspending"
RAW_IRS_BMF = RAW / "irs_bmf"
RAW_SAM = RAW / "sam"
RAW_REFERENCE = RAW / "reference"

# In-scope financial-assistance award types per DAIMS (methodology 3.1).
ASSISTANCE_AWARD_TYPE_CODES: tuple[str, ...] = (
    "02", "03", "04", "05", "06", "07", "08", "09", "10", "11",
)

# Fiscal-year window. FY22 starts Oct 1, 2021; FY25 ends Sep 30, 2025.
FY_START_DATE = "2021-10-01"
FY_END_DATE = "2025-09-30"
FISCAL_YEARS: tuple[int, ...] = (2022, 2023, 2024, 2025)

# Recipient-match thresholds (Section 5).
TIER3_JARO_WINKLER_THRESHOLD = 0.94
TIER4_MANUAL_REVIEW_TOP_N = 200
TIER3_FALSE_POSITIVE_AUDIT_SAMPLE = 200
TIER3_PRECISION_TARGET = 0.95

# International rule threshold for NTEE Q-series (Section 4.4).
INTL_Q_SERIES_DOLLAR_FLOOR = 100_000

# Reporting defaults.
DEFAULT_DEFLATOR = "CPI-U"
DEFLATOR_BASE_FY = 2025

# Probabilistic-match score (rapidfuzz token_set_ratio is 0-100, threshold per 3.2).
PROB_MATCH_TOKEN_SET_RATIO = 92


@dataclass(frozen=True)
class RunConfig:
    """Per-run knobs. Snapshots into the manifest."""
    bmf_release_tag: str = "current"  # e.g. "2026-04"
    classification_priority: tuple[str, ...] = ("international", "hospital", "educational", "core")
    deflator: str = DEFAULT_DEFLATOR
    deflator_base_fy: int = DEFLATOR_BASE_FY
    intl_q_series_floor: float = INTL_Q_SERIES_DOLLAR_FLOOR
    fiscal_years: tuple[int, ...] = FISCAL_YEARS
    award_type_codes: tuple[str, ...] = ASSISTANCE_AWARD_TYPE_CODES


def ensure_dirs(extras: Iterable[Path] = ()) -> None:
    """Create the project directories if missing. Idempotent."""
    for d in (
        RAW, INTERIM, REFERENCE, REFERENCE_LISTS, PROCESSED, EXHIBITS, MANIFESTS,
        RAW_USASPENDING, RAW_IRS_BMF, RAW_SAM, RAW_REFERENCE,
        *extras,
    ):
        d.mkdir(parents=True, exist_ok=True)
