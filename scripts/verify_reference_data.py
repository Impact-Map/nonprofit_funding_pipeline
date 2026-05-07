"""Verify pipeline reference data against authoritative sources.

Two checks:
  1. CFDA / Assistance Listing numbers in the YAML rule files: report any
     listing that does not appear in the actual FY22-FY25 USAspending data
     under interim/transactions_fy*.parquet, then optionally probe the
     USAspending API for the listing's existence elsewhere. Listings
     missing from both signals are flagged for manual review (likely
     retired or renamed since the YAMLs were authored).
  2. CPI-U deflator values hardcoded in src/analytic/tables.py: pull the
     current monthly series (CUUR0000SA0) from the BLS public API,
     compute fiscal-year averages, and report the delta vs the hardcoded
     values.

Usage:
    python3 scripts/verify_reference_data.py
    python3 scripts/verify_reference_data.py --skip-api    # offline mode
    python3 scripts/verify_reference_data.py --cpi-only
    python3 scripts/verify_reference_data.py --cfda-only

Exits 0 even if discrepancies are found; this is a reporting tool, not a
gate. Read the output and decide whether updates are warranted.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.analytic.tables import CPI_U_FY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("verify_reference_data")


# ---------------------------------------------------------------------------
# 1. CFDA / Assistance Listing verification
# ---------------------------------------------------------------------------

YAML_TO_LISTING_FIELD = [
    # (yaml_file, top-level key, sub-key or None) -> list of CFDA strings
    ("intl_listings.yaml", "explicit_includes", None),
    ("hospital_listings.yaml", "fqhc", "listings"),
    ("hospital_listings.yaml", "general", "listings"),
    ("covid_programs.yaml", "listings", None),
]


def collect_yaml_listings() -> dict[str, list[tuple[str, str]]]:
    """Return {cfda_number: [(yaml_file, key_path), ...]} mapping each
    listing in the YAML files back to where it came from. The same CFDA
    can be referenced by multiple YAMLs."""
    refs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fname, key1, key2 in YAML_TO_LISTING_FIELD:
        path = config.REFERENCE_LISTS / fname
        if not path.exists():
            LOG.warning("YAML not found: %s", path)
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if doc is None:
            continue
        listings = doc.get(key1, [])
        if key2 is not None:
            listings = listings.get(key2, []) if isinstance(listings, dict) else []
        for cfda in listings or []:
            cfda_s = str(cfda).strip()
            if cfda_s:
                key_path = f"{fname}:{key1}" + (f".{key2}" if key2 else "")
                refs[cfda_s].append((fname, key_path))
    return refs


def listings_present_in_local_data() -> set[str]:
    """Distinct assistance_listing_number values in our FY22-FY25 interim
    parquets. Listings present here are by definition active in the
    USAspending data we work with."""
    seen: set[str] = set()
    for fy in (2022, 2023, 2024, 2025):
        p = config.INTERIM / f"transactions_fy{fy}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["assistance_listing_number"])
        except Exception as exc:
            LOG.warning("Could not read %s: %s", p, exc)
            continue
        for v in df["assistance_listing_number"].dropna().unique():
            seen.add(str(v).strip())
    return seen


def probe_usaspending_listing(cfda: str, timeout: int = 15) -> str:
    """Returns 'exists', 'missing', or 'unknown'. USAspending's autocomplete
    endpoint accepts a CFDA filter and returns matches."""
    url = "https://api.usaspending.gov/api/v2/autocomplete/cfda/"
    body = json.dumps({"search_text": cfda, "limit": 5}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        LOG.debug("HTTP %d on CFDA %s: %s", e.code, cfda, e.reason)
        return "unknown"
    except Exception as e:
        LOG.debug("Request failed for CFDA %s: %s", cfda, e)
        return "unknown"

    matches = data.get("results", []) or []
    for m in matches:
        if str(m.get("program_number") or "").strip() == cfda.strip():
            return "exists"
    return "missing"


def verify_cfda(skip_api: bool = False) -> int:
    refs = collect_yaml_listings()
    if not refs:
        LOG.warning("No CFDA listings found in any YAML — nothing to verify.")
        return 0

    LOG.info("Loaded %d distinct CFDA listings from %d YAML files",
             len(refs), len({fn for cfda_refs in refs.values() for fn, _ in cfda_refs}))

    local = listings_present_in_local_data()
    LOG.info("Local data has %d distinct CFDA listings observed in FY22-FY25 transactions",
             len(local))

    # Bucket each YAML listing by signal:
    #   in local data -> definitely active
    #   not in local data, exists per USAspending -> active but didn't fund 501c3 in our window
    #   not in local data, missing per USAspending -> likely retired/renamed
    #   not in local data, API unknown -> couldn't tell
    in_data: list[str] = []
    not_in_data_exists: list[str] = []
    not_in_data_missing: list[str] = []
    not_in_data_unknown: list[str] = []

    for cfda in sorted(refs):
        if cfda in local:
            in_data.append(cfda)
            continue
        if skip_api:
            not_in_data_unknown.append(cfda)
            continue
        verdict = probe_usaspending_listing(cfda)
        if verdict == "exists":
            not_in_data_exists.append(cfda)
        elif verdict == "missing":
            not_in_data_missing.append(cfda)
        else:
            not_in_data_unknown.append(cfda)
        time.sleep(0.25)  # be polite to USAspending

    print()
    print("=" * 78)
    print("CFDA / ASSISTANCE LISTING VERIFICATION")
    print("=" * 78)
    print(f"\n  YAML listings checked:         {len(refs):>4}")
    print(f"  Found in local FY22-FY25 data: {len(in_data):>4}  (definitely active)")
    print(f"  Not in data, API says exists:  {len(not_in_data_exists):>4}  (active but didn't fund 501(c)(3) in window)")
    print(f"  Not in data, API says missing: {len(not_in_data_missing):>4}  (likely retired - REVIEW)")
    print(f"  Not in data, API unknown:      {len(not_in_data_unknown):>4}  (probe failed - manual check)")

    if not_in_data_missing:
        print("\n  Listings flagged for review (likely retired or renumbered):")
        for cfda in not_in_data_missing:
            sources = ", ".join(p for _, p in refs[cfda])
            print(f"    {cfda:<10s}  referenced by: {sources}")

    if not_in_data_exists and not skip_api:
        print("\n  Listings active per API but absent from FY22-FY25 501(c)(3) data:")
        print("    (informational - these CFDAs exist but no 501(c)(3) recipient")
        print("     received an award under them in our window. Often expected;")
        print("     a few might be misclassified.)")
        for cfda in not_in_data_exists:
            sources = ", ".join(p for _, p in refs[cfda])
            print(f"    {cfda:<10s}  referenced by: {sources}")

    if not_in_data_unknown:
        print("\n  Listings the API could not verify (network or rate-limit):")
        for cfda in not_in_data_unknown[:10]:
            print(f"    {cfda:<10s}  re-run later or check sam.gov manually")
        if len(not_in_data_unknown) > 10:
            print(f"    ... and {len(not_in_data_unknown)-10} more")

    return len(not_in_data_missing)  # return how many need attention


# ---------------------------------------------------------------------------
# 2. CPI-U verification against BLS
# ---------------------------------------------------------------------------

BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CPI_U_SERIES = "CUUR0000SA0"  # All Urban Consumers, U.S. city avg, all items


def fetch_bls_cpi_u(start_year: int = 2021, end_year: int = 2025,
                    timeout: int = 20) -> dict[tuple[int, int], float]:
    """Returns {(year, month): cpi_value} for the requested span."""
    body = json.dumps({
        "seriesid": [CPI_U_SERIES],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }).encode()
    req = urllib.request.Request(
        BLS_API + CPI_U_SERIES,
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    if data.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API: {data.get('status')} — {data.get('message')}")
    series = data["Results"]["series"]
    if not series:
        raise RuntimeError("BLS API returned no series data")
    out: dict[tuple[int, int], float] = {}
    for entry in series[0]["data"]:
        period = entry["period"]
        if not period.startswith("M") or period == "M13":  # M13 is annual avg
            continue
        # BLS reports "-" for months that are not yet released. Skip them.
        raw = str(entry["value"]).strip()
        if not raw or raw == "-":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        month = int(period[1:])
        year = int(entry["year"])
        out[(year, month)] = value
    return out


def fy_average(monthly: dict[tuple[int, int], float], fy: int) -> float | None:
    """FY = Oct (fy-1) through Sep (fy). Average of 12 monthly values."""
    months_needed = [(fy - 1, m) for m in (10, 11, 12)] + [(fy, m) for m in range(1, 10)]
    values = [monthly.get(ym) for ym in months_needed]
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)


def verify_cpi() -> int:
    print()
    print("=" * 78)
    print("CPI-U DEFLATOR VERIFICATION")
    print("=" * 78)
    print(f"\nHardcoded values in src/analytic/tables.py:")
    for fy, val in sorted(CPI_U_FY.items()):
        print(f"  FY{fy}: {val:>9.3f}")

    try:
        monthly = fetch_bls_cpi_u()
    except Exception as e:
        print(f"\nBLS API fetch FAILED: {e}")
        print("  (Check network, or run with --skip-api to bypass.)")
        return -1

    print(f"\nBLS series CUUR0000SA0 — fiscal-year averages from {len(monthly)} monthly observations:")
    print(f"  {'FY':<6s} {'BLS-derived':>12s} {'Hardcoded':>12s} {'Delta':>10s}")
    flagged = 0
    for fy in sorted(CPI_U_FY):
        live = fy_average(monthly, fy)
        hardcoded = CPI_U_FY[fy]
        if live is None:
            print(f"  FY{fy}: BLS data incomplete (FY may not have closed yet)")
            continue
        delta = live - hardcoded
        delta_pct = 100 * delta / hardcoded
        flag = ""
        if abs(delta_pct) > 0.5:
            flag = "  <- check"
            flagged += 1
        print(f"  FY{fy:<5d} {live:>12.3f} {hardcoded:>12.3f} {delta:>+9.3f} ({delta_pct:>+5.2f}%){flag}")

    print(f"\nFY-averages with delta > 0.5% from hardcoded value: {flagged}")
    if flagged:
        print("  Recommend updating CPI_U_FY in src/analytic/tables.py and re-running")
        print("  --tables --exhibits to refresh the *_real columns.")
    return flagged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cfda-only", action="store_true")
    p.add_argument("--cpi-only", action="store_true")
    p.add_argument("--skip-api", action="store_true",
                   help="Skip CFDA API probes and BLS fetch; report only what's "
                        "verifiable from local data.")
    args = p.parse_args()

    flagged_total = 0
    if not args.cpi_only:
        flagged_total += max(0, verify_cfda(skip_api=args.skip_api))
    if not args.cfda_only and not args.skip_api:
        flagged_total += max(0, verify_cpi())

    print()
    print("=" * 78)
    print(f"TOTAL items flagged for review: {flagged_total}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
