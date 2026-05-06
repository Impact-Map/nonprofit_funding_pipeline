"""End-to-end pipeline orchestrator.

Implements the step sequence in methodology Section 10:

  Step 1 - stand up project storage             (config.ensure_dirs)
  Step 2 - acquire USASpending data             (acquire.usaspending)
  Step 3 - acquire and prepare IRS BMF          (acquire.irs_bmf)
  Step 4 - build recipient_match                (match.recipient_match)
  Step 5 - apply category classification        (classify.categorize)
  Step 6 - build analytic tables                (analytic.tables)
  Step 7 - aggregations and exhibits            (aggregate.exhibits)
  Step 8 - QA and sign-off                      (qa.checks)
  + manifest                                    (manifest)

Run subsets via CLI flags. Default `--all` runs the full sequence end-to-end
once raw data is in place.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config, manifest as manifest_mod
from src.acquire import award_archive, irs_bmf, manual as manual_acquire, sam, usaspending
from src.aggregate import exhibits as exhibits_mod
from src.analytic import tables as tables_mod
from src.classify import categorize
from src.match import recipient_match
from src.qa import checks as qa_checks
from src.refdata import load_reference_lists


LOG = logging.getLogger("usasp.pipeline")


# ---------------------------------------------------------------------------
# Loaders for the parquet artifacts produced by upstream steps. The orchestrator
# runs each step independently when invoked, so each step reads its inputs
# from disk and writes its outputs to disk.
# ---------------------------------------------------------------------------


# Columns each downstream step actually consumes. Reading only what's needed
# keeps the four-FY load under ~3 GB instead of ~30 GB (113 cols * dtype=str).
_MATCH_COLUMNS = (
    "recipient_uei", "recipient_duns", "recipient_name", "recipient_state_code",
    "recipient_ein", "federal_action_obligation", "action_date",
    "awarding_agency_name",  # for the match coverage report
)

_CLASSIFY_COLUMNS = (
    "transaction_id", "award_id_unique", "action_date", "award_type_code",
    "action_type", "awarding_agency_name", "awarding_sub_agency_name",
    "assistance_listing_number", "assistance_listing_title",
    "recipient_uei", "recipient_name", "recipient_state_code",
    "recipient_business_types", "primary_place_of_performance_country_code",
    "federal_action_obligation",
    "total_outlayed_amount_for_overall_award",
    "generated_unique_award_id",
    "program_activity_name",
)


def _read_transactions(columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    parts = sorted(config.INTERIM.glob("transactions_fy*.parquet"))
    if not parts:
        raise FileNotFoundError("No transactions parquet under /interim. Run --acquire first.")
    if columns is None:
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # Project columns at read time. Use pyarrow metadata to skip columns that
    # don't exist in a given parquet (different acquisition sources include
    # different column subsets).
    import pyarrow.parquet as pq
    frames = []
    for p in parts:
        schema_cols = set(pq.read_metadata(p).schema.to_arrow_schema().names)
        wanted = [c for c in columns if c in schema_cols]
        frames.append(pd.read_parquet(p, columns=wanted))
    return pd.concat(frames, ignore_index=True)


def _read_bmf() -> pd.DataFrame:
    p = config.INTERIM / "bmf_501c3_current.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Expected BMF parquet at {p}. Run --bmf first.")
    return pd.read_parquet(p)


def _read_sam() -> pd.DataFrame | None:
    p = config.INTERIM / "sam_entity.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _read_match_table() -> pd.DataFrame:
    p = config.PROCESSED / "recipient_match.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Expected match table at {p}. Run --match first.")
    return pd.read_parquet(p)


def _read_classified_transactions() -> pd.DataFrame:
    p = config.PROCESSED / "transactions_classified.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Expected classified parquet at {p}. Run --classify first.")
    return pd.read_parquet(p)


def _read_analytic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    txn = pd.read_parquet(config.PROCESSED / "assistance_txn_501c3.parquet")
    awd = pd.read_parquet(config.PROCESSED / "assistance_awards_501c3.parquet")
    return txn, awd


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def step_acquire(run_cfg: config.RunConfig, run_manifest: manifest_mod.RunManifest,
                 download_workers: int | None = None,
                 source: str = "bulk_download") -> None:
    """Step 2: acquire transaction data.

    `source` selects the acquisition path:
      - "bulk_download": POST to /api/v2/bulk_download/awards/, poll until the
        job finishes, stream the zip. Freshest data; very slow generation
        (~30 min - 2 hr per FY) because the file is generated on demand.
      - "archive": pull pre-generated per-agency zips from
        files.usaspending.gov/award_data_archive/ in parallel. ~10 min total
        for FY22-FY25 because the files already exist on S3. Snapshot lags
        the live data by up to ~30 days; the snapshot date is recorded in
        the manifest.
    """
    if source == "archive":
        LOG.info("Step 2: Award Data Archive (per-agency zips, workers=%s)", download_workers or 16)
        paths, records = award_archive.download_all(
            run_cfg.fiscal_years,
            max_workers=download_workers or 16,
        )
        run_manifest.usaspending_downloads = [asdict(r) for r in records]
        run_manifest.notes.append(
            f"Used Award Data Archive snapshot; {len(paths)} per-agency zips downloaded"
        )
        LOG.info("Step 2: extracting %d archive zips to parquet", len(paths))
        award_archive.extract_archives_to_parquet(run_cfg.fiscal_years)
        return

    if source == "manual":
        LOG.info("Step 2: manual mode - extracting locally-staged zips")
        records, extracted = manual_acquire.acquire_from_manual(run_cfg.fiscal_years)
        run_manifest.usaspending_downloads = [asdict(r) for r in records]
        total_files = sum(r.files for r in records)
        total_mb = sum(r.total_bytes for r in records) / 1e6
        run_manifest.notes.append(
            f"Manual mode: extracted {total_files} local zip(s), {total_mb:.1f} MB; "
            "no network calls to USAspending."
        )
        return

    LOG.info("Step 2: USASpending bulk downloads (workers=%s)", download_workers)
    records = usaspending.download_all(run_cfg.fiscal_years, max_workers=download_workers)
    run_manifest.usaspending_downloads = [asdict(r) for r in records]
    LOG.info("Step 2: extracting CSVs to parquet")
    usaspending.extract_all(run_cfg.fiscal_years)


def step_bmf(run_manifest: manifest_mod.RunManifest) -> None:
    LOG.info("Step 3: IRS BMF download + parse")
    parquet, records = irs_bmf.acquire_and_prepare()
    run_manifest.irs_bmf_files = [asdict(r) for r in records]
    LOG.info("BMF parquet at %s", parquet)


def step_sam(run_manifest: manifest_mod.RunManifest) -> None:
    LOG.info("Step 3b: SAM entity extract")
    result = sam.parse_extract()
    if result is None:
        run_manifest.notes.append("SAM extract not provided; UEI->EIN backfill skipped")
    else:
        _, rec = result
        run_manifest.sam_extract = asdict(rec)


def step_match(run_manifest: manifest_mod.RunManifest) -> None:
    LOG.info("Step 4: build recipient_match")
    txn = _read_transactions(columns=_MATCH_COLUMNS)
    bmf = _read_bmf()
    sam_df = _read_sam()
    match_df, stats, review = recipient_match.build_recipient_match(txn, bmf, sam_df)
    recipient_match.write_outputs(match_df, stats, review)
    run_manifest.match_stats = asdict(stats)

    coverage = recipient_match.coverage_report(match_df, txn)
    coverage.to_csv(config.PROCESSED / "match_coverage_report.csv", index=False)
    flagged = int(coverage["flag_under_90pct"].sum())
    if flagged:
        run_manifest.notes.append(f"{flagged} agency-FY cells under 90% dollar-weighted match")


def step_classify(run_cfg: config.RunConfig, run_manifest: manifest_mod.RunManifest) -> None:
    LOG.info("Step 5: category classification")
    txn = _read_transactions(columns=_CLASSIFY_COLUMNS)
    match_df = _read_match_table()

    aha_eins = _load_optional_eins("aha_eins.txt")
    hrsa_eins = _load_optional_eins("hrsa_uds_eins.txt")
    ipeds_eins = _load_optional_eins("ipeds_eins.txt")
    nces_names = _load_optional_lines("nces_school_district_names.txt")

    classified, stats = categorize.classify(
        txn, match_df,
        aha_eins=aha_eins, hrsa_eins=hrsa_eins,
        ipeds_eins=ipeds_eins, nces_school_district_names=nces_names,
        priority=run_cfg.classification_priority,
        intl_q_dollar_floor=run_cfg.intl_q_series_floor,
    )
    classified.to_parquet(config.PROCESSED / "transactions_classified.parquet", index=False)
    run_manifest.category_stats = asdict(stats)


def _load_optional_eins(name: str) -> list[str]:
    p = config.RAW_REFERENCE / name
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_optional_lines(name: str) -> list[str]:
    return _load_optional_eins(name)


def step_tables(run_cfg: config.RunConfig, run_manifest: manifest_mod.RunManifest) -> None:
    LOG.info("Step 6: analytic table assembly")
    classified = _read_classified_transactions()
    match_df = _read_match_table()

    txn_table = tables_mod.build_transactions_table(
        classified, match_df,
        deflator=run_cfg.deflator, base_fy=run_cfg.deflator_base_fy,
    )
    awards_table = tables_mod.build_awards_table(classified)
    paths = tables_mod.write_outputs(txn_table, awards_table)
    manifest_mod.add_output(run_manifest, paths.transactions, "assistance_txn_501c3")
    manifest_mod.add_output(run_manifest, paths.awards, "assistance_awards_501c3")


def step_exhibits(run_manifest: manifest_mod.RunManifest) -> None:
    LOG.info("Step 7: aggregations and exhibits")
    txn, awd = _read_analytic_tables()
    artifacts = exhibits_mod.produce_all(txn, awd)
    for art in artifacts:
        manifest_mod.add_output(run_manifest, art.path,
                                f"{art.name}/{art.panel or 'Total'}")


def step_qa(run_manifest: manifest_mod.RunManifest) -> None:
    LOG.info("Step 8: QA checks")
    txn, awd = _read_analytic_tables()
    match_df = _read_match_table()
    qa_checks.run(txn, awd, match_df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="USASpending FY22-FY25 501(c)(3) pipeline")
    p.add_argument("--all", action="store_true", help="Run every step end-to-end")
    p.add_argument("--init", action="store_true", help="Step 1: create directories")
    p.add_argument("--acquire", action="store_true", help="Step 2: USASpending downloads")
    p.add_argument("--bmf", action="store_true", help="Step 3: IRS BMF")
    p.add_argument("--sam", action="store_true", help="Step 3b: SAM extract parse")
    p.add_argument("--match", action="store_true", help="Step 4: recipient match")
    p.add_argument("--classify", action="store_true", help="Step 5: category classification")
    p.add_argument("--tables", action="store_true", help="Step 6: analytic tables")
    p.add_argument("--exhibits", action="store_true", help="Step 7: exhibits")
    p.add_argument("--qa", action="store_true", help="Step 8: QA checks")
    p.add_argument("--deflator", default=config.DEFAULT_DEFLATOR, choices=["CPI-U", "GDP"])
    p.add_argument("--download-workers", type=int, default=None,
                   help="Parallel download workers. bulk_download default: one per FY (4). archive default: 16.")
    p.add_argument("--acquire-source", choices=["bulk_download", "archive", "manual"],
                   default="archive",
                   help="Where to pull USASpending data from. 'archive' is the pre-generated per-agency zip set "
                        "(fast, ~30-day snapshot lag). 'bulk_download' is the POST-and-poll API (freshest, slow). "
                        "'manual' extracts zips that the operator has staged under raw/usaspending/manual/fy{N}/ "
                        "(no network required - use this when USAspending is down or unreachable).")
    p.add_argument("--priority", nargs=4,
                   default=list(config.RunConfig().classification_priority),
                   metavar=("P1", "P2", "P3", "P4"),
                   help="Override classification hierarchy (default: international hospital educational core)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    run_cfg = config.RunConfig(
        deflator=args.deflator,
        classification_priority=tuple(args.priority),
    )

    do_all = args.all
    do_init = args.init or do_all
    do_acquire = args.acquire or do_all
    do_bmf = args.bmf or do_all
    do_sam = args.sam or do_all
    do_match = args.match or do_all
    do_classify = args.classify or do_all
    do_tables = args.tables or do_all
    do_exhibits = args.exhibits or do_all
    do_qa = args.qa or do_all

    if do_init:
        config.ensure_dirs()
        LOG.info("Step 1: project directories ready under %s", config.PROJECT_ROOT)

    refs = load_reference_lists()
    rm = manifest_mod.new_manifest(run_cfg, refs.rules_version_hash)

    try:
        if do_acquire:
            step_acquire(run_cfg, rm,
                         download_workers=args.download_workers,
                         source=args.acquire_source)
        if do_bmf:
            step_bmf(rm)
        if do_sam:
            step_sam(rm)
        if do_match:
            step_match(rm)
        if do_classify:
            step_classify(run_cfg, rm)
        if do_tables:
            step_tables(run_cfg, rm)
        if do_exhibits:
            step_exhibits(rm)
        if do_qa:
            step_qa(rm)
    finally:
        manifest_path = manifest_mod.write(rm)
        LOG.info("Run manifest -> %s", manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
