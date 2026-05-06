"""End-to-end smoke test on a real-data sample.

Loads ~250k transactions from FY24 plus the full BMF, then runs match ->
classify -> tables -> exhibits -> qa in a temp directory. Surfaces real-data
bugs (missing columns, type coercion, edge cases) without waiting on the
full 26M-row run.

Usage:
    python tests/integration_sample.py [--rows 250000] [--fy 2024]
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG = logging.getLogger("integration_sample")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=250_000)
    p.add_argument("--fy", type=int, default=2024)
    args = p.parse_args()

    t0 = time.time()

    LOG.info("Loading sample: first %d rows of transactions_fy%d.parquet", args.rows, args.fy)
    txn = pd.read_parquet(ROOT / f"interim/transactions_fy{args.fy}.parquet")
    LOG.info("FY%d total rows: %d", args.fy, len(txn))
    txn = txn.head(args.rows).copy()
    LOG.info("Sample size: %d rows, %d cols", len(txn), len(txn.columns))

    LOG.info("Loading BMF")
    bmf = pd.read_parquet(ROOT / "interim/bmf_501c3_current.parquet")
    LOG.info("BMF rows: %d", len(bmf))

    # --- Step 4: match
    from src.match.recipient_match import build_recipient_match
    LOG.info("Step 4: build_recipient_match")
    match_df, stats, review = build_recipient_match(txn, bmf)
    LOG.info("Match stats: %s", stats)
    LOG.info("Tier-2 review queue size: %d", len(review))

    # --- Step 5: classify
    from src.classify.categorize import classify
    LOG.info("Step 5: classify")
    classified, cstats = classify(txn, match_df)
    LOG.info("Classify stats: %s", cstats)

    # --- Step 6: tables
    from src.analytic.tables import build_transactions_table, build_awards_table
    LOG.info("Step 6: build analytic tables")
    txn_tbl = build_transactions_table(classified, match_df)
    awards = build_awards_table(classified)
    LOG.info("txn_tbl rows: %d, cols: %d", len(txn_tbl), len(txn_tbl.columns))
    LOG.info("awards rows: %d, cols: %d", len(awards), len(awards.columns))

    # --- Step 7: exhibits
    from src.aggregate.exhibits import produce_all
    out = Path(tempfile.mkdtemp(prefix="usasp_int_"))
    LOG.info("Step 7: exhibits -> %s", out)
    arts = produce_all(txn_tbl, awards, out_dir=out)
    LOG.info("Produced %d exhibits", len(arts))

    # --- Step 8: QA
    from src.qa.checks import run as run_qa
    LOG.info("Step 8: QA")
    qa = run_qa(txn_tbl, awards, match_df, out_dir=out / "qa")

    LOG.info("Sample run complete in %.1fs", time.time() - t0)
    LOG.info("Exhibits dir: %s", out)


if __name__ == "__main__":
    main()
