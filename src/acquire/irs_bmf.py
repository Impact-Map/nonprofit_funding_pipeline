"""IRS Exempt Organizations Business Master File acquisition (Step 3).

The IRS publishes the EO BMF as four regional CSVs at irs.gov. The methodology
calls for downloading the current monthly release plus four historical FY-end
snapshots (Sep 2022, Sep 2023, Sep 2024, Sep 2025) so each transaction can be
evaluated against the BMF that was current at its action_date.

Filter to SUBSECTION = '03' (501(c)(3)) on read; persist as parquet.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .. import config

LOG = logging.getLogger(__name__)

# IRS EO BMF download URLs are stable file names under data.irs.gov.
BMF_REGIONS = ("eo1", "eo2", "eo3", "eo4")
BMF_BASE_URL = "https://www.irs.gov/pub/irs-soi/{region}.csv"

# Columns we rely on downstream. The BMF has ~30 columns; keeping the schema
# here makes the pipeline robust to silent IRS additions.
BMF_KEEP_COLUMNS = [
    "EIN", "NAME", "STREET", "CITY", "STATE", "ZIP",
    "SUBSECTION", "AFFILIATION", "CLASSIFICATION", "RULING", "DEDUCTIBILITY",
    "FOUNDATION", "ACTIVITY", "ORGANIZATION", "STATUS", "TAX_PERIOD",
    "ASSET_CD", "INCOME_CD", "FILING_REQ_CD", "PF_FILING_REQ_CD",
    "ACCT_PD", "ASSET_AMT", "INCOME_AMT", "REVENUE_AMT", "NTEE_CD", "SORT_NAME",
]


@dataclass
class BMFFileRecord:
    region: str
    url: str
    local_path: str
    sha256: str
    bytes: int
    downloaded_at: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_region(region: str, out_dir: Path,
                    session: requests.Session | None = None) -> BMFFileRecord:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = BMF_BASE_URL.format(region=region)
    dest = out_dir / f"{region}.csv"
    sess = session or requests.Session()
    LOG.info("Downloading IRS BMF region %s", region)
    with sess.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return BMFFileRecord(
        region=region,
        url=url,
        local_path=str(dest),
        sha256=_sha256(dest),
        bytes=dest.stat().st_size,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )


def download_all_regions(out_dir: Path | None = None) -> list[BMFFileRecord]:
    out_dir = out_dir or config.RAW_IRS_BMF / "current"
    sess = requests.Session()
    return [download_region(r, out_dir, sess) for r in BMF_REGIONS]


def parse_and_filter(in_dir: Path | None = None,
                     out_path: Path | None = None) -> Path:
    """Concatenate the four regional CSVs, filter to SUBSECTION='03', write parquet."""
    in_dir = in_dir or (config.RAW_IRS_BMF / "current")
    out_path = out_path or (config.INTERIM / "bmf_501c3_current.parquet")
    frames: list[pd.DataFrame] = []
    for region in BMF_REGIONS:
        p = in_dir / f"{region}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Expected BMF region file: {p}")
        # IRS publishes BMF as comma-separated, all-text. dtype=str preserves
        # leading zeros on EIN, ZIP, SUBSECTION codes.
        df = pd.read_csv(p, dtype=str, low_memory=False)
        frames.append(df)
    bmf = pd.concat(frames, ignore_index=True)
    LOG.info("Concatenated BMF rows: %d", len(bmf))
    bmf = bmf[bmf["SUBSECTION"].astype(str).str.zfill(2) == "03"].copy()
    LOG.info("BMF rows after SUBSECTION=03 filter: %d", len(bmf))
    keep = [c for c in BMF_KEEP_COLUMNS if c in bmf.columns]
    bmf = bmf[keep]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bmf.to_parquet(out_path, index=False)
    return out_path


def acquire_and_prepare(out_path: Path | None = None) -> tuple[Path, list[BMFFileRecord]]:
    records = download_all_regions()
    parquet = parse_and_filter(out_path=out_path)
    return parquet, records
