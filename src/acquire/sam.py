"""SAM.gov entity-registration extract (methodology Section 2.3).

The SAM.gov public entity-extract is the UEI <-> EIN <-> legal-business-name
crosswalk used to backfill EIN where USAspending does not expose it. The
extract is a daily file behind a SAM.gov account; this module loads a local
extract that has been placed under raw/sam/, and persists a slim parquet with
just the columns we need.

If the user has not yet placed a SAM extract, the pipeline runs without it
(EIN-first matching falls back to UEI->recipient_profile and name+state).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config

LOG = logging.getLogger(__name__)

# SAM Entity Public extract column names (subset). The actual extract has 100+
# columns; the load is column-list-driven so unrelated columns are dropped.
SAM_KEEP_COLUMNS = [
    "UEI", "TIN", "LEGAL_BUSINESS_NAME", "DBA_NAME",
    "PHYSICAL_ADDRESS_STATE_OR_PROVINCE", "PHYSICAL_ADDRESS_COUNTRY_CODE",
    "REGISTRATION_DATE", "EXPIRATION_DATE", "PURPOSE_OF_REGISTRATION",
]


@dataclass
class SAMExtractRecord:
    local_path: str
    sha256: str
    bytes: int
    rows: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_extract(extract_path: Path | None = None,
                  out_path: Path | None = None) -> tuple[Path, SAMExtractRecord] | None:
    """Read a SAM extract CSV and persist a slim parquet.

    Returns None if no extract is available. The match step in this case will
    skip the SAM-based EIN backfill and rely on UEI->recipient_profile +
    deterministic/probabilistic name matching only.
    """
    extract_path = extract_path or _autodetect_extract(config.RAW_SAM)
    if extract_path is None:
        LOG.info("No SAM extract found under %s; SAM backfill disabled", config.RAW_SAM)
        return None
    out_path = out_path or (config.INTERIM / "sam_entity.parquet")

    LOG.info("Reading SAM extract %s", extract_path)
    df = pd.read_csv(extract_path, dtype=str, low_memory=False)
    keep = [c for c in SAM_KEEP_COLUMNS if c in df.columns]
    df = df[keep]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    rec = SAMExtractRecord(
        local_path=str(extract_path),
        sha256=_sha256(extract_path),
        bytes=extract_path.stat().st_size,
        rows=len(df),
    )
    return out_path, rec


def _autodetect_extract(directory: Path) -> Path | None:
    for ext in ("*.csv", "*.zip"):
        for p in directory.glob(ext):
            return p
    return None
