"""Award Data Archive client - faster alternative to bulk_download/awards.

USAspending publishes pre-generated per-agency, per-FY zips at
https://files.usaspending.gov/award_data_archive/ . They use the naming
convention `FY{YYYY}_{agency_code}_Assistance_Full_{YYYYMMDD}.zip` and are
refreshed roughly monthly. Because the files already exist on S3, no job
queue or polling is involved - downloads start immediately and parallelize
trivially.

Trade-off vs. bulk_download/awards:
  - Archive snapshot lags the live data by up to ~30 days; the snapshot date
    is in the filename and is recorded in the run manifest.
  - Archive zips are "Full" - all assistance award_type_codes - so we still
    apply the 02..11 award_type_code filter on extraction.
  - Archive coverage is per top-tier agency; the pipeline downloads every
    available agency code per FY, then concatenates.

For FY25 (in flight), the archive snapshot is sufficient for headline numbers
but will trail the live API by the snapshot age. If freshest possible FY25
numbers matter, use bulk_download for FY25 and archive for FY22-FY24.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .. import config

LOG = logging.getLogger(__name__)


def _resilient_session() -> requests.Session:
    """Session with exponential-backoff retries on transient network errors.

    The S3 endpoint hosting the archive sometimes closes idle keep-alive
    connections mid-pagination (RemoteDisconnected) and occasionally returns
    503 under load. urllib3's Retry handles both. We retry on connect, read,
    and on 5xx with backoff up to ~60s.
    """
    sess = requests.Session()
    retry = Retry(
        total=8, connect=5, read=5, status=5,
        backoff_factor=1.5,           # 1.5s, 3s, 6s, 12s, 24s, 48s, ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

S3_BASE = "https://files.usaspending.gov/award_data_archive/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
ARCHIVE_PATTERN = re.compile(
    r"^FY(?P<fy>\d{4})_(?P<agency>\d+)_Assistance_Full_(?P<date>\d{8})\.zip$"
)


@dataclass
class ArchiveEntry:
    fy: int
    agency_code: str
    snapshot_date: str  # YYYYMMDD
    key: str
    size: int

    @property
    def url(self) -> str:
        return S3_BASE + self.key


@dataclass
class ArchiveDownloadRecord:
    fy: int
    files: int
    total_bytes: int
    snapshot_dates: list[str]
    completed_at: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_archive_bucket(session: requests.Session | None = None) -> list[ArchiveEntry]:
    """Enumerate the S3 bucket listing (paginated, 1000 keys per page).

    Uses `params=` so requests URL-encodes the marker properly - bucket keys
    like `FY(All)_086_Contracts_Delta_20260406.zip` contain parens that some
    HTTP parsers reject if pasted raw into a query string.
    """
    sess = session or _resilient_session()
    entries: list[ArchiveEntry] = []
    marker: str | None = None
    page = 0
    while True:
        params = {"marker": marker} if marker else None
        resp = sess.get(S3_BASE, params=params, timeout=(15, 90))
        resp.raise_for_status()
        page += 1
        root = ET.fromstring(resp.content)
        contents = root.findall(f"{S3_NS}Contents")
        for c in contents:
            key = c.findtext(f"{S3_NS}Key") or ""
            size = int(c.findtext(f"{S3_NS}Size") or 0)
            m = ARCHIVE_PATTERN.match(key)
            if not m:
                continue
            entries.append(ArchiveEntry(
                fy=int(m["fy"]), agency_code=m["agency"],
                snapshot_date=m["date"], key=key, size=size,
            ))
        truncated = (root.findtext(f"{S3_NS}IsTruncated") == "true")
        LOG.debug("bucket listing page=%d entries=%d truncated=%s",
                  page, len(contents), truncated)
        if not truncated:
            break
        # Prefer the server-supplied NextMarker if present; fall back to the
        # last Key on the page (S3 spec for non-versioned listings).
        marker = root.findtext(f"{S3_NS}NextMarker") or (
            contents[-1].findtext(f"{S3_NS}Key") if contents else None
        )
        if not marker:
            break
    LOG.info("Archive bucket: %d total entries across %d page(s)", len(entries), page)
    return entries


def select_latest(entries: list[ArchiveEntry],
                  fiscal_years: tuple[int, ...]) -> list[ArchiveEntry]:
    """Keep the newest snapshot per (fy, agency_code) within target FYs."""
    latest: dict[tuple[int, str], ArchiveEntry] = {}
    for e in entries:
        if e.fy not in fiscal_years:
            continue
        k = (e.fy, e.agency_code)
        if k not in latest or e.snapshot_date > latest[k].snapshot_date:
            latest[k] = e
    return sorted(latest.values(), key=lambda e: (e.fy, e.agency_code))


def _download_one(entry: ArchiveEntry, out_dir: Path,
                  session: requests.Session,
                  position: int | None = None,
                  max_attempts: int = 4) -> Path:
    """Download one zip with file-level retry on mid-stream failures.

    urllib3's Retry adapter only re-issues a request that hasn't started
    streaming. If S3 closes the connection mid-body (RemoteDisconnected
    during iter_content), we catch it here and re-issue the GET. The .part
    file is rewritten from scratch each attempt; backoff doubles each try.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / entry.key
    if dest.exists() and dest.stat().st_size == entry.size:
        return dest  # already downloaded
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with session.get(entry.url, stream=True, timeout=(15, 120)) as r:
                r.raise_for_status()
                with tmp.open("wb") as f, tqdm(
                    total=entry.size, unit="B", unit_scale=True, desc=entry.key,
                    leave=False, position=position,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            tmp.rename(dest)
            return dest
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout) as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            LOG.warning("download attempt %d/%d failed for %s: %s; retrying in %ds",
                        attempt, max_attempts, entry.key, exc, wait)
            tmp.unlink(missing_ok=True)
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {entry.key} after {max_attempts} attempts") from last_exc


def download_all(fiscal_years: tuple[int, ...] = config.FISCAL_YEARS,
                 max_workers: int = 16,
                 out_dir: Path | None = None
                 ) -> tuple[list[Path], list[ArchiveDownloadRecord]]:
    """Enumerate, dedupe, and download every per-agency archive in parallel.

    Returns (downloaded_paths, per_fy_records). Records summarize each FY's
    downloaded file count, total bytes, and the set of snapshot dates seen
    (multiple agencies can publish on different days within a refresh cycle).
    """
    out_dir = out_dir or (config.RAW_USASPENDING / "archive")
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Enumerating Award Data Archive bucket")
    sess = _resilient_session()
    entries = list_archive_bucket(sess)
    selected = select_latest(entries, tuple(fiscal_years))
    LOG.info("Archive plan: %d files across FY %s (latest snapshots)",
             len(selected), list(fiscal_years))

    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix="usasp-archive") as pool:
        # Each thread gets its own session; tqdm position cycles within pool size.
        futures = {
            pool.submit(_download_one, e, out_dir / f"fy{e.fy}",
                        _resilient_session(), i % max_workers): e
            for i, e in enumerate(selected)
        }
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                downloaded.append(fut.result())
            except Exception:
                LOG.exception("Failed to download %s", entry.key)
                raise

    # Per-FY summary record
    by_fy: dict[int, list[ArchiveEntry]] = {}
    for e in selected:
        by_fy.setdefault(e.fy, []).append(e)
    records = [
        ArchiveDownloadRecord(
            fy=fy,
            files=len(es),
            total_bytes=sum(e.size for e in es),
            snapshot_dates=sorted({e.snapshot_date for e in es}),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        for fy, es in sorted(by_fy.items())
    ]
    return downloaded, records


# ---------------------------------------------------------------------------
# Extraction: each per-agency zip contains one or more CSVs at the top level.
# We read them in chunks, filter to assistance award_type_codes, and append
# to a single parquet per FY (transactions / awards).
# ---------------------------------------------------------------------------


_TXN_HINT = "transaction"
_AWD_HINT = "award"


def _classify_csv(name: str) -> str | None:
    """Classify a member CSV as transactions vs award-summaries.

    Archive zips often contain a single transaction-grain CSV
    `FY{YYYY}_{agency}_Assistance_Full_{date}_{N}.csv` with no obvious
    "transaction"/"award" hint. Treat the un-hinted CSVs as transactions and
    fall through.
    """
    base = Path(name).name.lower()
    if not base.endswith(".csv"):
        return None
    if "subaward" in base:
        return None  # methodology Section 1.2: sub-awards out of scope
    if _AWD_HINT + "summar" in base or "awardsummar" in base:
        return "awards"
    # The standard archive CSV is transaction-grain.
    return "transactions"


# ---------------------------------------------------------------------------
# Schema bridge: archive CSVs use the Public Profile column names; the rest of
# the pipeline (and the methodology Section 11 schema) uses the bulk_download
# / DAIMS API codes. Rename columns on extraction so downstream code is
# identical regardless of acquisition source.
# ---------------------------------------------------------------------------

ARCHIVE_TO_PIPELINE_COLUMNS: dict[str, str] = {
    # Identity
    "assistance_transaction_unique_key": "transaction_id",
    "assistance_award_unique_key":       "award_id_unique",
    # Award-type / action-type
    "assistance_type_code":              "award_type_code",
    "action_type_code":                  "action_type",
    # Recipient
    "business_types_code":               "recipient_business_types",
    # CFDA / Assistance Listing
    "cfda_number":                       "assistance_listing_number",
    "cfda_title":                        "assistance_listing_title",
    # action_date, federal_action_obligation, recipient_uei, recipient_name,
    # recipient_state_code, awarding_agency_name, awarding_sub_agency_name,
    # primary_place_of_performance_country_code,
    # total_outlayed_amount_for_overall_award - already match DAIMS names.
}


def _rename_to_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Rename archive columns to the pipeline's expected names. No-op on
    columns already in the canonical form."""
    overlap = {src: dst for src, dst in ARCHIVE_TO_PIPELINE_COLUMNS.items()
               if src in df.columns and dst not in df.columns}
    if overlap:
        df = df.rename(columns=overlap)
    # `generated_unique_award_id` is referenced as a fallback in tables.py;
    # the archive provides the same value as award_id_unique.
    if "award_id_unique" in df.columns and "generated_unique_award_id" not in df.columns:
        df["generated_unique_award_id"] = df["award_id_unique"]
    return df


def extract_archives_to_parquet(
    fiscal_years: tuple[int, ...] = config.FISCAL_YEARS,
    archive_root: Path | None = None,
    out_dir: Path | None = None,
    award_type_codes: tuple[str, ...] = config.ASSISTANCE_AWARD_TYPE_CODES,
) -> dict[int, dict[str, Path]]:
    """Concatenate every per-agency zip for an FY into a single parquet pair."""
    archive_root = archive_root or (config.RAW_USASPENDING / "archive")
    out_dir = out_dir or config.INTERIM
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[int, dict[str, Path]] = {}
    for fy in fiscal_years:
        fy_dir = archive_root / f"fy{fy}"
        zips = sorted(fy_dir.glob("FY*_Assistance_Full_*.zip"))
        if not zips:
            LOG.warning("No archive zips under %s; skipping FY%d", fy_dir, fy)
            continue

        txn_chunks: list[pd.DataFrame] = []
        awd_chunks: list[pd.DataFrame] = []
        LOG.info("FY%d: extracting %d agency archive zips", fy, len(zips))
        for zp in tqdm(zips, desc=f"FY{fy} extract", leave=False):
            with zipfile.ZipFile(zp) as zf:
                for member in zf.namelist():
                    kind = _classify_csv(member)
                    if not kind:
                        continue
                    with zf.open(member) as raw:
                        buf = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                        for chunk in pd.read_csv(
                            buf, chunksize=200_000, low_memory=False, dtype=str,
                        ):
                            chunk = _rename_to_pipeline(chunk)
                            if "award_type_code" in chunk.columns:
                                chunk = chunk[chunk["award_type_code"].isin(award_type_codes)]
                            (txn_chunks if kind == "transactions" else awd_chunks).append(chunk)

        out: dict[str, Path] = {}
        if txn_chunks:
            out["transactions"] = out_dir / f"transactions_fy{fy}.parquet"
            pd.concat(txn_chunks, ignore_index=True).to_parquet(out["transactions"], index=False)
        if awd_chunks:
            out["awards"] = out_dir / f"awards_fy{fy}.parquet"
            pd.concat(awd_chunks, ignore_index=True).to_parquet(out["awards"], index=False)
        written[fy] = out
    return written
