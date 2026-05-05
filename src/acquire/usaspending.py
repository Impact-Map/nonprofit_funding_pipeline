"""USASpending Custom Award Downloads client (methodology Step 2).

The Custom Award Downloads endpoint enforces a per-request size cap, so the
methodology requires splitting by FY (one POST per FY). Each call returns a
status URL that we poll until the file is ready, then download the zip and
record the API response headers (refresh date) and the SHA-256 of the zip
into the run manifest.

After download, the prime-award transactions and prime-award summaries CSVs
are extracted, filtered to assistance award_type_codes, and persisted as
parquet under /interim.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from tqdm import tqdm

from .. import config

LOG = logging.getLogger(__name__)

BASE_URL = "https://api.usaspending.gov"
BULK_DOWNLOAD_URL = f"{BASE_URL}/api/v2/bulk_download/awards/"
STATUS_URL = f"{BASE_URL}/api/v2/bulk_download/status/"

# FY -> action_date window (Oct 1 prior calendar year through Sep 30 of FY).
FY_DATE_RANGES: dict[int, tuple[str, str]] = {
    2022: ("2021-10-01", "2022-09-30"),
    2023: ("2022-10-01", "2023-09-30"),
    2024: ("2023-10-01", "2024-09-30"),
    2025: ("2024-10-01", "2025-09-30"),
}


@dataclass
class DownloadRecord:
    """Recorded artifact metadata for the run manifest."""
    fy: int
    request_payload: dict
    status_url: str
    zip_path: str
    zip_sha256: str
    api_refresh_date: str | None
    completed_at: str
    file_count: int


def build_request_payload(fy: int, agencies: list[dict] | None = None) -> dict:
    """Construct the JSON payload for /api/v2/bulk_download/awards/.

    The methodology's example payload (Section 10.2) shows `"agencies": []`,
    but the live API rejects that with HTTP 422:
        {"detail": "Field 'filters|agencies' value '[]' is below min '1' items"}
    The `agencies` filter is therefore *omitted* when no agencies are
    requested ("all agencies" is the default behavior). When supplied, each
    item must be an object: {"type": "awarding"|"funding",
    "tier": "toptier"|"subtier", "name": "<agency name>"}.
    """
    start, end = FY_DATE_RANGES[fy]
    filters: dict = {
        "prime_award_types": list(config.ASSISTANCE_AWARD_TYPE_CODES),
        "date_type": "action_date",
        "date_range": {"start_date": start, "end_date": end},
    }
    if agencies:
        filters["agencies"] = agencies
    return {"filters": filters, "file_format": "csv"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def submit_bulk_download(payload: dict, session: requests.Session | None = None) -> dict:
    """Submit a bulk-download request. On HTTP error, log the API's response
    body before re-raising so 4xx validation failures are debuggable.
    """
    sess = session or requests.Session()
    resp = sess.post(BULK_DOWNLOAD_URL, json=payload, timeout=60)
    if not resp.ok:
        body = resp.text[:2000]
        LOG.error(
            "USASpending bulk download POST failed: %s %s\nrequest=%s\nresponse_body=%s",
            resp.status_code, resp.reason, json.dumps(payload), body,
        )
    resp.raise_for_status()
    return resp.json()


def poll_status(file_name_or_status_url: str,
                session: requests.Session | None = None,
                poll_interval_s: int = 30,
                timeout_s: int = 60 * 60 * 4) -> dict:
    """Poll the status endpoint until the job is `finished` or `failed`.

    Accepts either the bare file_name (preferred per API) or a full status URL.
    """
    sess = session or requests.Session()
    deadline = time.time() + timeout_s
    if file_name_or_status_url.startswith("http"):
        url = file_name_or_status_url
    else:
        url = f"{STATUS_URL}?file_name={file_name_or_status_url}"
    while time.time() < deadline:
        r = sess.get(url, timeout=60)
        r.raise_for_status()
        body = r.json()
        status = body.get("status")
        if status == "finished":
            return body
        if status == "failed":
            raise RuntimeError(f"USASpending download failed: {body}")
        LOG.info("USASpending download status=%s, sleeping %ds", status, poll_interval_s)
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Bulk download did not finish within {timeout_s}s")


def stream_download(url: str, dest: Path, session: requests.Session | None = None) -> tuple[Path, dict[str, str]]:
    """Stream a URL to disk; return (path, response_headers)."""
    sess = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with sess.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return dest, dict(r.headers)


def download_fy(fy: int,
                out_dir: Path | None = None,
                session: requests.Session | None = None) -> DownloadRecord:
    """End-to-end: submit, poll, download zip for one FY."""
    out_dir = out_dir or config.RAW_USASPENDING / f"fy{fy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()

    payload = build_request_payload(fy)
    LOG.info("Submitting USASpending bulk download for FY%d", fy)
    submit = submit_bulk_download(payload, sess)
    status_url = submit.get("status_url") or submit.get("url") or ""
    file_name = submit.get("file_name", "")
    LOG.info("FY%d submitted: file_name=%s status_url=%s", fy, file_name, status_url)

    # Persist the submit response for audit.
    (out_dir / "submit_response.json").write_text(json.dumps(submit, indent=2))

    final = poll_status(file_name or status_url, sess)
    download_url = final.get("file_url") or final.get("url")
    if not download_url:
        raise RuntimeError(f"Bulk download finished but no file_url: {final}")

    zip_dest = out_dir / Path(file_name or download_url.split("/")[-1]).name
    if not zip_dest.suffix:
        zip_dest = zip_dest.with_suffix(".zip")
    zip_path, headers = stream_download(download_url, zip_dest, sess)

    api_refresh = headers.get("Last-Modified") or final.get("update_date")
    file_count = 0
    with zipfile.ZipFile(zip_path) as zf:
        file_count = len(zf.namelist())

    record = DownloadRecord(
        fy=fy,
        request_payload=payload,
        status_url=status_url,
        zip_path=str(zip_path),
        zip_sha256=_sha256(zip_path),
        api_refresh_date=api_refresh,
        completed_at=datetime.now(timezone.utc).isoformat(),
        file_count=file_count,
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(record), indent=2))
    return record


def download_all(fiscal_years: tuple[int, ...] = config.FISCAL_YEARS) -> list[DownloadRecord]:
    sess = requests.Session()
    records: list[DownloadRecord] = []
    for fy in fiscal_years:
        records.append(download_fy(fy, session=sess))
    return records


# ---------------------------------------------------------------------------
# Extraction: turn the raw zips into parquet under /interim.
# ---------------------------------------------------------------------------

PRIME_TXN_NAME_FRAGMENTS = ("PrimeTransactions", "PrimeAwardTransactions", "Transactions")
PRIME_AWARD_NAME_FRAGMENTS = ("PrimeAwardSummaries", "Awards", "PrimeAwards")


def _classify_member(name: str) -> str | None:
    base = Path(name).name
    if not base.lower().endswith(".csv"):
        return None
    low = base.lower()
    if "transaction" in low:
        return "transactions"
    if "award" in low:
        return "awards"
    return None


def iter_csv_members(zip_path: Path) -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            kind = _classify_member(member)
            if kind:
                yield kind, member


def extract_zip_to_parquet(zip_path: Path,
                           out_dir: Path,
                           fy: int,
                           award_type_codes: tuple[str, ...] = config.ASSISTANCE_AWARD_TYPE_CODES) -> dict[str, Path]:
    """Read each CSV in the zip in chunks; write filtered parquet per kind."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    with zipfile.ZipFile(zip_path) as zf:
        for kind, member in iter_csv_members(zip_path):
            out_path = out_dir / f"{kind}_fy{fy}.parquet"
            LOG.info("Extracting %s -> %s", member, out_path)
            chunks: list[pd.DataFrame] = []
            with zf.open(member) as raw:
                # The zip entry is a binary stream; pandas needs a text wrapper.
                buf = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                for chunk in pd.read_csv(
                    buf, chunksize=200_000, low_memory=False, dtype=str,
                ):
                    if "award_type_code" in chunk.columns:
                        chunk = chunk[chunk["award_type_code"].isin(award_type_codes)]
                    chunks.append(chunk)
            if not chunks:
                continue
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(out_path, index=False)
            written[kind] = out_path
    return written


def extract_all(fiscal_years: tuple[int, ...] = config.FISCAL_YEARS) -> dict[int, dict[str, Path]]:
    results: dict[int, dict[str, Path]] = {}
    for fy in fiscal_years:
        fy_dir = config.RAW_USASPENDING / f"fy{fy}"
        zips = sorted(fy_dir.glob("*.zip"))
        if not zips:
            LOG.warning("No zip in %s for FY%d; skipping", fy_dir, fy)
            continue
        # If multiple zips landed (e.g., a re-run), use the latest.
        zip_path = zips[-1]
        results[fy] = extract_zip_to_parquet(zip_path, config.INTERIM, fy)
    return results
