"""Manual / offline acquisition mode.

When USAspending's API and S3 endpoints are unavailable (or behind a network
the operator can't reach from the pipeline host), the user can download zips
through the public web UI - https://www.usaspending.gov/download_center -
or copy them off another machine, drop them under
`raw/usaspending/manual/fy{YYYY}/` and run

    python3 pipeline.py --acquire --acquire-source manual

which extracts in place, no network required.

Conventions:
  - One sub-directory per fiscal year: `manual/fy2022/`, `manual/fy2023/`, ...
  - Inside each FY directory, drop one or more zips. The pipeline accepts:
      * Custom Award Downloads (bulk_download/awards) zips, named like
        `All_PrimeTransactions_YYYY-MM-DD_*.zip`. These use the DAIMS / API
        column codes (award_type_code, action_type, ...).
      * Award Data Archive zips, named like
        `FY{YYYY}_{agency}_Assistance_Full_{date}.zip`. These use the
        Public-Profile column names; the schema bridge in award_archive.py
        renames them to the canonical pipeline schema.
  - Multiple zips per FY are concatenated. Sub-award zips are skipped.
  - The award_type_code filter (02-11) is re-applied during extraction in
    case the zip contains contracts or other types.

A summary file `manual/manifest.json` is written so the run manifest captures
what local files contributed to the parquet output (file path, bytes, sha256,
fiscal year). That preserves Section 9's reproducibility contract.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from .award_archive import extract_archives_to_parquet

LOG = logging.getLogger(__name__)

MANUAL_ROOT = config.RAW_USASPENDING / "manual"


@dataclass
class ManualFileRecord:
    fy: int
    path: str
    bytes: int
    sha256: str


@dataclass
class ManualAcquireRecord:
    fy: int
    files: int
    total_bytes: int
    completed_at: str
    file_records: list[dict]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(fiscal_years: tuple[int, ...] = config.FISCAL_YEARS,
             root: Path | None = None) -> dict[int, list[Path]]:
    """Map fy -> list of zip paths under manual/fy{N}/.

    Returns only FYs that have at least one zip. FYs with no zips are silently
    omitted; the caller decides whether to fail-fast or proceed with whatever
    is available.
    """
    root = root or MANUAL_ROOT
    out: dict[int, list[Path]] = {}
    for fy in fiscal_years:
        fy_dir = root / f"fy{fy}"
        if not fy_dir.exists():
            continue
        zips = sorted(p for p in fy_dir.glob("*.zip") if p.is_file())
        if zips:
            out[fy] = zips
    return out


def _scaffold_message(missing: list[int]) -> str:
    return (
        "No zips found for FY " + ", ".join(map(str, missing)) + ".\n"
        f"Drop one or more USAspending zips into:\n"
        + "\n".join(f"  {MANUAL_ROOT / f'fy{fy}'}" for fy in missing)
        + "\nAccepted: Custom Award Downloads or Award Data Archive zips."
    )


def acquire_from_manual(fiscal_years: tuple[int, ...] = config.FISCAL_YEARS,
                        root: Path | None = None,
                        require_all_fys: bool = True
                        ) -> tuple[list[ManualAcquireRecord], dict[int, dict[str, Path]]]:
    """Validate that zips exist, hash them for the manifest, then extract.

    Returns (per_fy_records, parquet_paths_by_fy). The caller plumbs the
    records into the run manifest.
    """
    root = root or MANUAL_ROOT
    root.mkdir(parents=True, exist_ok=True)

    discovered = discover(fiscal_years, root=root)
    missing = [fy for fy in fiscal_years if fy not in discovered]
    if missing:
        msg = _scaffold_message(missing)
        # Pre-create the empty FY dirs so the user has somewhere to drop files.
        for fy in missing:
            (root / f"fy{fy}").mkdir(parents=True, exist_ok=True)
        if require_all_fys:
            raise FileNotFoundError(msg)
        LOG.warning(msg)

    if not discovered:
        raise FileNotFoundError(
            f"No zips found anywhere under {root}. See the manual-mode docs in README.md."
        )

    # Hash every zip so the manifest pins the exact local files used.
    records: list[ManualAcquireRecord] = []
    file_records_summary: dict[str, list[ManualFileRecord]] = {"all": []}
    for fy in sorted(discovered):
        fr_list = []
        for zp in discovered[fy]:
            fr = ManualFileRecord(
                fy=fy, path=str(zp),
                bytes=zp.stat().st_size, sha256=_sha256(zp),
            )
            fr_list.append(fr)
            file_records_summary["all"].append(fr)
        records.append(ManualAcquireRecord(
            fy=fy,
            files=len(fr_list),
            total_bytes=sum(fr.bytes for fr in fr_list),
            completed_at=datetime.now(timezone.utc).isoformat(),
            file_records=[asdict(fr) for fr in fr_list],
        ))
        LOG.info("FY%d manual: %d zip(s), %.1f MB",
                 fy, len(fr_list), sum(fr.bytes for fr in fr_list) / 1e6)

    # Persist a small index file under manual/ for operator audit.
    index_path = root / "manifest.json"
    index_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    LOG.info("Manual file manifest -> %s", index_path)

    # Reuse the archive extractor: it already handles both schema variants
    # (Public-Profile names get renamed; API-code names pass through), and
    # already iterates fy{N} sub-directories.
    extracted = extract_archives_to_parquet(
        fiscal_years=tuple(sorted(discovered)),
        archive_root=root,
        out_dir=config.INTERIM,
    )
    return records, extracted
