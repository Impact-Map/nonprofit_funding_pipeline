"""Run manifest writer (methodology Section 9).

Captures the minimum reproducibility contract:
  - USASpending custom-download payload + refresh date
  - IRS BMF release date and SHA-256 per regional file
  - SAM extract date and SHA-256
  - Match-tier thresholds and abbreviation list version hash
  - Code commit hash (best-effort: from git if available)
  - Output file SHA-256 hashes for every exhibit
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_hash(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    finished_at: str | None = None
    project_root: str = str(config.PROJECT_ROOT)
    code_commit: str | None = None
    python_version: str = sys.version.split()[0]
    platform: str = platform.platform()

    classification_priority: list[str] = field(default_factory=list)
    rules_version_hash: str = ""
    deflator: str = ""
    deflator_base_fy: int = 2025
    intl_q_dollar_floor: float = 100_000.0
    fiscal_years: list[int] = field(default_factory=list)
    award_type_codes: list[str] = field(default_factory=list)
    tier3_threshold: float = 0.94
    tier4_review_top_n: int = 200

    usaspending_downloads: list[dict[str, Any]] = field(default_factory=list)
    irs_bmf_files: list[dict[str, Any]] = field(default_factory=list)
    sam_extract: dict[str, Any] | None = None
    match_stats: dict[str, Any] | None = None
    category_stats: dict[str, Any] | None = None

    output_files: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def new_manifest(run_cfg: config.RunConfig, rules_hash: str) -> RunManifest:
    now = datetime.now(timezone.utc)
    rid = now.strftime("run-%Y%m%dT%H%M%SZ")
    return RunManifest(
        run_id=rid,
        started_at=now.isoformat(),
        code_commit=_git_commit_hash(config.PROJECT_ROOT),
        classification_priority=list(run_cfg.classification_priority),
        rules_version_hash=rules_hash,
        deflator=run_cfg.deflator,
        deflator_base_fy=run_cfg.deflator_base_fy,
        intl_q_dollar_floor=run_cfg.intl_q_series_floor,
        fiscal_years=list(run_cfg.fiscal_years),
        award_type_codes=list(run_cfg.award_type_codes),
        tier3_threshold=config.TIER3_JARO_WINKLER_THRESHOLD,
        tier4_review_top_n=config.TIER4_MANUAL_REVIEW_TOP_N,
    )


def add_output(manifest: RunManifest, path: Path, label: str) -> None:
    if not path.exists():
        return
    manifest.output_files.append({
        "label": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    })


def write(manifest: RunManifest, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.MANIFESTS
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    out_path = out_dir / f"{manifest.run_id}.json"
    out_path.write_text(json.dumps(asdict(manifest), indent=2, default=str))
    return out_path
