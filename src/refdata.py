"""Loader and version-hasher for the YAML reference lists.

The classification rules in Section 4 are driven entirely by these files. The
combined SHA-256 of all reference YAMLs is recorded in the run manifest so a
result can be tied to the exact rule set that produced it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import config


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ReferenceLists:
    intl_subagencies: dict[str, Any]
    intl_listings: dict[str, Any]
    intl_subcategory_rules: dict[str, Any]
    abbreviations: dict[str, Any]
    covid_programs: dict[str, Any]
    ntee_categories: dict[str, Any]
    business_types: dict[str, Any]
    rules_version_hash: str


REFERENCE_FILES = (
    "intl_subagencies.yaml",
    "intl_listings.yaml",
    "intl_subcategory_rules.yaml",
    "abbreviations.yaml",
    "covid_programs.yaml",
    "ntee_categories.yaml",
    "business_types.yaml",
)


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_reference_lists() -> ReferenceLists:
    base = config.REFERENCE_LISTS
    paths = [base / name for name in REFERENCE_FILES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference list files: {missing}")
    docs = {p.stem: _read_yaml(p) for p in paths}
    return ReferenceLists(
        intl_subagencies=docs["intl_subagencies"],
        intl_listings=docs["intl_listings"],
        intl_subcategory_rules=docs["intl_subcategory_rules"],
        abbreviations=docs["abbreviations"],
        covid_programs=docs["covid_programs"],
        ntee_categories=docs["ntee_categories"],
        business_types=docs["business_types"],
        rules_version_hash=_hash_files(paths),
    )
