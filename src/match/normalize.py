"""Name normalization (methodology Section 5.4.1).

The same normalizer is applied to USAspending recipient_name and BMF NAME so
deterministic name+state matching has a chance of agreeing. The normalization
sequence:

  1. casefold
  2. ASCII-fold (drop accents)
  3. strip non-alphanumerics (keep spaces)
  4. collapse whitespace
  5. expand controlled abbreviations from reference_lists/abbreviations.yaml
  6. drop stopwords
  7. final whitespace collapse

The abbreviation list is version-pinned via the rules_version_hash in the
manifest, so a re-run on a different list produces a diff that is auditable.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable

from ..refdata import load_reference_lists


_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_MULTISPACE = re.compile(r"\s+")


def _ascii_fold(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def _abbrev_pairs() -> tuple[tuple[str, str], ...]:
    refs = load_reference_lists()
    expansions: dict[str, str] = refs.abbreviations.get("expansions", {})
    # Sort longer keys first so "med ctr" expands before "med".
    return tuple(sorted(expansions.items(), key=lambda kv: -len(kv[0])))


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    refs = load_reference_lists()
    return frozenset(refs.abbreviations.get("stopwords_strip", []))


def normalize_name(name: str | None) -> str:
    """Return the normalized form. Empty string for None / non-string input."""
    if not name or not isinstance(name, str):
        return ""
    s = _ascii_fold(name).casefold()
    s = _NON_ALNUM.sub(" ", s)
    s = _MULTISPACE.sub(" ", s).strip()

    # Word-boundary abbreviation expansion. Use a single pass with a regex
    # alternation built from the abbreviation list.
    pairs = _abbrev_pairs()
    for abbr, expansion in pairs:
        # Escape and require word boundaries on both sides.
        pattern = rf"\b{re.escape(abbr)}\b"
        s = re.sub(pattern, expansion, s)

    # Stopword removal.
    sw = _stopwords()
    if sw:
        s = " ".join(tok for tok in s.split() if tok not in sw)

    return _MULTISPACE.sub(" ", s).strip()


def normalize_state(state: str | None) -> str:
    if not state or not isinstance(state, str):
        return ""
    return state.strip().upper()[:2]


def normalize_ein(ein: str | None) -> str:
    """EINs are 9 digits sometimes printed with a hyphen (XX-XXXXXXX)."""
    if not ein or not isinstance(ein, str):
        return ""
    digits = re.sub(r"\D", "", ein)
    return digits.zfill(9) if 0 < len(digits) <= 9 else digits


def normalize_uei(uei: str | None) -> str:
    if not uei or not isinstance(uei, str):
        return ""
    return uei.strip().upper()


def normalize_many(names: Iterable[str | None]) -> list[str]:
    return [normalize_name(n) for n in names]
