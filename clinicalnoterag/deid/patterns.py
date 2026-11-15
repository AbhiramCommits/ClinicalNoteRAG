"""Regex patterns and the deterministic offline PHI de-identification backend.

The patterns cover the entity types injected by the synthetic note generator:
names from the synthetic name list, MRN, SSN, phone, email, date, address and
ZIP. They are deliberately conservative: replacing a CPT code because it looks
like a five-digit ZIP would corrupt clinical evidence, so ZIP detection is
anchored to address/state context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clinicalnoterag.data.synthea_gen import all_known_names

ENTITY_TYPES: tuple[str, ...] = (
    "NAME",
    "MRN",
    "SSN",
    "PHONE",
    "EMAIL",
    "DATE",
    "ADDRESS",
    "ZIP",
)

#: Pattern specificity used to resolve overlaps (higher wins when spans overlap).
PRIORITY = {
    "EMAIL": 100,
    "SSN": 95,
    "MRN": 90,
    "PHONE": 85,
    "ADDRESS": 80,
    "ZIP": 70,
    "DATE": 60,
    "NAME": 50,
}

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN_RE = re.compile(r"\bMRN[-][A-Z0-9][A-Z0-9\-]{4,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:\+1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}\b"
)
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][A-Za-z]+\s+){1,4}"
    r"(?:Street|St|Lane|Ln|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Court|Ct|Way|Place|Pl)\b"
    r"(?:,\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
)
_ZIP_RE = re.compile(
    r"(?<=,\s[A-Z]{2}\s)\d{5}(?:-\d{4})?\b"
    r"|\b(?:ZIP|Zip|zip)\s*:?\s*\d{5}(?:-\d{4})?\b"
)


@dataclass(frozen=True)
class PatternEntity:
    """A PHI span detected by the regex backend."""

    entity_type: str
    start: int
    end: int
    text: str


def known_names(extra: list[str] | None = None) -> list[str]:
    """Return the synthetic name gazetteer, longest-first, plus extras."""
    names = set(all_known_names())
    if extra:
        names.update(n for n in extra if n.strip())
    return sorted(names, key=len, reverse=True)


def _name_regex(names: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(name) for name in names)
    return re.compile(rf"(?:(?:Dr|Mr|Mrs|Ms)\.?\s+)?\b(?:{alternatives})\b")


def build_patterns(extra_names: list[str] | None = None) -> dict[str, re.Pattern[str]]:
    """Build the compiled pattern map used by the fallback backend."""
    return {
        "EMAIL": _EMAIL_RE,
        "SSN": _SSN_RE,
        "MRN": _MRN_RE,
        "PHONE": _PHONE_RE,
        "ADDRESS": _ADDRESS_RE,
        "ZIP": _ZIP_RE,
        "DATE": _DATE_RE,
        "NAME": _name_regex(known_names(extra_names)),
    }


def find_entities(text: str, extra_names: list[str] | None = None) -> list[PatternEntity]:
    """Find non-overlapping PHI spans in ``text``.

    Overlaps are resolved by (start, longest, priority), so a full address wins
    over the ZIP inside it and a full name wins over a first-name fragment.

    Args:
        text: Free text to scan.
        extra_names: Optional extra known names (e.g. corpus-specific names).

    Returns:
        Entities sorted by start offset.
    """
    patterns = build_patterns(extra_names)
    candidates: list[PatternEntity] = []
    for entity_type, pattern in patterns.items():
        for match in pattern.finditer(text):
            if match.group(0).strip():
                candidates.append(
                    PatternEntity(entity_type, match.start(), match.end(), match.group(0))
                )
    candidates.sort(key=lambda e: (e.start, -(e.end - e.start), -PRIORITY[e.entity_type]))
    selected: list[PatternEntity] = []
    for candidate in candidates:
        if any(
            candidate.start < chosen.end and chosen.start < candidate.end for chosen in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda e: e.start)


__all__ = [
    "ENTITY_TYPES",
    "PRIORITY",
    "PatternEntity",
    "build_patterns",
    "find_entities",
    "known_names",
]
