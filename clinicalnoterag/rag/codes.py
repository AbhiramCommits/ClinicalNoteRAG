"""Code extraction for the code-aware QA metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from clinicalnoterag.data.synthea_gen import (
    CARC_REASONS,
    CPT_PROCEDURES,
    ICD10_CONDITIONS,
    RX_MEDICATIONS,
    SNOMED_OBSERVATIONS,
)

ICD10_RE = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?\b")
CARC_RE = re.compile(r"\b(?:CO|PR|OA)-\d{1,3}\b")
CPT_KNOWN = {entry["code"] for entry in CPT_PROCEDURES}
ICD10_KNOWN = {entry["code"] for entry in ICD10_CONDITIONS}
CARC_KNOWN = set(CARC_REASONS)
SNOMED_KNOWN = {entry["code"] for entry in SNOMED_OBSERVATIONS}
RXNORM_KNOWN = {entry["code"] for entry in RX_MEDICATIONS}


@dataclass(frozen=True, order=True)
class Code:
    """A normalized clinical or payment code."""

    system: str
    code: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""
        return {"system": self.system, "code": self.code}


def normalize_code_system(system: str) -> str:
    """Normalize code-system labels used in notes, FHIR and metadata."""
    value = system.strip().lower()
    if value in {"icd-10", "icd10", "icd-10-cm", "icd10cm"}:
        return "ICD-10"
    if value in {"cpt", "hcpcs"}:
        return "CPT"
    if value in {"snomed", "snomed-ct", "snomed ct"}:
        return "SNOMED"
    if value in {"rxnorm", "rx-norm"}:
        return "RxNorm"
    if value in {"carc", "claim adjustment reason code"}:
        return "CARC"
    return system


def extract_codes(text: str) -> list[Code]:
    """Extract ICD-10, CPT and CARC codes from free text.

    Only CPT codes present in the project's CPT table are returned unless the
    text explicitly prefixes a five-digit token with ``CPT``. This avoids
    treating arbitrary five-digit numbers (zip codes, dollar amounts) as codes.
    """
    codes: set[Code] = set()
    for match in ICD10_RE.finditer(text):
        code = match.group(0)
        # The ICD regex also matches some 3-character words (e.g. "I10" is fine,
        # but require a digit in the second position to reduce false positives).
        if len(code) >= 3 and code[1].isdigit():
            codes.add(Code("ICD-10", code))
    for match in CARC_RE.finditer(text):
        codes.add(Code("CARC", match.group(0)))
    for code in CPT_KNOWN:
        if re.search(rf"\b{re.escape(code)}\b", text):
            codes.add(Code("CPT", code))
    for match in re.finditer(r"\bCPT\s*[:#]?\s*(\d{5})\b", text, re.IGNORECASE):
        codes.add(Code("CPT", match.group(1)))
    for code in SNOMED_KNOWN:
        if re.search(rf"\b{re.escape(code)}\b", text):
            codes.add(Code("SNOMED", code))
    for code in RXNORM_KNOWN:
        if re.search(rf"\b{re.escape(code)}\b", text):
            codes.add(Code("RxNorm", code))
    return sorted(codes)


__all__ = [
    "CARC_KNOWN",
    "CPT_KNOWN",
    "ICD10_KNOWN",
    "RXNORM_KNOWN",
    "SNOMED_KNOWN",
    "Code",
    "extract_codes",
    "normalize_code_system",
]
