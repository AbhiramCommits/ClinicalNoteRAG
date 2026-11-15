"""Section-aware chunking for notes plus structured FHIR fact chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from clinicalnoterag.data.fhir_models import (
    CARC_SYSTEM,
    CPT_SYSTEM,
    ICD10_SYSTEM,
    RXNORM_SYSTEM,
    SNOMED_SYSTEM,
)

HEADING_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 /&'\-]{2,70}):\s*$", re.MULTILINE)

SYSTEM_LABEL = {
    ICD10_SYSTEM: "ICD-10",
    CPT_SYSTEM: "CPT",
    SNOMED_SYSTEM: "SNOMED",
    RXNORM_SYSTEM: "RxNorm",
    CARC_SYSTEM: "CARC",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "section"


def content_hash(text: str, *parts: object) -> str:
    """Return a stable SHA-256 content hash used for idempotent upserts."""
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    for part in parts:
        digest.update(b"|")
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()


@dataclass
class Chunk:
    """A retrievable unit with clinical metadata."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(
                self.text, self.chunk_id, self.metadata.get("patient_id", "")
            )

    @property
    def storage_id(self) -> str:
        """Primary key in the vector store (content hash => idempotent)."""
        return self.content_hash

    def to_row(self) -> dict[str, Any]:
        """Return a metadata row for the store."""
        return {
            "id": self.storage_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": {**self.metadata, "chunk_id": self.chunk_id},
        }


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a note into (section name, section text) pairs on heading lines.

    A preamble before the first heading is returned as section ``header``.
    Headings are retained in the section text so retrieved quotes stay readable.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("header", text.strip())]
    sections: list[tuple[str, str]] = []
    first = matches[0]
    preamble = text[: first.start()].strip()
    if preamble:
        sections.append(("header", preamble))
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((_slug(match.group(1)), section_text))
    return sections


def token_windows(text: str, target_tokens: int = 256, overlap_tokens: int = 32) -> list[str]:
    """Split text into overlapping whitespace-token windows.

    Args:
        text: Section text.
        target_tokens: Maximum tokens per window.
        overlap_tokens: Tokens shared between adjacent windows.

    Returns:
        A list of windows, each at most ``target_tokens`` tokens.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not 0 <= overlap_tokens < target_tokens:
        raise ValueError("overlap_tokens must be in [0, target_tokens)")
    tokens = text.split()
    if len(tokens) <= target_tokens:
        return [text]
    step = target_tokens - overlap_tokens
    windows: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + target_tokens]
        if not window:
            break
        windows.append(" ".join(window))
        if start + target_tokens >= len(tokens):
            break
    return windows


def chunk_note(
    note: dict[str, Any],
    text: str | None = None,
    target_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[Chunk]:
    """Chunk one clinical or payment note, respecting section headings.

    Args:
        note: Note record with ``doc_id``, patient/encounter ids, doc_type and
            service_date.
        text: Optional de-identified text. Defaults to ``note["text"]``.
        target_tokens: Token target per chunk.
        overlap_tokens: Token overlap between windows of a long section.

    Returns:
        Chunks with metadata for patient, encounter, doc_type and service date.
    """
    body = text if text is not None else str(note["text"])
    base_meta = {
        "patient_id": note.get("patient_id"),
        "encounter_id": note.get("encounter_id"),
        "claim_id": note.get("claim_id"),
        "doc_type": note.get("doc_type", "clinical_note"),
        "resource_type": "Note",
        "service_date": note.get("service_date"),
        "title": note.get("title", ""),
        "synthetic": True,
    }
    chunks: list[Chunk] = []
    index = 0
    for section, section_text in split_sections(body):
        for window in token_windows(section_text, target_tokens, overlap_tokens):
            chunk_id = f"{note['doc_id']}::chunk-{index:03d}"
            metadata = {**base_meta, "section": section, "chunk_index": index}
            if note.get("denial"):
                denial = note["denial"]
                if denial.get("denied"):
                    metadata["denial_code"] = denial.get("carc_code")
                    metadata["denied"] = True
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=str(note["doc_id"]),
                    text=window,
                    metadata=metadata,
                )
            )
            index += 1
    return chunks


def _coding(resource: dict[str, Any], field_name: str = "code") -> dict[str, str]:
    concept = resource.get(field_name) or {}
    coding = (concept.get("coding") or [{}])[0]
    return {
        "system": SYSTEM_LABEL.get(coding.get("system", ""), coding.get("system", "UNKNOWN")),
        "code": coding.get("code", ""),
        "display": coding.get("display") or concept.get("text", ""),
    }


def _fact(
    resource: dict[str, Any],
    resource_type: str,
    patient_id: str,
    encounter_id: str | None,
    service_date: str | None,
    body: str,
    code_system: str | None = None,
    code: str | None = None,
    display: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Chunk:
    metadata: dict[str, Any] = {
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "resource_type": resource_type,
        "resource_id": resource["id"],
        "doc_type": "fhir_fact",
        "service_date": service_date,
        "code_system": code_system,
        "code": code,
        "code_display": display,
        "synthetic": True,
    }
    if extra:
        metadata.update(extra)
    return Chunk(
        chunk_id=f"FACT::{resource_type}::{resource['id']}",
        doc_id=f"FHIR-{resource_type}-{resource['id']}",
        text=body,
        metadata=metadata,
    )


def _find_encounter_for_claim(
    claim: dict[str, Any], encounters: list[dict[str, Any]]
) -> str | None:
    start = claim.get("billablePeriod", {}).get("start")
    for encounter in encounters:
        if encounter.get("period", {}).get("start") == start:
            return encounter["id"]
    return None


def chunk_bundle(bundle: dict[str, Any]) -> list[Chunk]:
    """Render one retrievable fact chunk per FHIR resource.

    Conditions, Procedures and Claims are required by the project spec; the
    function additionally renders Encounters, Observations, MedicationRequests
    and ClaimResponses because treatment and payment questions depend on them.
    """
    resources = [entry["resource"] for entry in bundle.get("entry", [])]
    patient = next(r for r in resources if r["resourceType"] == "Patient")
    patient_id = patient["id"]
    encounters = [r for r in resources if r["resourceType"] == "Encounter"]
    claims = {r["id"]: r for r in resources if r["resourceType"] == "Claim"}
    responses = {
        r["request"]["reference"].split("/")[-1]: r
        for r in resources
        if r["resourceType"] == "ClaimResponse"
    }
    chunks: list[Chunk] = []
    for resource in resources:
        rtype = resource["resourceType"]
        if rtype == "Patient":
            continue
        if rtype == "Encounter":
            enc_date = resource["period"]["start"]
            enc_type = resource["type"][0].get("text", "")
            body = (
                f"FHIR Encounter {resource['id']} for patient {patient_id}. "
                f"Encounter type: {enc_type}. Date of service: {enc_date}. "
                f"Period: {resource['period']['start']} to {resource['period'].get('end', enc_date)}. "
                f"Status: {resource.get('status', 'finished')}."
            )
            chunks.append(_fact(resource, rtype, patient_id, resource["id"], enc_date, body))
        elif rtype in {"Condition", "Procedure"}:
            coding = _coding(resource)
            enc_id = resource["encounter"]["reference"].split("/")[-1]
            enc = next((e for e in encounters if e["id"] == enc_id), None)
            service_date = (
                resource.get("onsetDateTime")
                or resource.get("performedPeriod", {}).get("start")
                or (enc["period"]["start"] if enc else None)
            )
            label = "Diagnosis" if rtype == "Condition" else "Procedure performed"
            body = (
                f"FHIR {rtype} {resource['id']} for patient {patient_id}, encounter {enc_id}. "
                f"{label}: {coding['display']} ({coding['system']} {coding['code']}). "
                f"Service date: {service_date}. Status: "
                f"{resource.get('clinicalStatus', {}).get('text', resource.get('status', ''))}."
            )
            chunks.append(
                _fact(
                    resource,
                    rtype,
                    patient_id,
                    enc_id,
                    service_date,
                    body,
                    coding["system"],
                    coding["code"],
                    coding["display"],
                )
            )
        elif rtype == "Observation":
            coding = _coding(resource)
            enc_id = resource["encounter"]["reference"].split("/")[-1]
            quantity = resource.get("valueQuantity") or {}
            value = quantity.get("value", resource.get("valueString", ""))
            unit = quantity.get("unit", "")
            service_date = resource.get("effectiveDateTime", "")[:10]
            body = (
                f"FHIR Observation {resource['id']} for patient {patient_id}, encounter {enc_id}. "
                f"Observation: {coding['display']} (SNOMED {coding['code']}). "
                f"Value: {value} {unit}. Date: {service_date}."
            )
            chunks.append(
                _fact(
                    resource,
                    rtype,
                    patient_id,
                    enc_id,
                    service_date,
                    body,
                    coding["system"],
                    coding["code"],
                    coding["display"],
                )
            )
        elif rtype == "MedicationRequest":
            coding = _coding(resource, "medicationCodeableConcept")
            enc_id = resource["encounter"]["reference"].split("/")[-1]
            service_date = resource.get("authoredOn")
            body = (
                f"FHIR MedicationRequest {resource['id']} for patient {patient_id}, encounter {enc_id}. "
                f"Medication ordered: {coding['display']} (RxNorm {coding['code']}). "
                f"Ordered on: {service_date}. Status: {resource.get('status', 'active')}."
            )
            chunks.append(
                _fact(
                    resource,
                    rtype,
                    patient_id,
                    enc_id,
                    service_date,
                    body,
                    coding["system"],
                    coding["code"],
                    coding["display"],
                )
            )
        elif rtype == "Claim":
            enc_id = _find_encounter_for_claim(resource, encounters)
            service_date = resource.get("billablePeriod", {}).get("start")
            item_codes = [_coding(item, "productOrService") for item in resource.get("item", [])]
            cpt = item_codes[0] if item_codes else {"code": "", "display": "", "system": "CPT"}
            dx = _coding(resource.get("diagnosis", [{}])[0], "diagnosisCodeableConcept")
            response = responses.get(resource["id"])
            denied = False
            denial_code = None
            allowed = 0.0
            paid = 0.0
            if response:
                for adjudication in response.get("adjudication", []):
                    category = adjudication.get("category", {}).get("coding", [{}])[0].get("code")
                    amount = float((adjudication.get("amount") or {}).get("value", 0.0))
                    if category == "eligible":
                        allowed = amount
                    elif category == "benefit":
                        paid = amount
                    elif category == "denial":
                        denied = True
                        denial_code = (adjudication.get("reason", {}).get("coding") or [{}])[0].get(
                            "code"
                        )
            total = float((resource.get("total") or {}).get("value", 0.0))
            body = (
                f"FHIR Claim {resource['id']} for patient {patient_id}, encounter {enc_id}. "
                f"Claim date: {service_date}. Total billed: ${total:,.2f}. "
                f"Primary procedure: CPT {cpt['code']} ({cpt['display']}). "
                f"Supporting diagnosis: ICD-10 {dx['code']} ({dx['display']}). "
                f"All procedure codes: {', '.join(c['code'] for c in item_codes)}. "
                f"Denied: {'yes' if denied else 'no'}. "
                + (f"Denial reason code: {denial_code}. " if denial_code else "")
                + f"Allowed amount: ${allowed:,.2f}. Paid amount: ${paid:,.2f}."
            )
            chunks.append(
                _fact(
                    resource,
                    rtype,
                    patient_id,
                    enc_id,
                    service_date,
                    body,
                    cpt["system"],
                    cpt["code"],
                    cpt["display"],
                    {
                        "claim_id": resource["id"],
                        "denied": denied,
                        "denial_code": denial_code,
                        "billed_amount": total,
                        "allowed_amount": allowed,
                        "paid_amount": paid,
                    },
                )
            )
        elif rtype == "ClaimResponse":
            claim = claims.get(resource["request"]["reference"].split("/")[-1], {})
            enc_id = _find_encounter_for_claim(claim, encounters)
            service_date = claim.get("billablePeriod", {}).get("start")
            denial_code = None
            reason = None
            for adjudication in resource.get("adjudication", []):
                category = adjudication.get("category", {}).get("coding", [{}])[0].get("code")
                if category == "denial":
                    coding = (adjudication.get("reason", {}).get("coding") or [{}])[0]
                    denial_code = coding.get("code")
                    reason = coding.get("display")
            body = (
                f"FHIR ClaimResponse {resource['id']} for claim {resource.get('request', {}).get('reference', '')} "
                f"and patient {patient_id}. Outcome: {resource.get('outcome', 'complete')}. "
                f"Denied: {'yes' if denial_code else 'no'}. "
                + (f"Denial reason code: {denial_code} ({reason}). " if denial_code else "")
                + f"Disposition: {resource.get('disposition', '')}"
            )
            chunks.append(
                _fact(
                    resource,
                    rtype,
                    patient_id,
                    enc_id,
                    service_date,
                    body,
                    "CARC" if denial_code else None,
                    denial_code,
                    reason,
                    {
                        "claim_id": claim.get("id"),
                        "denied": bool(denial_code),
                        "denial_code": denial_code,
                    },
                )
            )
    return chunks


__all__ = [
    "HEADING_RE",
    "Chunk",
    "chunk_bundle",
    "chunk_note",
    "content_hash",
    "split_sections",
    "token_windows",
]
