"""Build a labeled clinical RAG benchmark from the synthetic generator's truth."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinicalnoterag.data.fhir_models import (
    CARC_SYSTEM,
    CPT_SYSTEM,
    ICD10_SYSTEM,
    RXNORM_SYSTEM,
    SNOMED_SYSTEM,
)
from clinicalnoterag.data.synthea_gen import load_bundle
from clinicalnoterag.rag.generators import ABSTENTION_TEXT

SYSTEM_LABEL = {
    ICD10_SYSTEM: "ICD-10",
    CPT_SYSTEM: "CPT",
    SNOMED_SYSTEM: "SNOMED",
    RXNORM_SYSTEM: "RxNorm",
    CARC_SYSTEM: "CARC",
}


def _resources(bundle: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    return [
        entry["resource"]
        for entry in bundle["entry"]
        if entry["resource"]["resourceType"] == resource_type
    ]


def _coding(concept: dict[str, Any]) -> dict[str, str]:
    coding = (concept.get("coding") or [{}])[0]
    return {
        "system": SYSTEM_LABEL.get(coding.get("system", ""), coding.get("system", "UNKNOWN")),
        "code": str(coding.get("code", "")),
        "display": str(coding.get("display") or concept.get("text") or ""),
    }


def _denial(claim_response: dict[str, Any]) -> dict[str, str] | None:
    for adjudication in claim_response.get("adjudication", []):
        category = adjudication.get("category", {}).get("coding", [{}])[0].get("code")
        if category == "denial":
            reason = adjudication.get("reason", {})
            coding = (reason.get("coding") or [{}])[0]
            return {
                "code": str(coding.get("code", "")),
                "reason": str(coding.get("display") or reason.get("text") or ""),
            }
    return None


def _sanitize_reason(reason: str) -> str:
    return reason.replace(". ", "; ").rstrip(".")


def _join_sentences(parts: list[str]) -> str:
    return " ".join(part.rstrip(".") + "." for part in parts if part)


def _diagnosis_answer(conditions: list[dict[str, Any]]) -> str:
    parts = []
    for condition in sorted(conditions, key=lambda c: _coding(c["code"])["code"]):
        coding = _coding(condition["code"])
        parts.append(f"{coding['display']} was documented (ICD-10 {coding['code']})")
    return _join_sentences(parts)


def _procedure_answer(procedures: list[dict[str, Any]]) -> str:
    parts = []
    for procedure in sorted(procedures, key=lambda p: _coding(p["code"])["code"]):
        coding = _coding(procedure["code"])
        parts.append(f"{coding['display']} was performed (CPT {coding['code']})")
    return _join_sentences(parts)


def _medication_answer(medications: list[dict[str, Any]]) -> str:
    parts = []
    for medication in sorted(
        medications, key=lambda m: _coding(m["medicationCodeableConcept"])["code"]
    ):
        coding = _coding(medication["medicationCodeableConcept"])
        parts.append(f"{coding['display']} was ordered (RxNorm {coding['code']})")
    return _join_sentences(parts)


def _observation_answer(observation: dict[str, Any]) -> tuple[str, float]:
    coding = _coding(observation["code"])
    quantity = observation["valueQuantity"]
    value = float(quantity["value"])
    unit = quantity["unit"]
    date = observation["effectiveDateTime"][:10]
    answer = f"{coding['display']} was {value} {unit} on {date} (SNOMED {coding['code']})"
    return _join_sentences([answer]), value


def _codes_for(resources: list[dict[str, Any]], field: str = "code") -> list[dict[str, str]]:
    codes: list[dict[str, str]] = []
    for resource in resources:
        coding = _coding(resource[field])
        if coding["code"]:
            codes.append({"system": coding["system"], "code": coding["code"]})
    return sorted(codes, key=lambda item: (item["system"], item["code"]))


def _item(
    item_id: str,
    family: str,
    question: str,
    gold_answer: str,
    filters: dict[str, Any],
    k: int,
    *,
    gold_codes: list[dict[str, str]] | None = None,
    gold_supporting_chunk_ids: list[str] | None = None,
    answerable: bool = True,
    gold_numeric: float | None = None,
    patient_id: str | None = None,
    encounter_id: str | None = None,
    claim_id: str | None = None,
    task_type: str = "",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "family": family,
        "task_type": task_type,
        "question": question,
        "gold_answer": gold_answer,
        "gold_codes": gold_codes or [],
        "gold_supporting_chunk_ids": gold_supporting_chunk_ids or [],
        "answerable": answerable,
        "gold_numeric": gold_numeric,
        "filters": filters,
        "k": k,
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "claim_id": claim_id,
    }


def build_benchmark(
    corpus_dir: str | Path,
    out_path: str | Path | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create >=200 QA items across treatment, payment, operations and adversarial.

    Ground truth comes from the same FHIR bundle that generated the notes, so
    every answerable item has exact supporting facts and code labels. The
    adversarial items ask for evidence that is deliberately absent and are used
    only to measure abstention and false-answer behavior.
    """
    corpus = Path(corpus_dir)
    bundles = [load_bundle(path) for path in sorted((corpus / "bundles").glob("*.json"))]
    if not bundles:
        raise FileNotFoundError(f"no bundles found under {corpus / 'bundles'}")
    items: list[dict[str, Any]] = []

    # --- treatment / clinical QA ---------------------------------------
    treatment_count = 0
    for bundle in bundles[:20]:
        patient = _resources(bundle, "Patient")[0]["id"]
        encounters = _resources(bundle, "Encounter")
        conditions = _resources(bundle, "Condition")
        procedures = _resources(bundle, "Procedure")
        observations = _resources(bundle, "Observation")
        medications = _resources(bundle, "MedicationRequest")
        if not encounters:
            continue
        encounter = encounters[0]
        enc_id = encounter["id"]
        enc_conditions = [c for c in conditions if c["encounter"]["reference"].endswith(enc_id)]
        enc_procedures = [p for p in procedures if p["encounter"]["reference"].endswith(enc_id)]
        enc_observations = [o for o in observations if o["encounter"]["reference"].endswith(enc_id)]
        enc_medications = [m for m in medications if m["encounter"]["reference"].endswith(enc_id)]
        if enc_conditions:
            treatment_count += 1
            items.append(
                _item(
                    f"T-{treatment_count:03d}",
                    "treatment",
                    f"What diagnoses were documented for encounter {enc_id}?",
                    _diagnosis_answer(enc_conditions),
                    {
                        "patient_id": patient,
                        "encounter_id": enc_id,
                        "resource_type": "Condition",
                        "doc_type": "fhir_fact",
                    },
                    5,
                    gold_codes=_codes_for(enc_conditions),
                    gold_supporting_chunk_ids=[
                        f"FACT::Condition::{c['id']}" for c in enc_conditions
                    ],
                    patient_id=patient,
                    encounter_id=enc_id,
                    task_type="diagnoses",
                )
            )
        if enc_procedures:
            treatment_count += 1
            items.append(
                _item(
                    f"T-{treatment_count:03d}",
                    "treatment",
                    f"What procedures were performed for encounter {enc_id}?",
                    _procedure_answer(enc_procedures),
                    {
                        "patient_id": patient,
                        "encounter_id": enc_id,
                        "resource_type": "Procedure",
                        "doc_type": "fhir_fact",
                    },
                    5,
                    gold_codes=_codes_for(enc_procedures),
                    gold_supporting_chunk_ids=[
                        f"FACT::Procedure::{p['id']}" for p in enc_procedures
                    ],
                    patient_id=patient,
                    encounter_id=enc_id,
                    task_type="procedures",
                )
            )
        if enc_observations:
            observation = enc_observations[0]
            display = _coding(observation["code"])["display"]
            answer, value = _observation_answer(observation)
            treatment_count += 1
            items.append(
                _item(
                    f"T-{treatment_count:03d}",
                    "treatment",
                    f"What was the {display} recorded during encounter {enc_id}?",
                    answer,
                    {
                        "patient_id": patient,
                        "encounter_id": enc_id,
                        "resource_type": "Observation",
                        "doc_type": "fhir_fact",
                    },
                    5,
                    gold_codes=_codes_for([observation]),
                    gold_supporting_chunk_ids=[f"FACT::Observation::{observation['id']}"],
                    gold_numeric=value,
                    patient_id=patient,
                    encounter_id=enc_id,
                    task_type="observation",
                )
            )
        if enc_medications:
            treatment_count += 1
            items.append(
                _item(
                    f"T-{treatment_count:03d}",
                    "treatment",
                    f"What medications were ordered for encounter {enc_id}?",
                    _medication_answer(enc_medications),
                    {
                        "patient_id": patient,
                        "encounter_id": enc_id,
                        "resource_type": "MedicationRequest",
                        "doc_type": "fhir_fact",
                    },
                    5,
                    gold_codes=_codes_for(enc_medications, "medicationCodeableConcept"),
                    gold_supporting_chunk_ids=[
                        f"FACT::MedicationRequest::{m['id']}" for m in enc_medications
                    ],
                    patient_id=patient,
                    encounter_id=enc_id,
                    task_type="medications",
                )
            )

    # --- payment QA -----------------------------------------------------
    payment_count = 0
    for bundle in bundles:
        patient = _resources(bundle, "Patient")[0]["id"]
        claims = {c["id"]: c for c in _resources(bundle, "Claim")}
        responses = {
            r["request"]["reference"].split("/")[-1]: r for r in _resources(bundle, "ClaimResponse")
        }
        for claim_id, claim in claims.items():
            response = responses.get(claim_id)
            denial = _denial(response) if response else None
            if denial is None:
                continue
            payment_count += 1
            cpt = _coding(claim["item"][0]["productOrService"])
            dx = _coding(claim["diagnosis"][0]["diagnosisCodeableConcept"])
            reason = _sanitize_reason(denial["reason"])
            why_answer = _join_sentences(
                [
                    f"Claim {claim_id} was denied with CARC reason {denial['code']}: {reason}",
                    f"The primary procedure associated with the denial was CPT {cpt['code']} ({cpt['display']})",
                    f"The supporting diagnosis for the appeal was ICD-10 {dx['code']} ({dx['display']})",
                ]
            )
            filters = {
                "patient_id": patient,
                "claim_id": claim_id,
                "doc_type": "fhir_fact",
            }
            items.append(
                _item(
                    f"P-{payment_count:03d}",
                    "payment",
                    f"Why was claim {claim_id} denied and what supports an appeal?",
                    why_answer,
                    filters,
                    5,
                    gold_codes=[
                        {"system": "CARC", "code": denial["code"]},
                        {"system": "CPT", "code": cpt["code"]},
                        {"system": "ICD-10", "code": dx["code"]},
                    ],
                    gold_supporting_chunk_ids=[
                        f"FACT::Claim::{claim_id}",
                        f"FACT::ClaimResponse::{response['id']}",
                    ],
                    patient_id=patient,
                    claim_id=claim_id,
                    task_type="denial_reason",
                )
            )
            payment_count += 1
            code_answer = _join_sentences(
                [
                    f"The denial reason code was {denial['code']} and the primary procedure code was CPT {cpt['code']}"
                ]
            )
            items.append(
                _item(
                    f"P-{payment_count:03d}",
                    "payment",
                    f"Which code drove the denial for claim {claim_id}?",
                    code_answer,
                    filters,
                    5,
                    gold_codes=[
                        {"system": "CARC", "code": denial["code"]},
                        {"system": "CPT", "code": cpt["code"]},
                    ],
                    gold_supporting_chunk_ids=[f"FACT::Claim::{claim_id}"],
                    patient_id=patient,
                    claim_id=claim_id,
                    task_type="denial_code",
                )
            )
            if payment_count >= 60:
                break
        if payment_count >= 60:
            break

    # --- operations QA --------------------------------------------------
    operations_count = 0
    for bundle in bundles[:20]:
        patient = _resources(bundle, "Patient")[0]["id"]
        encounters = _resources(bundle, "Encounter")
        claims = _resources(bundle, "Claim")
        responses = {
            r["request"]["reference"].split("/")[-1]: r for r in _resources(bundle, "ClaimResponse")
        }
        denied_claims = [c for c in claims if _denial(responses.get(c["id"], {}))]
        operations_count += 1
        items.append(
            _item(
                f"O-{operations_count:03d}",
                "operations",
                f"How many encounters did patient {patient} have?",
                f"There {'is' if len(encounters) == 1 else 'are'} {len(encounters)} "
                f"{'encounter' if len(encounters) == 1 else 'encounters'} for patient {patient}.",
                {
                    "patient_id": patient,
                    "resource_type": "Encounter",
                    "doc_type": "fhir_fact",
                },
                5,
                gold_supporting_chunk_ids=[f"FACT::Encounter::{e['id']}" for e in encounters],
                gold_numeric=float(len(encounters)),
                patient_id=patient,
                task_type="encounter_count",
            )
        )
        total = sum(float(c["total"]["value"]) for c in claims)
        operations_count += 1
        items.append(
            _item(
                f"O-{operations_count:03d}",
                "operations",
                f"What is the total billed amount for patient {patient}?",
                f"The total billed amount for patient {patient} is ${total:,.2f} "
                f"across {len(claims)} {'claim' if len(claims) == 1 else 'claims'}.",
                {
                    "patient_id": patient,
                    "resource_type": "Claim",
                    "doc_type": "fhir_fact",
                },
                5,
                gold_supporting_chunk_ids=[f"FACT::Claim::{c['id']}" for c in claims],
                gold_numeric=total,
                patient_id=patient,
                task_type="total_billed",
            )
        )
        operations_count += 1
        items.append(
            _item(
                f"O-{operations_count:03d}",
                "operations",
                f"How many claims for patient {patient} were denied?",
                f"There {'is' if len(denied_claims) == 1 else 'are'} {len(denied_claims)} "
                f"{'denied claim' if len(denied_claims) == 1 else 'denied claims'} "
                f"for patient {patient}.",
                {
                    "patient_id": patient,
                    "resource_type": "Claim",
                    "doc_type": "fhir_fact",
                },
                5,
                gold_supporting_chunk_ids=[f"FACT::Claim::{c['id']}" for c in claims],
                gold_numeric=float(len(denied_claims)),
                patient_id=patient,
                task_type="denied_count",
            )
        )

    # --- adversarial / unanswerable ------------------------------------
    adversarial_count = 0
    for bundle in bundles[:10]:
        patient = _resources(bundle, "Patient")[0]["id"]
        number = patient.split("-")[-1]
        adversarial_count += 1
        items.append(
            _item(
                f"A-{adversarial_count:03d}",
                "adversarial",
                f"What diagnosis code Z99.999 was documented for patient {patient}?",
                ABSTENTION_TEXT,
                {
                    "patient_id": patient,
                    "resource_type": "Condition",
                    "doc_type": "fhir_fact",
                },
                5,
                answerable=False,
                patient_id=patient,
                task_type="absent_diagnosis",
            )
        )
        adversarial_count += 1
        items.append(
            _item(
                f"A-{adversarial_count:03d}",
                "adversarial",
                f"Why was claim CLM-{number}-99 denied?",
                ABSTENTION_TEXT,
                {
                    "patient_id": patient,
                    "claim_id": f"CLM-{number}-99",
                    "doc_type": "fhir_fact",
                },
                5,
                answerable=False,
                patient_id=patient,
                task_type="absent_claim",
            )
        )
        adversarial_count += 1
        items.append(
            _item(
                f"A-{adversarial_count:03d}",
                "adversarial",
                f"What was the LDL cholesterol recorded for patient {patient}?",
                ABSTENTION_TEXT,
                {
                    "patient_id": patient,
                    "resource_type": "Observation",
                    "doc_type": "fhir_fact",
                },
                5,
                answerable=False,
                patient_id=patient,
                task_type="absent_observation",
            )
        )
        adversarial_count += 1
        items.append(
            _item(
                f"A-{adversarial_count:03d}",
                "adversarial",
                f"What procedure with CPT 99999 was performed for patient {patient}?",
                ABSTENTION_TEXT,
                {
                    "patient_id": patient,
                    "resource_type": "Procedure",
                    "doc_type": "fhir_fact",
                },
                5,
                answerable=False,
                patient_id=patient,
                task_type="absent_procedure",
            )
        )

    if len(items) < 200:
        raise RuntimeError(
            f"benchmark construction produced only {len(items)} items; expected >=200"
        )
    target = Path(out_path) if out_path else Path("data/benchmark/clinical_qa.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "n_items": len(items),
        "n_answerable": sum(1 for item in items if item["answerable"]),
        "n_unanswerable": sum(1 for item in items if not item["answerable"]),
        "by_family": {
            family: sum(1 for item in items if item["family"] == family)
            for family in ("treatment", "payment", "operations", "adversarial")
        },
        "path": str(target),
        "provenance": "Derived from synthetic FHIR bundles generated by this repository.",
    }
    (target.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return items


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL benchmark."""
    items: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))
    return items


__all__ = ["build_benchmark", "load_benchmark"]
