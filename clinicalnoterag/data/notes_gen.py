"""Templated synthetic clinical and payment-operations notes.

Notes are generated per encounter. They deliberately embed the fake PHI that
was assigned to the synthetic patient so the de-identification audit has exact
ground truth, and they explicitly repeat the ICD-10/CPT/SNOMED/RxNorm codes
that appear in the corresponding FHIR bundle so downstream QA evaluation has
verifiable gold evidence.
"""

from __future__ import annotations

import random
from datetime import date, datetime
from typing import Any

from clinicalnoterag.data.fhir_models import CARC_SYSTEM, CPT_SYSTEM, ICD10_SYSTEM, RXNORM_SYSTEM, SNOMED_SYSTEM
from clinicalnoterag.data.synthea_gen import CARC_REASONS, PROVIDERS, provider_for

SYSTEM_LABEL = {
    ICD10_SYSTEM: "ICD-10",
    CPT_SYSTEM: "CPT",
    SNOMED_SYSTEM: "SNOMED",
    RXNORM_SYSTEM: "RxNorm",
    CARC_SYSTEM: "CARC",
}


def _by_type(bundle: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == resource_type]


def _ref_id(reference: str) -> str:
    return reference.split("/", 1)[-1]


def _codes_of(resource: dict[str, Any], field: str = "code") -> list[dict[str, str]]:
    concept = resource.get(field) or {}
    out: list[dict[str, str]] = []
    for coding in concept.get("coding", []):
        label = SYSTEM_LABEL.get(coding.get("system", ""), coding.get("system", "UNKNOWN"))
        out.append({"system": label, "code": coding["code"], "display": coding.get("display", "")})
    return out


def _fmt_date(value: str, style: str = "iso") -> str:
    d = datetime.strptime(value[:10], "%Y-%m-%d").date()
    if style == "us":
        return d.strftime("%m/%d/%Y")
    return d.isoformat()


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def _injected_phi(candidates: dict[str, list[str]], text: str) -> dict[str, list[str]]:
    """Keep only injected values that actually occur in the text (audit truth)."""
    out: dict[str, list[str]] = {}
    for entity_type, values in candidates.items():
        present = sorted({value for value in values if value and value in text})
        if present:
            out[entity_type] = present
    return out


def _patient_facts(bundle: dict[str, Any]) -> dict[str, Any]:
    patient = _by_type(bundle, "Patient")[0]
    address = patient["address"][0]
    full_address = (
        f"{address['line'][0]}, {address['city']}, {address['state']} {address['postalCode']}"
    )
    phone = next(t["value"] for t in patient["telecom"] if t["system"] == "phone")
    email = next(t["value"] for t in patient["telecom"] if t["system"] == "email")
    identifiers = {i["system"]: i["value"] for i in patient["identifier"]}
    mrn = identifiers["urn:oid:2.16.840.1.113883.19.5.99999.1"]
    ssn = identifiers["http://hl7.org/fhir/sid/us-ssn"]
    name = patient["name"][0]
    full_name = f"{name['given'][0]} {name['family']}"
    return {
        "patient_id": patient["id"],
        "full_name": full_name,
        "mrn": mrn,
        "ssn": ssn,
        "dob": patient["birthDate"],
        "dob_us": _fmt_date(patient["birthDate"], "us"),
        "phone": phone,
        "email": email,
        "address": address,
        "full_address": full_address,
        "gender": patient["gender"],
    }


def _provider_for_encounter(encounter: dict[str, Any]) -> dict[str, str]:
    participant = (encounter.get("participant") or [{}])[0]
    reference = participant.get("individual", {}).get("reference", "Practitioner/PRAC-001")
    prac_id = _ref_id(reference)
    for provider in PROVIDERS:
        if provider["id"] == prac_id:
            return provider
    return PROVIDERS[0]


def _age(dob: str, service_date: str) -> int:
    born = datetime.strptime(dob, "%Y-%m-%d").date()
    seen = datetime.strptime(service_date, "%Y-%m-%d").date()
    return seen.year - born.year - ((seen.month, seen.day) < (born.month, born.day))


def _claim_for_encounter(claim: dict[str, Any]) -> str:
    return claim["id"]


def _denial_info(claim_response: dict[str, Any]) -> dict[str, Any]:
    denial = None
    for adjudication in claim_response.get("adjudication", []):
        code = adjudication.get("category", {}).get("coding", [{}])[0].get("code")
        if code == "denial":
            denial = adjudication
            break
    if denial is None:
        return {"denied": False, "code": None, "reason": None}
    reason = denial.get("reason", {})
    coding = (reason.get("coding") or [{}])[0]
    return {
        "denied": True,
        "code": coding.get("code"),
        "reason": coding.get("display") or reason.get("text"),
    }


def _amounts(claim_response: dict[str, Any]) -> dict[str, float]:
    amounts = {"submitted": 0.0, "allowed": 0.0, "paid": 0.0}
    for adjudication in claim_response.get("adjudication", []):
        code = adjudication.get("category", {}).get("coding", [{}])[0].get("code")
        value = float((adjudication.get("amount") or {}).get("value", 0.0))
        if code == "submitted":
            amounts["submitted"] = value
        elif code == "eligible":
            amounts["allowed"] = value
        elif code == "benefit":
            amounts["paid"] = value
    return amounts


def _clinical_note(
    bundle: dict[str, Any],
    encounter: dict[str, Any],
    conditions: list[dict[str, Any]],
    procedures: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    medications: list[dict[str, Any]],
    claim: dict[str, Any],
    provider: dict[str, str],
    rng: random.Random,
) -> dict[str, Any]:
    p = _patient_facts(bundle)
    enc_date = encounter["period"]["start"]
    age = _age(p["dob"], enc_date)
    enc_label = encounter["type"][0]["text"]

    condition_line = "; ".join(
        f"{c['code']['coding'][0]['display']} (ICD-10 {c['code']['coding'][0]['code']})"
        for c in conditions
    )
    procedure_line = "; ".join(
        f"{pr['code']['coding'][0]['display']} (CPT {pr['code']['coding'][0]['code']})"
        for pr in procedures
    )
    med_line = (
        "; ".join(
            f"{m['medicationCodeableConcept']['coding'][0]['display']} "
            f"(RxNorm {m['medicationCodeableConcept']['coding'][0]['code']})"
            for m in medications
        )
        if medications
        else "No new medications were prescribed at this visit."
    )
    obs_line = "; ".join(
        f"{o['code']['coding'][0]['display']} {o['valueQuantity']['value']} "
        f"{o['valueQuantity']['unit']}"
        for o in observations
    )
    symptoms = "; ".join(c["code"]["coding"][0]["display"].split(",")[0].lower() for c in conditions)
    history_problem = conditions[0]["code"]["coding"][0]["display"]

    chief = f"{p['full_name']} presents with {conditions[0]['code']['coding'][0]['display'].split(',')[0].lower()}."
    hpi = (
        f"{p['full_name']} is a {age}-year-old {p['gender']} with a history of {history_problem} "
        f"(ICD-10 {conditions[0]['code']['coding'][0]['code']}). The patient reports {symptoms}. "
        f"Vital signs at this {enc_label} include {obs_line}. Current medications: {med_line}. "
        f"The patient can be reached at {p['phone']} or {p['email']}. "
        f"Pharmacy and billing correspondence use the address {p['full_address']}."
    )
    assessment = (
        f"Assessment: {condition_line}. "
        f"Plan: ordered or performed {procedure_line}. Medication plan: {med_line}. "
        f"Patient education was provided, and follow-up was scheduled. "
        f"Electronically signed by Dr. {provider['name']}, {provider['specialty']}."
    )
    discharge = (
        f"The patient was seen for a {enc_label} on {enc_date}. "
        f"Diagnoses addressed: {', '.join(c['code']['coding'][0]['code'] for c in conditions)}. "
        f"Procedures performed: {', '.join(pr['code']['coding'][0]['code'] for pr in procedures)}. "
        f"Disposition: discharged home in stable condition. "
        f"Billing questions may be directed to {provider['phone']} or {provider['email']}."
    )
    text = (
        "CLINICAL NOTE\n"
        f"Patient: {p['full_name']}\n"
        f"MRN: {p['mrn']}\n"
        f"Date of Birth: {p['dob_us']}\n"
        f"SSN: {p['ssn']}\n"
        f"Date of Service: {enc_date}\n"
        f"Provider: Dr. {provider['name']}, {provider['specialty']}\n"
        f"Patient Phone: {p['phone']}\n"
        f"Patient Email: {p['email']}\n"
        f"Address: {p['full_address']}\n\n"
        f"Chief Complaint:\n{chief}\n\n"
        f"History of Present Illness:\n{hpi}\n\n"
        f"Assessment and Plan:\n{assessment}\n\n"
        f"Discharge Summary:\n{discharge}\n"
    )
    codes = []
    for c in conditions:
        codes.extend(_codes_of(c))
    for pr in procedures:
        codes.extend(_codes_of(pr))
    for o in observations:
        codes.extend(_codes_of(o))
    for m in medications:
        codes.extend(_codes_of(m, "medicationCodeableConcept"))

    injected = _injected_phi(
        {
            "NAME": [p["full_name"], provider["name"]],
            "MRN": [p["mrn"]],
            "SSN": [p["ssn"]],
            "PHONE": [p["phone"], provider["phone"]],
            "EMAIL": [p["email"], provider["email"]],
            "DATE": [p["dob_us"], enc_date],
            "ADDRESS": [p["full_address"]],
            "ZIP": [p["address"]["postalCode"]],
        },
        text,
    )
    return {
        "doc_id": f"NOTE-{p['patient_id']}-{encounter['id']}-clinical",
        "patient_id": p["patient_id"],
        "encounter_id": encounter["id"],
        "claim_id": _claim_for_encounter(claim),
        "doc_type": "clinical_note",
        "title": "Clinical Note",
        "service_date": enc_date,
        "text": text,
        "injected_phi": injected,
        "codes": codes,
        "provider": {"id": provider["id"], "name": provider["name"], "specialty": provider["specialty"]},
        "resource_ids": [r["id"] for r in conditions + procedures + observations + medications],
        "synthetic": True,
    }


def _payment_note(
    bundle: dict[str, Any],
    encounter: dict[str, Any],
    conditions: list[dict[str, Any]],
    procedures: list[dict[str, Any]],
    claim: dict[str, Any],
    claim_response: dict[str, Any],
    provider: dict[str, str],
    rng: random.Random,
) -> dict[str, Any]:
    p = _patient_facts(bundle)
    enc_date = encounter["period"]["start"]
    denial = _denial_info(claim_response)
    amounts = _amounts(claim_response)
    claim_id = claim["id"]
    claim_created = claim["created"]
    response_created = claim_response["created"]
    cpt_codes = [item["productOrService"]["coding"][0] for item in claim["item"]]
    primary_cpt = cpt_codes[0]
    dx = conditions[0]["code"]["coding"][0]
    reason_code = denial["code"] or "NONE"
    reason_text = denial["reason"] or "No denial was issued; the claim was paid according to contract."
    status_line = "DENIED" if denial["denied"] else "PAID"
    denial_section = (
        f"Denial Rationale:\n"
        f"Claim {claim_id} for date of service {enc_date} was {status_line} by the synthetic payer. "
        f"The denial reason code is {reason_code}: {reason_text} "
        f"The primary procedure associated with this decision is CPT {primary_cpt['code']} "
        f"({primary_cpt['display']}). The billed amount was {_money(amounts['submitted'])}; "
        f"the allowed amount was {_money(amounts['allowed'])} and the paid amount was "
        f"{_money(amounts['paid'])}. The supporting diagnosis is ICD-10 {dx['code']} "
        f"({dx['display']}). The claim was submitted on {claim_created} and adjudicated on "
        f"{response_created}."
    )
    if denial["denied"]:
        appeal = (
            f"Appeal Letter Draft:\n"
            f"Re: Claim {claim_id}; Patient {p['full_name']}; MRN {p['mrn']}; Date of Birth "
            f"{p['dob_us']}; Date of Service {enc_date}.\n"
            f"Dear Appeals Department,\n"
            f"We are writing to appeal the denial of claim {claim_id} (CARC {reason_code}) for "
            f"procedure CPT {primary_cpt['code']} performed on {enc_date}. The clinical "
            f"documentation supports medical necessity for ICD-10 {dx['code']} "
            f"({dx['display']}). The patient can be reached at {p['phone']} or {p['email']}, "
            f"and correspondence may be sent to {p['full_address']}.\n"
            f"Sincerely,\nDr. {provider['name']}, {provider['specialty']}\n"
            f"Billing contact: {provider['phone']} / {provider['email']}"
        )
    else:
        appeal = (
            f"Appeal Letter Draft:\n"
            f"No appeal is required for claim {claim_id}. The claim was paid in full under "
            f"the synthetic contract, and no denial reason code was assigned. Patient "
            f"{p['full_name']} (MRN {p['mrn']}) does not owe a denial-related balance. "
            f"Questions can be directed to {provider['phone']} or {provider['email']}."
        )
    text = (
        "PAYMENT OPERATIONS NOTE\n"
        f"Claim ID: {claim_id}\n"
        f"Patient: {p['full_name']}\n"
        f"MRN: {p['mrn']}\n"
        f"Date of Birth: {p['dob_us']}\n"
        f"SSN: {p['ssn']}\n"
        f"Date of Service: {enc_date}\n"
        f"Patient Phone: {p['phone']}\n"
        f"Patient Email: {p['email']}\n"
        f"Address: {p['full_address']}\n"
        f"Provider: Dr. {provider['name']}, {provider['specialty']}\n\n"
        f"{denial_section}\n\n"
        f"{appeal}\n"
    )
    codes = []
    for c in conditions:
        codes.extend(_codes_of(c))
    for pr in procedures:
        codes.extend(_codes_of(pr))
    if denial["denied"] and denial["code"]:
        codes.append({"system": "CARC", "code": denial["code"], "display": denial["reason"] or ""})

    injected = _injected_phi(
        {
            "NAME": [p["full_name"], provider["name"]],
            "MRN": [p["mrn"]],
            "SSN": [p["ssn"]],
            "PHONE": [p["phone"], provider["phone"]],
            "EMAIL": [p["email"], provider["email"]],
            "DATE": [p["dob_us"], enc_date, claim_created, response_created],
            "ADDRESS": [p["full_address"]],
            "ZIP": [p["address"]["postalCode"]],
        },
        text,
    )
    return {
        "doc_id": f"NOTE-{p['patient_id']}-{encounter['id']}-payment",
        "patient_id": p["patient_id"],
        "encounter_id": encounter["id"],
        "claim_id": claim_id,
        "doc_type": "denial_rationale",
        "title": "Payment Operations Note" if not denial["denied"] else "Denial and Appeal Note",
        "service_date": enc_date,
        "text": text,
        "injected_phi": injected,
        "codes": codes,
        "denial": {
            "denied": denial["denied"],
            "carc_code": denial["code"],
            "reason": denial["reason"],
            "cpt_codes": [c["code"] for c in cpt_codes],
            "primary_cpt": primary_cpt["code"],
            "submitted": amounts["submitted"],
            "allowed": amounts["allowed"],
            "paid": amounts["paid"],
            "claim_created": claim_created,
            "adjudicated_on": response_created,
        },
        "provider": {"id": provider["id"], "name": provider["name"], "specialty": provider["specialty"]},
        "resource_ids": [claim_id, claim_response["id"]],
        "synthetic": True,
    }


def generate_notes_for_bundle(
    bundle: dict[str, Any], notes_per_encounter: int = 2
) -> list[dict[str, Any]]:
    """Generate clinical and payment notes for every encounter in a bundle.

    Args:
        bundle: A validated synthetic FHIR R4 bundle.
        notes_per_encounter: 1 emits only clinical notes; >= 2 emits clinical and
            payment-operations notes.

    Returns:
        A list of note record dicts ready to be serialized as JSONL.
    """
    encounters = _by_type(bundle, "Encounter")
    conditions = _by_type(bundle, "Condition")
    procedures = _by_type(bundle, "Procedure")
    observations = _by_type(bundle, "Observation")
    medications = _by_type(bundle, "MedicationRequest")
    claims = {c["id"]: c for c in _by_type(bundle, "Claim")}
    responses = {r["request"]["reference"].split("/")[-1]: r for r in _by_type(bundle, "ClaimResponse")}
    notes: list[dict[str, Any]] = []
    for enc_idx, encounter in enumerate(encounters):
        enc_id = encounter["id"]
        enc_conditions = [
            c for c in conditions if c["encounter"]["reference"].endswith(f"/{enc_id}")
        ]
        enc_procedures = [
            p for p in procedures if p["encounter"]["reference"].endswith(f"/{enc_id}")
        ]
        enc_observations = [
            o for o in observations if o["encounter"]["reference"].endswith(f"/{enc_id}")
        ]
        enc_medications = [
            m for m in medications if m["encounter"]["reference"].endswith(f"/{enc_id}")
        ]
        expected_claim_id = enc_id.replace("ENC-", "CLM-")
        claim = claims.get(expected_claim_id)
        if claim is None:
            claim = list(claims.values())[enc_idx % len(claims)]
        response = responses.get(claim["id"])
        if response is None:
            raise ValueError(f"Claim {claim['id']} has no ClaimResponse in bundle {bundle.get('id')}")
        provider = _provider_for_encounter(encounter)
        rng = random.Random(f"{bundle.get('id')}:{enc_id}")
        notes.append(
            _clinical_note(
                bundle,
                encounter,
                enc_conditions,
                enc_procedures,
                enc_observations,
                enc_medications,
                claim,
                provider,
                rng,
            )
        )
        if notes_per_encounter >= 2:
            notes.append(
                _payment_note(
                    bundle,
                    encounter,
                    enc_conditions,
                    enc_procedures,
                    claim,
                    response,
                    provider,
                    rng,
                )
            )
    return notes
