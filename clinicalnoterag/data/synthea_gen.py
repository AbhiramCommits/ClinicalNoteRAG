"""Synthetic FHIR R4 corpus generator (Synthea-style, fully local).

This module does not download or use real patient records. It deterministically
synthesizes patients, encounters, coded conditions/procedures/observations,
medication orders, claims and claim responses. Every resource is validated with
the Pydantic models in :mod:`clinicalnoterag.data.fhir_models` before it is
written.

The generator is intentionally transparent: the code tables below are small
and readable so that downstream evaluation has exact ground truth for codes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from clinicalnoterag.data.fhir_models import (
    CARC_SYSTEM,
    CPT_SYSTEM,
    ICD10_SYSTEM,
    RXNORM_SYSTEM,
    SNOMED_SYSTEM,
    SYNTHETIC_TAG_SYSTEM,
    validate_bundle,
)
from clinicalnoterag.seeds import DEFAULT_SEED

# ---------------------------------------------------------------------------
# Small, readable synthetic code tables
# ---------------------------------------------------------------------------

ICD10_CONDITIONS: list[dict[str, str]] = [
    {"code": "E11.9", "display": "Type 2 diabetes mellitus without complications", "symptom": "increased thirst and fatigue"},
    {"code": "I10", "display": "Essential (primary) hypertension", "symptom": "occasional headaches"},
    {"code": "J18.9", "display": "Pneumonia, unspecified organism", "symptom": "productive cough and fever"},
    {"code": "I48.91", "display": "Unspecified atrial fibrillation", "symptom": "palpitations"},
    {"code": "N18.3", "display": "Chronic kidney disease, stage 3", "symptom": "reduced urine output and edema"},
    {"code": "J44.1", "display": "Chronic obstructive pulmonary disease with acute exacerbation", "symptom": "worsening dyspnea"},
    {"code": "I50.32", "display": "Chronic diastolic (congestive) heart failure", "symptom": "exertional dyspnea and leg swelling"},
    {"code": "M17.11", "display": "Unilateral primary osteoarthritis, right knee", "symptom": "right knee pain"},
    {"code": "K21.9", "display": "Gastro-esophageal reflux disease without esophagitis", "symptom": "heartburn"},
    {"code": "F32.9", "display": "Major depressive disorder, single episode, unspecified", "symptom": "low mood and poor sleep"},
    {"code": "E78.5", "display": "Hyperlipidemia, unspecified", "symptom": "no specific symptoms"},
    {"code": "G47.33", "display": "Obstructive sleep apnea", "symptom": "daytime sleepiness and snoring"},
    {"code": "I25.10", "display": "Atherosclerotic heart disease of native coronary artery", "symptom": "chest pressure with exertion"},
    {"code": "D64.9", "display": "Anemia, unspecified", "symptom": "fatigue and pallor"},
    {"code": "J45.909", "display": "Unspecified asthma, uncomplicated", "symptom": "wheezing"},
    {"code": "L03.115", "display": "Cellulitis of right lower limb", "symptom": "right leg redness and warmth"},
    {"code": "R07.9", "display": "Chest pain, unspecified", "symptom": "sharp chest pain"},
    {"code": "I63.9", "display": "Cerebral infarction, unspecified", "symptom": "left-sided weakness"},
    {"code": "K57.30", "display": "Diverticulosis of large intestine without perforation", "symptom": "intermittent abdominal pain"},
    {"code": "Z79.4", "display": "Long term (current) use of insulin", "symptom": "insulin dependence"},
]

CPT_PROCEDURES: list[dict[str, Any]] = [
    {"code": "99213", "display": "Office or other outpatient visit, established patient, low complexity", "price": 138.0},
    {"code": "99214", "display": "Office or other outpatient visit, established patient, moderate complexity", "price": 202.0},
    {"code": "99283", "display": "Emergency department visit, moderate severity", "price": 356.0},
    {"code": "99232", "display": "Subsequent hospital care, moderate complexity", "price": 221.0},
    {"code": "80053", "display": "Comprehensive metabolic panel", "price": 44.0},
    {"code": "83036", "display": "Hemoglobin A1c level", "price": 39.0},
    {"code": "93000", "display": "Electrocardiogram, routine ECG with interpretation", "price": 61.0},
    {"code": "71046", "display": "Radiologic examination, chest, two views", "price": 91.0},
    {"code": "36415", "display": "Collection of venous blood by venipuncture", "price": 18.0},
    {"code": "20610", "display": "Arthrocentesis, aspiration and/or injection, major joint", "price": 132.0},
    {"code": "45378", "display": "Colonoscopy, flexible, diagnostic", "price": 812.0},
    {"code": "27447", "display": "Total knee arthroplasty", "price": 8420.0},
    {"code": "96372", "display": "Therapeutic injection, subcutaneous or intramuscular", "price": 34.0},
    {"code": "97110", "display": "Therapeutic exercise, each 15 minutes", "price": 47.0},
    {"code": "43239", "display": "Upper gastrointestinal endoscopy with biopsy", "price": 743.0},
    {"code": "12002", "display": "Simple repair of superficial wounds, 2.6 cm to 7.5 cm", "price": 178.0},
]

SNOMED_OBSERVATIONS: list[dict[str, Any]] = [
    {"code": "271649006", "display": "Systolic blood pressure", "unit": "mmHg", "code_unit": "mm[Hg]", "low": 96, "high": 184, "decimals": 0},
    {"code": "271650006", "display": "Diastolic blood pressure", "unit": "mmHg", "code_unit": "mm[Hg]", "low": 58, "high": 112, "decimals": 0},
    {"code": "386725007", "display": "Body temperature", "unit": "Cel", "code_unit": "Cel", "low": 36.1, "high": 39.4, "decimals": 1},
    {"code": "86290005", "display": "Respiratory rate", "unit": "/min", "code_unit": "/min", "low": 12, "high": 28, "decimals": 0},
    {"code": "78564009", "display": "Heart rate", "unit": "/min", "code_unit": "/min", "low": 54, "high": 118, "decimals": 0},
    {"code": "27113001", "display": "Body weight", "unit": "kg", "code_unit": "kg", "low": 48, "high": 141, "decimals": 1},
]

RX_MEDICATIONS: list[dict[str, str]] = [
    {"code": "860975", "display": "Metformin hydrochloride 500 MG oral tablet"},
    {"code": "314076", "display": "Lisinopril 10 MG oral tablet"},
    {"code": "310798", "display": "Furosemide 40 MG oral tablet"},
    {"code": "197361", "display": "Atorvastatin calcium 20 MG oral tablet"},
    {"code": "745752", "display": "Albuterol sulfate 90 MCG inhaler"},
    {"code": "866924", "display": "Metoprolol succinate 25 MG oral tablet"},
    {"code": "856987", "display": "Sertraline 50 MG oral tablet"},
    {"code": "904475", "display": "Insulin glargine 100 UNITS/ML injection"},
    {"code": "308135", "display": "Omeprazole 20 MG oral capsule"},
    {"code": "313782", "display": "Acetaminophen 325 MG oral tablet"},
]

#: Claim Adjustment Reason Codes (CARC-style, abbreviated for synthetic use).
CARC_REASONS: dict[str, str] = {
    "CO-4": "The procedure code is inconsistent with the modifier used or a required modifier is missing.",
    "CO-11": "The diagnosis is inconsistent with the procedure.",
    "CO-16": "Claim/service lacks information or has submission/billing error(s) which is needed for adjudication.",
    "CO-18": "Exact duplicate claim/service.",
    "CO-22": "This care may be covered by another payer per coordination of benefits.",
    "CO-27": "Expenses were incurred after coverage terminated.",
    "CO-29": "The time limit for filing has expired.",
    "CO-45": "Charge exceeds fee schedule/maximum allowable or contracted fee arrangement.",
    "CO-50": "Non-covered services because this is not deemed a medical necessity by the payer.",
    "CO-97": "The benefit for this service is included in the payment/allowance for another service/procedure.",
    "CO-109": "Claim/service not covered by this payer/contractor. Send the claim to the correct payer.",
    "PR-1": "Deductible amount.",
}

ENCOUNTER_TYPES: list[dict[str, str]] = [
    {"code": "AMB", "display": "ambulatory", "label": "office visit"},
    {"code": "EMER", "display": "emergency", "label": "emergency department visit"},
    {"code": "IMP", "display": "inpatient encounter", "label": "inpatient stay"},
    {"code": "VR", "display": "virtual", "label": "telehealth visit"},
]

FIRST_NAMES: list[str] = [
    "Maria", "James", "Aisha", "Robert", "Priya", "Michael", "Elena", "David",
    "Fatima", "Daniel", "Grace", "Carlos", "Nia", "Thomas", "Sofia", "Ethan",
    "Leila", "Samuel", "Amara", "Lucas", "Rosa", "Kevin", "Mei", "Omar",
    "Hannah", "Victor", "Ingrid", "Jamal", "Lucia", "Peter", "Anika", "Marcus",
    "Teresa", "Ali", "Naomi", "Diego", "Claire", "Ravi", "Zoe", "Benjamin",
]

LAST_NAMES: list[str] = [
    "Gonzalez", "Bennett", "Okafor", "Chen", "Patel", "Rivera", "Novak", "Johnson",
    "Hassan", "Kim", "Miller", "Alvarez", "Thompson", "Mensah", "Rossi", "Nguyen",
    "Wilson", "Kaur", "Martinez", "Lee", "Fischer", "Brown", "Ahmed", "Silva",
    "Cohen", "Walker", "Diallo", "Costa", "Reed", "Iyer", "Murphy", "Osei",
    "Lopez", "Zhang", "Evans", "Haddad", "Moreau", "Singh", "Bailey", "Park",
]

STREETS = [
    "Maple Street", "Cedar Lane", "Oak Avenue", "Sunset Boulevard", "Elm Street",
    "Pine Road", "Birch Drive", "Willow Court", "Lakeview Drive", "Chestnut Street",
    "Aspen Way", "Orchard Lane",
]
CITIES = [
    "Springfield", "Riverton", "Fairview", "Lakeside", "Brookfield", "Georgetown",
    "Millbrook", "Cedar Falls",
]
STATES = ["IL", "OH", "TX", "MA", "CO", "OR", "NC", "AZ"]
AREA_CODES = ["202", "303", "415", "617", "713", "801", "919", "602"]
SPECIALTIES = [
    "Internal Medicine", "Family Medicine", "Cardiology", "Endocrinology",
    "Pulmonology", "Nephrology", "Gastroenterology", "Orthopedics",
]

FIXED_TIMESTAMP = "2025-07-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def stable_id(prefix: str, *parts: object) -> str:
    """Build a stable FHIR-safe id from a prefix and arbitrary parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _person_name(rng: random.Random) -> tuple[str, str]:
    return rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)


def _email(first: str, last: str, tag: str) -> str:
    return f"{first.lower()}.{last.lower()}.{tag}@example.com"


def _phone(rng: random.Random) -> str:
    return f"({rng.choice(AREA_CODES)}) 555-{rng.randint(100, 199):04d}"


def _address(rng: random.Random) -> dict[str, Any]:
    street = f"{rng.randint(100, 9899)} {rng.choice(STREETS)}"
    return {
        "line": [street],
        "city": rng.choice(CITIES),
        "state": rng.choice(STATES),
        "postalCode": f"{rng.randint(10000, 99999):05d}",
        "country": "US",
    }


def _ssn(rng: random.Random) -> str:
    # Area codes 900-999 are never issued by the SSA; this is deliberately fake.
    return f"9{rng.randint(0, 9):02d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def _date_between(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _fhir_ref(resource_type: str, resource_id: str, display: str | None = None) -> dict[str, str]:
    ref = {"reference": f"{resource_type}/{resource_id}"}
    if display:
        ref["display"] = display
    return ref


def _coding(system: str, code: str, display: str) -> dict[str, str]:
    return {"system": system, "code": code, "display": display}


def _concept(system: str, code: str, display: str) -> dict[str, Any]:
    return {"coding": [_coding(system, code, display)], "text": display}


def synthetic_meta() -> dict[str, Any]:
    """Return a Meta block that labels a resource as synthetic."""
    return {"tag": [_coding(SYNTHETIC_TAG_SYSTEM, "synthetic", "Synthetic data - not a real patient")]}


def _category(system: str, code: str, display: str) -> dict[str, Any]:
    return _concept(system, code, display)


# ---------------------------------------------------------------------------
# Providers (deterministic directory used by notes and claims)
# ---------------------------------------------------------------------------


def provider_directory() -> list[dict[str, str]]:
    """Return a deterministic, synthetic provider directory of 24 clinicians."""
    rng = random.Random(4242)
    providers: list[dict[str, str]] = []
    used: set[tuple[str, str]] = set()
    while len(providers) < 24:
        first, last = _person_name(rng)
        if (first, last) in used:
            continue
        used.add((first, last))
        idx = len(providers) + 1
        providers.append(
            {
                "id": f"PRAC-{idx:03d}",
                "name": f"{first} {last}",
                "specialty": rng.choice(SPECIALTIES),
                "phone": _phone(rng),
                "email": _email(first, last, f"prov{idx:03d}"),
            }
        )
    return providers


PROVIDERS: list[dict[str, str]] = provider_directory()


def provider_for(seed: int) -> dict[str, str]:
    """Return the provider assigned to a patient seed."""
    return PROVIDERS[seed % len(PROVIDERS)]


def all_known_names() -> list[str]:
    """All synthetic patient/provider names used by the corpus.

    Notes draw from these pools. Exposing the list lets the regex de-identifier
    recognize names even when Presidio is unavailable.
    """
    names = {f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES}
    names.update(f"{p['name']}" for p in PROVIDERS)
    return sorted(names)


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------


def generate_patient_bundle(seed: int, patient_index: int | None = None) -> dict[str, Any]:
    """Generate one synthetic patient as a valid FHIR R4 collection bundle.

    Args:
        seed: Seed that determines every clinical detail for the patient.
        patient_index: Optional 1-based index used for human-readable ids. When
            omitted it is derived from ``seed``.

    Returns:
        A FHIR R4 Bundle dict (also validated with Pydantic before return).

    Raises:
        pydantic.ValidationError: If a generated resource is structurally invalid.
    """
    rng = random.Random(seed)
    idx = patient_index if patient_index is not None else max(1, seed - DEFAULT_SEED + 1)
    patient_id = f"PAT-{idx:04d}"
    patient_num = f"{idx:04d}"
    provider = PROVIDERS[idx % len(PROVIDERS)]

    first, last = _person_name(rng)
    full_name = f"{first} {last}"
    gender = rng.choice(["male", "female"])
    dob = _date_between(rng, date(1936, 1, 1), date(2004, 12, 31))
    mrn = f"MRN-{rng.choice([2022, 2023, 2024])}-{rng.randint(100000, 999999):06d}"
    ssn = _ssn(rng)
    phone = _phone(rng)
    email = _email(first, last, f"p{idx:04d}")
    address = _address(rng)

    entries: list[dict[str, Any]] = []

    patient = {
        "resourceType": "Patient",
        "id": patient_id,
        "meta": synthetic_meta(),
        "identifier": [
            {"system": "urn:oid:2.16.840.1.113883.19.5.99999.1", "value": mrn},
            {"system": "http://hl7.org/fhir/sid/us-ssn", "value": ssn},
        ],
        "active": True,
        "name": [{"use": "official", "family": last, "given": [first]}],
        "gender": gender,
        "birthDate": dob.isoformat(),
        "address": [address],
        "telecom": [
            {"system": "phone", "value": phone, "use": "home"},
            {"system": "email", "value": email, "use": "home"},
        ],
        "communication": [{"language": {"coding": [_coding("urn:ietf:bcp:47", "en-US", "English")]}}],
        "managingOrganization": _fhir_ref("Organization", "ORG-001", "Synthetic Health System"),
    }
    entries.append({"fullUrl": f"urn:uuid:{patient_id}", "resource": patient})

    n_encounters = rng.randint(1, 4)
    encounter_start = date(2023, 1, 1)
    encounter_end = date(2025, 6, 30)
    used_condition_codes: set[str] = set()
    used_encounter_dates: set[date] = set()

    for enc_idx in range(1, n_encounters + 1):
        enc_type = rng.choice(ENCOUNTER_TYPES)
        enc_date = _date_between(rng, encounter_start, encounter_end)
        while enc_date in used_encounter_dates:  # keep per-patient encounter dates unique
            enc_date = _date_between(rng, encounter_start, encounter_end)
        used_encounter_dates.add(enc_date)
        stay_days = rng.randint(1, 4) if enc_type["code"] == "IMP" else 0
        enc_end_date = enc_date + timedelta(days=stay_days)
        enc_id = f"ENC-{patient_num}-{enc_idx:02d}"

        # Conditions for this encounter (avoid duplicates within a patient when possible)
        condition_candidates = [c for c in ICD10_CONDITIONS if c["code"] not in used_condition_codes]
        if not condition_candidates:
            condition_candidates = ICD10_CONDITIONS
        n_conditions = rng.randint(1, 2)
        conditions = rng.sample(condition_candidates, n_conditions)
        condition_resources: list[dict[str, Any]] = []
        for cond_idx, condition in enumerate(conditions, start=1):
            used_condition_codes.add(condition["code"])
            cond_id = f"CON-{patient_num}-{enc_idx:02d}-{cond_idx}"
            resource = {
                "resourceType": "Condition",
                "id": cond_id,
                "meta": synthetic_meta(),
                "clinicalStatus": _concept(
                    "http://terminology.hl7.org/CodeSystem/condition-clinical", "active", "Active"
                ),
                "verificationStatus": _concept(
                    "http://terminology.hl7.org/CodeSystem/condition-ver-status", "confirmed", "Confirmed"
                ),
                "category": [
                    _concept(
                        "http://terminology.hl7.org/CodeSystem/condition-category",
                        "encounter-diagnosis",
                        "Encounter Diagnosis",
                    )
                ],
                "code": _concept(ICD10_SYSTEM, condition["code"], condition["display"]),
                "subject": _fhir_ref("Patient", patient_id, full_name),
                "encounter": _fhir_ref("Encounter", enc_id),
                "onsetDateTime": enc_date.isoformat(),
                "recordedDate": f"{enc_date.isoformat()}T10:00:00Z",
            }
            condition_resources.append(resource)
            entries.append({"fullUrl": f"urn:uuid:{cond_id}", "resource": resource})

        # Procedures, including an E/M code appropriate to the encounter type
        em_code = {"AMB": "99213", "EMER": "99283", "IMP": "99232", "VR": "99214"}[enc_type["code"]]
        em_proc = next(p for p in CPT_PROCEDURES if p["code"] == em_code)
        other_procs = [p for p in CPT_PROCEDURES if p["code"] not in {em_code}]
        n_other = rng.randint(0, 2)
        procedures = [em_proc, *rng.sample(other_procs, n_other)]
        procedure_resources: list[dict[str, Any]] = []
        for proc_idx, proc in enumerate(procedures, start=1):
            proc_id = f"PROC-{patient_num}-{enc_idx:02d}-{proc_idx}"
            resource = {
                "resourceType": "Procedure",
                "id": proc_id,
                "meta": synthetic_meta(),
                "status": "completed",
                "code": _concept(CPT_SYSTEM, proc["code"], proc["display"]),
                "subject": _fhir_ref("Patient", patient_id, full_name),
                "encounter": _fhir_ref("Encounter", enc_id),
                "performedPeriod": {"start": enc_date.isoformat(), "end": enc_end_date.isoformat()},
                "performer": [_fhir_ref("Practitioner", provider["id"], f"Dr. {provider['name']}")],
            }
            procedure_resources.append(resource)
            entries.append({"fullUrl": f"urn:uuid:{proc_id}", "resource": resource})

        # Observations
        observations = rng.sample(SNOMED_OBSERVATIONS, rng.randint(2, 4))
        for obs_idx, obs in enumerate(observations, start=1):
            obs_id = f"OBS-{patient_num}-{enc_idx:02d}-{obs_idx}"
            value = round(rng.uniform(obs["low"], obs["high"]), obs["decimals"])
            resource = {
                "resourceType": "Observation",
                "id": obs_id,
                "meta": synthetic_meta(),
                "status": "final",
                "category": [
                    _concept(
                        "http://terminology.hl7.org/CodeSystem/observation-category",
                        "vital-signs",
                        "Vital Signs",
                    )
                ],
                "code": _concept(SNOMED_SYSTEM, obs["code"], obs["display"]),
                "subject": _fhir_ref("Patient", patient_id, full_name),
                "encounter": _fhir_ref("Encounter", enc_id),
                "effectiveDateTime": f"{enc_date.isoformat()}T10:{rng.randint(0, 59):02d}:00Z",
                "valueQuantity": {
                    "value": value,
                    "unit": obs["unit"],
                    "system": "http://unitsofmeasure.org",
                    "code": obs["code_unit"],
                },
            }
            entries.append({"fullUrl": f"urn:uuid:{obs_id}", "resource": resource})

        # Medication orders
        meds = rng.sample(RX_MEDICATIONS, rng.randint(0, 2))
        for med_idx, med in enumerate(meds, start=1):
            med_id = f"MED-{patient_num}-{enc_idx:02d}-{med_idx}"
            resource = {
                "resourceType": "MedicationRequest",
                "id": med_id,
                "meta": synthetic_meta(),
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": _concept(RXNORM_SYSTEM, med["code"], med["display"]),
                "subject": _fhir_ref("Patient", patient_id, full_name),
                "encounter": _fhir_ref("Encounter", enc_id),
                "authoredOn": enc_date.isoformat(),
                "requester": _fhir_ref("Practitioner", provider["id"], f"Dr. {provider['name']}"),
                "dosageInstruction": [{"text": "Take as directed by the prescribing clinician."}],
            }
            entries.append({"fullUrl": f"urn:uuid:{med_id}", "resource": resource})

        # Encounter resource (added after children for narrative ordering)
        encounter = {
            "resourceType": "Encounter",
            "id": enc_id,
            "meta": synthetic_meta(),
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": enc_type["code"],
                "display": enc_type["display"],
            },
            "type": [
                _concept(
                    "http://terminology.hl7.org/CodeSystem/encounter-type",
                    enc_type["code"],
                    enc_type["label"],
                )
            ],
            "subject": _fhir_ref("Patient", patient_id, full_name),
            "participant": [
                {
                    "individual": _fhir_ref("Practitioner", provider["id"], f"Dr. {provider['name']}"),
                    "type": [
                        _concept(
                            "http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
                            "PPRF",
                            "primary performer",
                        )
                    ],
                }
            ],
            "period": {"start": enc_date.isoformat(), "end": enc_end_date.isoformat()},
            "reasonCode": [
                {
                    "coding": [
                        _coding(ICD10_SYSTEM, c["code"], c["display"])
                        for c in conditions
                    ],
                    "text": "; ".join(c["display"] for c in conditions),
                }
            ],
            "serviceProvider": _fhir_ref("Organization", "ORG-001", "Synthetic Health System"),
        }
        entries.append({"fullUrl": f"urn:uuid:{enc_id}", "resource": encounter})

        # Claim and ClaimResponse
        claim_id = f"CLM-{patient_num}-{enc_idx:02d}"
        claim_items: list[dict[str, Any]] = []
        total_value = 0.0
        for item_idx, proc in enumerate(procedures, start=1):
            amount = round(proc["price"] * rng.uniform(0.92, 1.18), 2)
            total_value += amount
            claim_items.append(
                {
                    "sequence": item_idx,
                    "productOrService": _concept(CPT_SYSTEM, proc["code"], proc["display"]),
                    "net": {"value": amount, "currency": "USD"},
                    "servicedDate": enc_date.isoformat(),
                    "diagnosisSequence": [1],
                }
            )
        claim_created = enc_date + timedelta(days=rng.randint(1, 5))
        claim = {
            "resourceType": "Claim",
            "id": claim_id,
            "meta": synthetic_meta(),
            "status": "active",
            "type": _concept("http://terminology.hl7.org/CodeSystem/claim-type", "professional", "Professional"),
            "patient": _fhir_ref("Patient", patient_id, full_name),
            "created": claim_created.isoformat(),
            "provider": _fhir_ref("Practitioner", provider["id"], f"Dr. {provider['name']}"),
            "priority": _concept("http://terminology.hl7.org/CodeSystem/processpriority", "normal", "Normal"),
            "insurance": [
                {
                    "sequence": 1,
                    "focal": True,
                    "coverage": _fhir_ref("Coverage", f"COV-{patient_num}", "Synthetic PPO Plan"),
                }
            ],
            "diagnosis": [
                {
                    "sequence": i + 1,
                    "diagnosisCodeableConcept": _concept(ICD10_SYSTEM, c["code"], c["display"]),
                }
                for i, c in enumerate(conditions)
            ],
            "item": claim_items,
            "total": {"value": round(total_value, 2), "currency": "USD"},
            "billablePeriod": {"start": enc_date.isoformat(), "end": enc_end_date.isoformat()},
        }
        entries.append({"fullUrl": f"urn:uuid:{claim_id}", "resource": claim})

        denied = rng.random() < 0.30
        denial_code = rng.choice(list(CARC_REASONS)) if denied else None
        allowed_value = 0.0 if denied else round(total_value * rng.uniform(0.62, 0.88), 2)
        paid_value = 0.0 if denied else round(allowed_value * rng.uniform(0.70, 0.95), 2)
        claim_response_id = f"CLMR-{patient_num}-{enc_idx:02d}"
        adjudication: list[dict[str, Any]] = [
            {
                "category": _category(
                    "http://terminology.hl7.org/CodeSystem/adjudication", "submitted", "Submitted Amount"
                ),
                "amount": {"value": round(total_value, 2), "currency": "USD"},
            },
            {
                "category": _category(
                    "http://terminology.hl7.org/CodeSystem/adjudication", "eligible", "Eligible Amount"
                ),
                "amount": {"value": allowed_value, "currency": "USD"},
            },
            {
                "category": _category(
                    "http://terminology.hl7.org/CodeSystem/adjudication", "benefit", "Benefit Amount"
                ),
                "amount": {"value": paid_value, "currency": "USD"},
            },
        ]
        if denied and denial_code:
            adjudication.append(
                {
                    "category": _category(
                        "http://terminology.hl7.org/CodeSystem/adjudication", "denial", "Denial"
                    ),
                    "reason": _concept(CARC_SYSTEM, denial_code, CARC_REASONS[denial_code]),
                    "amount": {"value": 0.0, "currency": "USD"},
                }
            )
        claim_response = {
            "resourceType": "ClaimResponse",
            "id": claim_response_id,
            "meta": synthetic_meta(),
            "status": "active",
            "type": _concept("http://terminology.hl7.org/CodeSystem/claim-type", "professional", "Professional"),
            "patient": _fhir_ref("Patient", patient_id, full_name),
            "created": (claim_created + timedelta(days=rng.randint(3, 21))).isoformat(),
            "request": _fhir_ref("Claim", claim_id),
            "outcome": "complete",
            "disposition": "Denied: " + CARC_REASONS[denial_code] if denial_code else "Paid per contract.",
            "adjudication": adjudication,
            "total": adjudication,
            "item": [
                {
                    "itemSequence": item["sequence"],
                    "adjudication": adjudication,
                }
                for item in claim_items
            ],
        }
        entries.append({"fullUrl": f"urn:uuid:{claim_response_id}", "resource": claim_response})

    bundle = {
        "resourceType": "Bundle",
        "id": f"BUNDLE-{patient_id}",
        "type": "collection",
        "timestamp": FIXED_TIMESTAMP,
        "entry": entries,
    }
    validate_bundle(bundle)
    return bundle


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------


@dataclass
class CorpusStats:
    """Summary statistics returned by :func:`generate_corpus`."""

    n_patients: int
    n_encounters: int
    n_conditions: int
    n_procedures: int
    n_observations: int
    n_medications: int
    n_claims: int
    n_denied_claims: int
    n_documents: int
    n_clinical_notes: int
    n_payment_notes: int
    seed: int
    out_dir: str
    manifest_path: str
    known_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _resource_counts(bundle: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in bundle.get("entry", []):
        rtype = entry["resource"]["resourceType"]
        counts[rtype] = counts.get(rtype, 0) + 1
    return counts


def generate_corpus(
    n_patients: int,
    out_dir: str | Path,
    seed: int = DEFAULT_SEED,
    notes_per_encounter: int = 2,
) -> CorpusStats:
    """Generate a complete synthetic corpus: bundles, notes, and a manifest.

    Writes one FHIR JSON bundle per patient under ``out_dir/bundles`` and one
    JSONL file of notes per patient under ``out_dir/notes``. Also writes
    ``out_dir/corpus_manifest.json`` with counts and the known synthetic names
    used by the de-identification layer.

    Args:
        n_patients: Number of synthetic patients to generate (>= 1).
        out_dir: Destination directory.
        seed: Base seed; patient ``i`` uses ``seed + i``.
        notes_per_encounter: Number of note documents per encounter (currently
            the generator always emits one clinical and one payment note).

    Returns:
        CorpusStats with real counts of everything written.
    """
    if n_patients < 1:
        raise ValueError("n_patients must be >= 1")
    from clinicalnoterag.data.notes_gen import generate_notes_for_bundle

    out = Path(out_dir)
    bundle_dir = out / "bundles"
    note_dir = out / "notes"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    note_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    n_encounters = n_denied = 0
    n_clinical = n_payment = 0
    generated_at = datetime.now(UTC).isoformat()
    manifest_patients: list[dict[str, Any]] = []

    for i in range(1, n_patients + 1):
        patient_seed = seed + i - 1
        bundle = generate_patient_bundle(patient_seed, patient_index=i)
        bundle_path = bundle_dir / f"patient_{i:04d}.json"
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

        counts = _resource_counts(bundle)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        n_encounters += counts.get("Encounter", 0)
        n_denied += sum(
            1
            for e in bundle["entry"]
            if e["resource"]["resourceType"] == "ClaimResponse"
            and any(a["category"]["coding"][0]["code"] == "denial" for a in e["resource"]["adjudication"])
        )

        notes = generate_notes_for_bundle(bundle, notes_per_encounter=notes_per_encounter)
        note_path = note_dir / f"patient_{i:04d}.jsonl"
        with note_path.open("w", encoding="utf-8") as fh:
            for note in notes:
                fh.write(json.dumps(note) + "\n")
        for note in notes:
            if note["doc_type"] == "clinical_note":
                n_clinical += 1
            else:
                n_payment += 1

        patient_res = next(
            e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient"
        )
        manifest_patients.append(
            {
                "patient_id": patient_res["id"],
                "seed": patient_seed,
                "bundle": str(bundle_path.relative_to(out)),
                "notes": str(note_path.relative_to(out)),
                "resource_counts": counts,
                "n_notes": len(notes),
                "has_denied_claim": any(
                    e["resource"]["resourceType"] == "ClaimResponse"
                    and any(
                        a["category"]["coding"][0]["code"] == "denial"
                        for a in e["resource"]["adjudication"]
                    )
                    for e in bundle["entry"]
                ),
            }
        )

    known_names = all_known_names()
    manifest = {
        "corpus_version": "1.0.0",
        "synthetic": True,
        "generated_at": generated_at,
        "seed": seed,
        "n_patients": n_patients,
        "n_encounters": n_encounters,
        "n_documents": n_clinical + n_payment,
        "n_clinical_notes": n_clinical,
        "n_payment_notes": n_payment,
        "n_denied_claims": n_denied,
        "resource_counts": totals,
        "known_names": known_names,
        "patients": manifest_patients,
        "code_systems": {
            "condition": ICD10_SYSTEM,
            "procedure": CPT_SYSTEM,
            "observation": SNOMED_SYSTEM,
            "medication": RXNORM_SYSTEM,
            "adjustment_reason": CARC_SYSTEM,
        },
        "note": (
            "All records are synthetic and generated by clinicalnoterag.data.synthea_gen. "
            "No real patient data was used."
        ),
    }
    manifest_path = out / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return CorpusStats(
        n_patients=n_patients,
        n_encounters=n_encounters,
        n_conditions=totals.get("Condition", 0),
        n_procedures=totals.get("Procedure", 0),
        n_observations=totals.get("Observation", 0),
        n_medications=totals.get("MedicationRequest", 0),
        n_claims=totals.get("Claim", 0),
        n_denied_claims=n_denied,
        n_documents=n_clinical + n_payment,
        n_clinical_notes=n_clinical,
        n_payment_notes=n_payment,
        seed=seed,
        out_dir=str(out),
        manifest_path=str(manifest_path),
        known_names=known_names,
    )


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load and validate a FHIR bundle from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_bundle(data)
    return data


def iter_resources(bundle: dict[str, Any], resource_type: str | None = None):
    """Yield resources in a bundle, optionally filtered by resourceType."""
    for entry in bundle.get("entry", []):
        resource = entry["resource"]
        if resource_type is None or resource["resourceType"] == resource_type:
            yield resource


__all__ = [
    "CARC_REASONS",
    "CPT_PROCEDURES",
    "CorpusStats",
    "ENCOUNTER_TYPES",
    "ICD10_CONDITIONS",
    "PROVIDERS",
    "RX_MEDICATIONS",
    "SNOMED_OBSERVATIONS",
    "all_known_names",
    "generate_corpus",
    "generate_patient_bundle",
    "iter_resources",
    "load_bundle",
    "provider_directory",
    "provider_for",
]
