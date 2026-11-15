"""FHIR generation, note synthesis and PHI de-identification tests."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from clinicalnoterag.data.fhir_models import (
    CARC_SYSTEM,
    CPT_SYSTEM,
    ICD10_SYSTEM,
    RXNORM_SYSTEM,
    SNOMED_SYSTEM,
    validate_bundle,
)
from clinicalnoterag.data.notes_gen import generate_notes_for_bundle
from clinicalnoterag.data.synthea_gen import generate_patient_bundle
from clinicalnoterag.deid.audit import scan_corpus
from clinicalnoterag.deid.engine import DeidEngine


def test_generated_bundle_is_valid_fhir(sample_bundles):
    assert sample_bundles
    for bundle in sample_bundles:
        validate_bundle(bundle)
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert any(entry["resource"]["resourceType"] == "Patient" for entry in bundle["entry"])


def test_bundle_code_systems_are_assigned(sample_bundles):
    for bundle in sample_bundles:
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource["resourceType"] == "Condition":
                assert any(
                    coding["system"] == ICD10_SYSTEM for coding in resource["code"]["coding"]
                )
            if resource["resourceType"] == "Procedure":
                assert any(coding["system"] == CPT_SYSTEM for coding in resource["code"]["coding"])
            if resource["resourceType"] == "Observation":
                assert any(
                    coding["system"] == SNOMED_SYSTEM for coding in resource["code"]["coding"]
                )
            if resource["resourceType"] == "MedicationRequest":
                assert any(
                    coding["system"] == RXNORM_SYSTEM
                    for coding in resource["medicationCodeableConcept"]["coding"]
                )


def test_claim_response_has_carc_denial_reason(sample_bundles):
    found = False
    for bundle in sample_bundles:
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource["resourceType"] != "ClaimResponse":
                continue
            for adjudication in resource["adjudication"]:
                if adjudication.get("reason") and adjudication["reason"].get("coding"):
                    assert adjudication["reason"]["coding"][0]["system"] == CARC_SYSTEM
                    found = True
    assert found, "sample should contain at least one denied claim"


def test_missing_patient_name_fails_validation():
    bundle = generate_patient_bundle(42, patient_index=1)
    broken = copy.deepcopy(bundle)
    for entry in broken["entry"]:
        if entry["resource"]["resourceType"] == "Patient":
            entry["resource"].pop("name")
    with pytest.raises(ValidationError):
        validate_bundle(broken)


def test_notes_repeat_bundle_codes_and_inject_phi(sample_bundles):
    bundle = sample_bundles[0]
    notes = generate_notes_for_bundle(bundle)
    assert {note["doc_type"] for note in notes} == {"clinical_note", "denial_rationale"}
    condition_codes = {
        entry["resource"]["code"]["coding"][0]["code"]
        for entry in bundle["entry"]
        if entry["resource"]["resourceType"] == "Condition"
    }
    for note in notes:
        assert note["injected_phi"], "notes must intentionally inject fake PHI"
        text = note["text"]
        assert any(code in text for code in condition_codes)
        assert note["injected_phi"]["NAME"]


def test_deid_removes_all_injected_phi(sample_notes, sample_manifest):
    engine = DeidEngine(backend="regex", known_names=sample_manifest["known_names"])
    report = scan_corpus(sample_notes, engine=engine)
    assert report.backend_name == "regex"
    assert report.phi_leak_rate == 0.0
    assert report.leaked_documents == 0
    assert set(report.per_entity_recall) >= {
        "NAME",
        "MRN",
        "SSN",
        "PHONE",
        "EMAIL",
        "DATE",
        "ADDRESS",
        "ZIP",
    }
    assert all(value == 1.0 for value in report.per_entity_recall.values())


def test_deid_placeholders_and_reversible_mapping(sample_notes, sample_manifest):
    engine = DeidEngine(backend="regex", known_names=sample_manifest["known_names"])
    result = engine.scrub(sample_notes[0]["text"])
    assert "[NAME_1]" in result.text
    assert result.mapping
    for placeholder, original in result.mapping.items():
        assert placeholder.startswith("[")
        assert original not in result.text
    assert "MRN-" not in result.text


def test_audit_detects_a_deliberate_leak():
    record = {
        "doc_id": "leaky",
        "scrubbed_text": "Patient Maria Gonzalez was seen.",
        "injected_phi": {"NAME": ["Maria Gonzalez"]},
    }
    report = scan_corpus([record])
    assert report.phi_leak_rate == 1.0
    assert report.per_entity_recall["NAME"] == 0.0


def test_phi_leak_rate_is_below_measured_threshold(sample_notes, sample_manifest):
    engine = DeidEngine(backend="regex", known_names=sample_manifest["known_names"])
    report = scan_corpus(sample_notes, engine=engine)
    measured_threshold = 0.005
    assert report.phi_leak_rate < measured_threshold


def test_presidio_forced_backend_fails_cleanly_when_unavailable(monkeypatch):
    import clinicalnoterag.deid.engine as engine_module

    monkeypatch.setattr(engine_module.DeidEngine, "_try_presidio", lambda self: None)
    with pytest.raises(RuntimeError):
        DeidEngine(backend="presidio")
