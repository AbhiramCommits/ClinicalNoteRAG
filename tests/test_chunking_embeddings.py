"""Chunking and embedding backend tests."""

from __future__ import annotations

import numpy as np
import pytest

from clinicalnoterag.data.synthea_gen import generate_patient_bundle
from clinicalnoterag.embeddings.backends import HashingTfidfBackend
from clinicalnoterag.pipeline.chunking import (
    chunk_bundle,
    chunk_note,
    content_hash,
    split_sections,
    token_windows,
)


def test_split_sections_on_note_headings():
    text = (
        "CLINICAL NOTE\nPatient: X\n\nChief Complaint:\nChest pain.\n\n"
        "History of Present Illness:\nTwo days of pain.\n\n"
        "Assessment and Plan:\nRule out ACS.\n"
    )
    sections = split_sections(text)
    names = [name for name, _ in sections]
    assert names[0] == "header"
    assert "chief_complaint" in names
    assert "history_of_present_illness" in names
    assert "assessment_and_plan" in names


def test_token_windows_boundaries_and_overlap():
    text = " ".join(f"t{i}" for i in range(600))
    windows = token_windows(text, target_tokens=256, overlap_tokens=32)
    assert len(windows) == 3
    first = windows[0].split()
    second = windows[1].split()
    third = windows[2].split()
    assert len(first) == 256
    assert len(second) == 256
    assert len(third) == 152
    assert first[-32:] == second[:32]
    assert second[-32:] == third[:32]


def test_token_windows_rejects_bad_overlap():
    with pytest.raises(ValueError):
        token_windows("a b c", target_tokens=10, overlap_tokens=10)


def test_chunk_note_marks_section_and_metadata(sample_notes):
    note = sample_notes[0]
    chunks = chunk_note(note, target_tokens=64, overlap_tokens=8)
    assert chunks
    assert all(chunk.metadata["patient_id"] == note["patient_id"] for chunk in chunks)
    assert all(chunk.chunk_id.startswith(note["doc_id"]) for chunk in chunks)
    sections = {chunk.metadata["section"] for chunk in chunks}
    assert "chief_complaint" in sections or "assessment_and_plan" in sections


def test_chunk_bundle_emits_fact_chunks(sample_bundles):
    chunks = chunk_bundle(sample_bundles[0])
    types = {chunk.metadata["resource_type"] for chunk in chunks}
    assert {"Condition", "Procedure", "Claim", "ClaimResponse", "Encounter"} <= types
    assert all(chunk.metadata["doc_type"] == "fhir_fact" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("FACT::") for chunk in chunks)


def test_fact_chunk_contains_codes(sample_bundles):
    chunks = chunk_bundle(sample_bundles[0])
    condition = next(chunk for chunk in chunks if chunk.metadata["resource_type"] == "Condition")
    assert condition.metadata["code"] in condition.text
    assert condition.metadata["code_system"] == "ICD-10"


def test_content_hash_is_deterministic():
    assert content_hash("abc", "doc", 1) == content_hash("abc", "doc", 1)
    assert content_hash("abc", "doc", 1) != content_hash("abc", "doc", 2)


def test_hashing_embedding_is_deterministic_and_normalized():
    backend = HashingTfidfBackend(dim=128)
    texts = ["essential hypertension ICD-10 I10", "claim denied CARC CO-16"]
    first = backend.encode(texts)
    second = backend.encode(texts)
    assert first.shape == (2, 128)
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


def test_hashing_fit_state_round_trip():
    texts = ["alpha beta", "beta gamma", "delta epsilon"]
    backend = HashingTfidfBackend(dim=64)
    backend.fit(texts)
    state = backend.state()
    clone = HashingTfidfBackend(dim=64, idf=np.asarray(state["idf"]))
    assert np.allclose(backend.encode(texts), clone.encode(texts))


def test_hashing_empty_input():
    backend = HashingTfidfBackend(dim=64)
    assert backend.encode([]).shape == (0, 64)


def test_same_seed_generates_identical_bundles():
    assert generate_patient_bundle(123, patient_index=7) == generate_patient_bundle(
        123, patient_index=7
    )
