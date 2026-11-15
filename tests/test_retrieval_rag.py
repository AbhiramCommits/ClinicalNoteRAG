"""Retrieval, citation verification, abstention and code extraction tests."""

from __future__ import annotations

import math

from clinicalnoterag.rag.answerer import key_entities
from clinicalnoterag.rag.codes import extract_codes
from clinicalnoterag.rag.generators import ABSTENTION_TEXT
from clinicalnoterag.rag.retriever import RetrievedChunk
from clinicalnoterag.rag.verify import verify_answer


def test_retriever_supports_dense_bm25_and_hybrid(tiny_system):
    filters = {
        "patient_id": "PAT-0001",
        "resource_type": "Condition",
        "doc_type": "fhir_fact",
    }
    for mode in ("hybrid", "dense", "bm25"):
        hits = tiny_system.retriever.retrieve(
            "What diagnosis was documented for this encounter?", k=5, filters=filters, mode=mode
        )
        assert hits, mode
        assert all(hit.metadata["patient_id"] == "PAT-0001" for hit in hits)


def test_reciprocal_rank_fusion_order(tiny_system, monkeypatch):
    retriever = tiny_system.retriever
    ids = [record.id for record in retriever.records[:3]]
    assert len(ids) == 3
    monkeypatch.setattr(retriever, "_dense", lambda *args, **kwargs: [(ids[0], 1.0), (ids[1], 0.9)])
    monkeypatch.setattr(
        retriever, "_sparse", lambda *args, **kwargs: [(ids[2], 1.0), (ids[0], 0.8)]
    )
    hits = retriever.retrieve("anything", k=3, mode="hybrid")
    assert hits[0].record_id == ids[0]
    assert math.isclose(hits[0].score, 1 / 61 + 1 / 62, rel_tol=1e-9)


def test_retriever_metadata_filter_prevents_cross_patient(tiny_system):
    hits = tiny_system.retriever.retrieve(
        "claim denied appeal", k=20, filters={"patient_id": "PAT-0002"}
    )
    assert hits
    assert all(hit.metadata.get("patient_id") == "PAT-0002" for hit in hits)


def _chunk(text: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        record_id=chunk_id,
        chunk_id=chunk_id,
        text=text,
        metadata={"patient_id": "PAT-0001", "doc_type": "fhir_fact"},
        score=0.5,
    )


def test_verifier_supports_a_cited_sentence():
    context = _chunk("Systolic blood pressure was 120 mmHg on 2024-01-01.")
    result = verify_answer("Systolic blood pressure was 120 mmHg [1].", {1: context})
    assert result.unsupported_claim is False
    assert result.citation_coverage == 1.0


def test_verifier_catches_a_fabricated_sentence():
    context = _chunk("Systolic blood pressure was 120 mmHg.")
    result = verify_answer("The patient has metastatic cancer [1].", {1: context})
    assert result.unsupported_claim is True
    assert result.unsupported_sentences[0].reason


def test_verifier_requires_a_citation_marker():
    context = _chunk("Systolic blood pressure was 120 mmHg.")
    result = verify_answer("Systolic blood pressure was 120 mmHg.", {1: context})
    assert result.unsupported_claim is True


def test_answerer_returns_citations(tiny_system):
    answer = tiny_system.answerer.answer(
        "What diagnoses were documented for encounter ENC-0001-01?",
        filters={
            "patient_id": "PAT-0001",
            "encounter_id": "ENC-0001-01",
            "resource_type": "Condition",
            "doc_type": "fhir_fact",
        },
        k=5,
    )
    assert not answer.abstained
    assert answer.text != ABSTENTION_TEXT
    assert answer.citations
    assert all(citation.chunk_id for citation in answer.citations)
    assert 0.0 <= answer.confidence <= 1.0
    assert answer.retrieval_scores


def test_answerer_abstains_on_missing_diagnosis_code(tiny_system):
    answer = tiny_system.answerer.answer(
        "What diagnosis code Z99.999 was documented for patient PAT-0001?",
        filters={
            "patient_id": "PAT-0001",
            "resource_type": "Condition",
            "doc_type": "fhir_fact",
        },
        k=5,
    )
    assert answer.abstained
    assert answer.text == ABSTENTION_TEXT
    assert answer.citations == []
    assert "Z99.999" in (answer.abstention_reason or "")


def test_answerer_abstains_on_absent_observation(tiny_system):
    answer = tiny_system.answerer.answer(
        "What was the LDL cholesterol recorded for patient PAT-0001?",
        filters={
            "patient_id": "PAT-0001",
            "resource_type": "Observation",
            "doc_type": "fhir_fact",
        },
        k=10,
    )
    assert answer.abstained
    assert answer.text == ABSTENTION_TEXT


def test_key_entities_extracts_hard_identifiers():
    entities = key_entities(
        "Why was claim CLM-0002-01 denied for patient PAT-0002 with code E11.9?"
    )
    assert "CLM-0002-01" in entities
    assert "PAT-0002" in entities
    assert "E11.9" in entities


def test_code_extraction_returns_expected_systems():
    text = (
        "ICD-10 E11.9; CPT 99213; SNOMED 271649006; RxNorm 860975; "
        "CARC CO-16; zip 62704 and amount 12345"
    )
    codes = {(code.system, code.code) for code in extract_codes(text)}
    assert ("ICD-10", "E11.9") in codes
    assert ("CPT", "99213") in codes
    assert ("SNOMED", "271649006") in codes
    assert ("RxNorm", "860975") in codes
    assert ("CARC", "CO-16") in codes
    assert ("CPT", "12345") not in codes
