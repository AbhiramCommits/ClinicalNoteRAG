"""FastAPI endpoints and Airflow DAG import tests."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from clinicalnoterag.api.app import create_app


@pytest.fixture(scope="module")
def client(tiny_system):
    app = create_app(settings=tiny_system.settings, answerer=tiny_system.answerer)
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["store_count"] > 0
    assert body["store_backend"] == "in-memory"
    assert "hashing" in body["embedding_backend"]


def test_ask_returns_cited_answer(client):
    response = client.post(
        "/ask",
        json={
            "question": "What diagnoses were documented for encounter ENC-0001-01?",
            "filters": {
                "patient_id": "PAT-0001",
                "encounter_id": "ENC-0001-01",
                "resource_type": "Condition",
                "doc_type": "fhir_fact",
            },
            "k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["citations"]
    assert body["text"]
    assert body["backend_name"] == "extractive-offline"


def test_ask_abstains_on_unanswerable_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "What diagnosis code Z99.999 was documented for patient PAT-0001?",
            "filters": {"patient_id": "PAT-0001", "resource_type": "Condition"},
            "k": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["text"] == "Insufficient evidence in the available record"
    assert body["citations"] == []


def test_claims_denial_endpoint(client):
    response = client.post(
        "/ask/claims-denial",
        json={"claim_id": "CLM-0002-01", "patient_id": "PAT-0002", "k": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == "CLM-0002-01"
    assert body["answer"]["abstained"] is False
    assert body["answer"]["citations"]


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "p50_latency_ms" in body
    assert "abstention_rate" in body
    assert body["n_queries"] >= 0


def test_unknown_filter_returns_422(client):
    response = client.post(
        "/ask",
        json={"question": "Any question", "filters": {"doctor_name": "x"}, "k": 5},
    )
    assert response.status_code == 422


def test_dag_imports_cleanly_without_airflow():
    module = importlib.import_module("dags.clinical_embedding_refresh_dag")
    assert module.dag.dag_id == "clinical_embedding_refresh"
    for task_name in (
        "generate_or_sync_corpus",
        "deid_scan",
        "chunk_and_embed",
        "upsert",
        "smoke_query",
        "publish_metrics",
    ):
        assert hasattr(module, task_name)
    from dags._compat import HAS_AIRFLOW

    assert isinstance(HAS_AIRFLOW, bool)
