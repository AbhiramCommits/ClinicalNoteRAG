"""Ingestion idempotency and PHI-boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

from clinicalnoterag.config import get_settings
from clinicalnoterag.data.synthea_gen import validate_bundle
from clinicalnoterag.deid.engine import DeidEngine
from clinicalnoterag.embeddings.backends import HashingTfidfBackend
from clinicalnoterag.pipeline.ingest import run_ingest
from clinicalnoterag.vectorstore.base import InMemoryVectorStore


def _engine(manifest: dict) -> DeidEngine:
    return DeidEngine(backend="regex", known_names=manifest["known_names"])


def _settings(tmp_path: Path):
    return get_settings(
        embedding_backend="hashing",
        embedding_dim=64,
        artifact_dir=tmp_path / "artifacts",
        faiss_index_path=tmp_path / "artifacts" / "index" / "test.faiss",
        sqlite_path=tmp_path / "artifacts" / "index" / "test.sqlite3",
        telemetry_path=tmp_path / "artifacts" / "telemetry.jsonl",
        retrieval_k=5,
        seed=42,
    )


def test_run_ingest_writes_real_report(sample_corpus, sample_manifest, tmp_path):
    settings = _settings(tmp_path)
    store = InMemoryVectorStore(dim=64)
    embedder = HashingTfidfBackend(dim=64)
    report = run_ingest(
        sample_corpus,
        store,
        _engine(sample_manifest),
        embedder,
        settings,
        out_path=tmp_path / "ingest_report.json",
        deid_map_dir=tmp_path / "deid_maps",
    )
    assert report["documents"] == 23
    assert report["bundles"] == 3
    assert report["notes"] == 20
    assert report["chunks"] > 0
    assert report["store_count"] == report["chunks"]
    assert report["embedding_backend"].startswith("hashing-tfidf")
    assert report["deid_backend"] == "regex"
    assert report["chunks_per_sec"] > 0
    assert (tmp_path / "ingest_report.json").exists()
    assert (tmp_path / "deid_maps").exists()


def test_run_ingest_is_idempotent(sample_corpus, sample_manifest, tmp_path):
    settings = _settings(tmp_path)
    store = InMemoryVectorStore(dim=64)
    embedder = HashingTfidfBackend(dim=64)
    deid = _engine(sample_manifest)
    first = run_ingest(
        sample_corpus,
        store,
        deid,
        embedder,
        settings,
        out_path=tmp_path / "first.json",
        deid_map_dir=tmp_path / "deid_maps",
    )
    second = run_ingest(
        sample_corpus,
        store,
        deid,
        embedder,
        settings,
        out_path=tmp_path / "second.json",
        deid_map_dir=tmp_path / "deid_maps",
    )
    assert first["inserted"] == first["chunks"]
    assert second["inserted"] == 0
    assert store.count() == first["chunks"]


def test_store_contains_no_injected_phi(sample_corpus, sample_manifest, tmp_path):
    settings = _settings(tmp_path)
    store = InMemoryVectorStore(dim=64)
    embedder = HashingTfidfBackend(dim=64)
    run_ingest(
        sample_corpus,
        store,
        _engine(sample_manifest),
        embedder,
        settings,
        out_path=tmp_path / "report.json",
        deid_map_dir=tmp_path / "deid_maps",
    )
    bundle = validate_bundle(
        json.loads((sample_corpus / "bundles" / "patient_0001.json").read_text(encoding="utf-8"))
    )
    patient = next(
        entry.resource for entry in bundle.entry if entry.resource.resourceType == "Patient"
    )
    name = patient.name[0]
    full_name = f"{name.given[0]} {name.family}"
    for record in store.all_records():
        assert full_name not in record.text
        assert "MRN-" not in record.text
        serialized = json.dumps(record.metadata)
        assert full_name not in serialized
        assert "MRN-" not in serialized
    maps = list((tmp_path / "deid_maps").glob("*.json"))
    assert maps, "reversible maps must be stored outside the vector store"
    map_payload = json.loads(maps[0].read_text(encoding="utf-8"))
    assert "warning" in map_payload
