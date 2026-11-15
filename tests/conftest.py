"""Shared pytest fixtures. All fixtures are offline and synthetic-data only."""

from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clinicalnoterag.config import get_settings
from clinicalnoterag.data.fhir_models import validate_bundle
from clinicalnoterag.deid.engine import DeidEngine
from clinicalnoterag.embeddings.backends import HashingTfidfBackend
from clinicalnoterag.pipeline.chunking import chunk_bundle, chunk_note
from clinicalnoterag.rag.answerer import ClinicalAnswerer
from clinicalnoterag.rag.retriever import HybridRetriever
from clinicalnoterag.vectorstore.base import InMemoryVectorStore, Record

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="session")
def sample_corpus() -> Path:
    """Committed 3-patient sample corpus (bundles, notes, manifest)."""
    assert (SAMPLE_DIR / "corpus_manifest.json").exists()
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def sample_manifest(sample_corpus: Path) -> dict:
    """Parsed sample manifest."""
    return json.loads((sample_corpus / "corpus_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_bundles(sample_corpus: Path) -> list[dict]:
    """All validated sample FHIR bundles."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((sample_corpus / "bundles").glob("*.json"))
    ]


@pytest.fixture(scope="session")
def sample_notes(sample_corpus: Path) -> list[dict]:
    """All sample note records."""
    notes: list[dict] = []
    for path in sorted((sample_corpus / "notes").glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            notes.extend(json.loads(line) for line in fh if line.strip())
    return notes


@pytest.fixture
def tiny_settings(tmp_path: Path):
    """Settings that force the deterministic offline embedding backend."""
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


@pytest.fixture(scope="session")
def tiny_system(tmp_path_factory: pytest.TempPathFactory):
    """Small in-memory RAG system built from the committed sample corpus."""
    tmp = tmp_path_factory.mktemp("tiny-system")
    settings = get_settings(
        embedding_backend="hashing",
        embedding_dim=64,
        artifact_dir=tmp / "artifacts",
        faiss_index_path=tmp / "artifacts" / "index" / "test.faiss",
        sqlite_path=tmp / "artifacts" / "index" / "test.sqlite3",
        telemetry_path=tmp / "artifacts" / "telemetry.jsonl",
        retrieval_k=5,
        seed=42,
    )
    manifest = json.loads((SAMPLE_DIR / "corpus_manifest.json").read_text(encoding="utf-8"))
    deid = DeidEngine(backend="regex", known_names=manifest.get("known_names", []))
    chunks = []
    for path in sorted((SAMPLE_DIR / "bundles").glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        validate_bundle(bundle)
        chunks.extend(chunk_bundle(bundle))
    for path in sorted((SAMPLE_DIR / "notes").glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                note = json.loads(line)
                result = deid.scrub(note["text"])
                chunks.extend(chunk_note(note, result.text))
    embedder = HashingTfidfBackend(dim=64)
    texts = [chunk.text for chunk in chunks]
    embedder.fit(texts)
    vectors = embedder.encode(texts)
    store = InMemoryVectorStore(dim=64)
    store.upsert(
        [
            Record(
                id=chunk.storage_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                vector=vectors[index],
            )
            for index, chunk in enumerate(chunks)
        ]
    )
    retriever = HybridRetriever(store, embedder, settings)
    answerer = ClinicalAnswerer(retriever, settings=settings, system_name="test")
    return SimpleNamespace(
        settings=settings,
        embedder=embedder,
        store=store,
        retriever=retriever,
        answerer=answerer,
        manifest=manifest,
        notes=[
            note
            for path in sorted((SAMPLE_DIR / "notes").glob("*.jsonl"))
            for note in (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        ],
    )


def make_records(dim: int = 8) -> list[Record]:
    """Two-patient records used by the store contract tests."""

    def vec(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)[:dim]
        norm = float(np.linalg.norm(array)) or 1.0
        return array / norm

    return [
        Record(
            id="r1",
            chunk_id="c1",
            text="Essential hypertension ICD-10 I10 diagnosis",
            metadata={
                "patient_id": "PAT-0001",
                "encounter_id": "ENC-1",
                "resource_type": "Condition",
                "code_system": "ICD-10",
                "code": "I10",
                "doc_type": "fhir_fact",
                "service_date": "2024-01-01",
            },
            vector=vec([1, 0, 0, 0, 0, 0, 0, 0]),
        ),
        Record(
            id="r2",
            chunk_id="c2",
            text="Claim denied CARC CO-16 procedure CPT 99213",
            metadata={
                "patient_id": "PAT-0002",
                "encounter_id": "ENC-2",
                "resource_type": "Claim",
                "code_system": "CPT",
                "code": "99213",
                "doc_type": "fhir_fact",
                "service_date": "2024-02-01",
                "claim_id": "CLM-2",
                "denied": True,
                "denial_code": "CO-16",
            },
            vector=vec([0, 1, 0, 0, 0, 0, 0, 0]),
        ),
        Record(
            id="r3",
            chunk_id="c3",
            text="Type 2 diabetes E11.9 clinical note",
            metadata={
                "patient_id": "PAT-0001",
                "encounter_id": "ENC-1",
                "resource_type": "Note",
                "doc_type": "clinical_note",
                "service_date": "2024-03-01",
            },
            vector=vec([0, 0, 1, 0, 0, 0, 0, 0]),
        ),
    ]
