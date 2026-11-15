"""Task functions shared by the Airflow DAG and the local refresh runner.

Each function performs real work and writes a small task artifact under
``artifacts/refresh/tasks``. The Airflow file only wires these functions
together; the same code path is executed locally by
``scripts/run_refresh.py``.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinicalnoterag.config import get_settings
from clinicalnoterag.data.synthea_gen import generate_corpus
from clinicalnoterag.deid.audit import scan_corpus
from clinicalnoterag.deid.engine import DeidEngine
from clinicalnoterag.embeddings.backends import get_embedding_backend, load_embedding_state
from clinicalnoterag.pipeline.ingest import (
    load_prepared,
    prepare_ingest,
    save_prepared,
)
from clinicalnoterag.seeds import set_global_seed
from clinicalnoterag.vectorstore.factory import get_store

DAG_ID = "clinical_embedding_refresh"
DEFAULT_CORPUS = Path("data/corpus")
DEFAULT_PATIENTS = 300


def _tasks_dir(settings) -> Path:
    path = settings.artifact_dir / "refresh" / "tasks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_task(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    payload = {**payload, "task": name, "recorded_at": datetime.now(UTC).isoformat()}
    (_tasks_dir(settings) / f"{name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _iter_notes(corpus: Path):
    for path in sorted((corpus / "notes").glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def generate_or_sync_corpus(
    corpus_dir: str | Path = DEFAULT_CORPUS,
    n_patients: int = DEFAULT_PATIENTS,
) -> dict[str, Any]:
    """Generate the corpus if absent, otherwise sync from the existing manifest."""
    settings = get_settings()
    set_global_seed(settings.seed)
    corpus = Path(corpus_dir)
    started = time.perf_counter()
    manifest_path = corpus / "corpus_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        action = "synced_existing"
    else:
        stats = generate_corpus(n_patients, corpus, seed=settings.seed)
        manifest = json.loads(Path(stats.manifest_path).read_text(encoding="utf-8"))
        action = "generated"
    payload = _record_task(
        "generate_or_sync_corpus",
        {
            "action": action,
            "corpus": str(corpus),
            "n_patients": manifest["n_patients"],
            "n_documents": manifest["n_documents"],
            "n_encounters": manifest["n_encounters"],
            "n_denied_claims": manifest["n_denied_claims"],
            "elapsed_sec": round(time.perf_counter() - started, 4),
        },
    )
    return payload


def deid_scan(corpus_dir: str | Path = DEFAULT_CORPUS) -> dict[str, Any]:
    """Scrub every note and publish a fresh PHI leak audit."""
    settings = get_settings()
    started = time.perf_counter()
    corpus = Path(corpus_dir)
    manifest = json.loads((corpus / "corpus_manifest.json").read_text(encoding="utf-8"))
    engine = DeidEngine(backend="auto", known_names=list(manifest.get("known_names", [])))
    records = list(_iter_notes(corpus))
    report = scan_corpus(records, engine=engine)
    out = settings.artifact_dir / "refresh" / "deid_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return _record_task(
        "deid_scan",
        {
            "documents": report.total_documents,
            "phi_leak_rate": report.phi_leak_rate,
            "backend_name": report.backend_name,
            "per_entity_recall": report.per_entity_recall,
            "elapsed_sec": round(time.perf_counter() - started, 4),
            "artifact": str(out),
        },
    )


def chunk_and_embed(
    corpus_dir: str | Path = DEFAULT_CORPUS,
    prepared_dir: str | Path = "artifacts/refresh/prepared",
) -> dict[str, Any]:
    """De-identify, chunk and embed; persist prepared chunks for the upsert task."""
    settings = get_settings()
    started = time.perf_counter()
    corpus = Path(corpus_dir)
    manifest = json.loads((corpus / "corpus_manifest.json").read_text(encoding="utf-8"))
    embedder = get_embedding_backend(settings)
    deid = DeidEngine(backend="auto", known_names=list(manifest.get("known_names", [])))
    prepared = prepare_ingest(corpus, deid, embedder, settings)
    save_prepared(prepared, prepared_dir)
    return _record_task(
        "chunk_and_embed",
        {
            "documents": prepared.n_notes + prepared.n_bundles,
            "chunks": len(prepared.records),
            "document_chunks": prepared.n_note_chunks,
            "fact_chunks": prepared.n_fact_chunks,
            "embedding_backend": prepared.embedding_backend,
            "embedding_dim": prepared.embedding_dim,
            "deid_backend": prepared.deid_backend,
            "prepared_dir": str(prepared_dir),
            "elapsed_sec": round(time.perf_counter() - started, 4),
        },
    )


def upsert(
    prepared_dir: str | Path = "artifacts/refresh/prepared",
    store_name: str | None = None,
) -> dict[str, Any]:
    """Idempotently upsert prepared chunks into the configured vector store."""
    settings = get_settings()
    started = time.perf_counter()
    prepared = load_prepared(prepared_dir)
    store = get_store(settings, prepared.embedding_dim, store_name=store_name, fresh=False)
    try:
        inserted = 0
        for start in range(0, len(prepared.records), 256):
            inserted += store.upsert(prepared.records[start : start + 256])
        count = store.count()
    finally:
        store.close()
    return _record_task(
        "upsert",
        {
            "inserted": inserted,
            "store_count": count,
            "store_backend": store.name,
            "elapsed_sec": round(time.perf_counter() - started, 4),
        },
    )


def smoke_query(
    query: str = "What diagnoses were documented for patient PAT-0001?",
    patient_id: str = "PAT-0001",
) -> dict[str, Any]:
    """Run a metadata-filtered retrieval smoke test against the live store."""
    settings = get_settings()
    started = time.perf_counter()
    embedder, _ = load_embedding_state(settings)
    store = get_store(settings, embedder.dim, fresh=False)
    try:
        vector = embedder.encode([query])[0]
        hits = store.search(vector, k=3, filters={"patient_id": patient_id})
        count = store.count()
    finally:
        store.close()
    return _record_task(
        "smoke_query",
        {
            "query": query,
            "patient_filter": patient_id,
            "top_chunk_ids": [hit.chunk_id for hit in hits],
            "top_scores": [round(hit.score, 6) for hit in hits],
            "store_count": count,
            "elapsed_sec": round(time.perf_counter() - started, 4),
        },
    )


def publish_metrics() -> dict[str, Any]:
    """Aggregate task artifacts into one publishable refresh metrics file."""
    settings = get_settings()
    tasks: dict[str, Any] = {}
    for path in sorted(_tasks_dir(settings).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks[payload["task"]] = payload
    ingest_path = settings.artifact_dir / "ingest_report.json"
    ingest = json.loads(ingest_path.read_text(encoding="utf-8")) if ingest_path.exists() else {}
    metrics = {
        "dag_id": DAG_ID,
        "published_at": datetime.now(UTC).isoformat(),
        "tasks_completed": sorted(tasks),
        "task_count": len(tasks),
        "tasks": tasks,
        "ingest": {
            key: ingest.get(key)
            for key in (
                "documents",
                "chunks",
                "store_backend",
                "embedding_backend",
                "deid_backend",
                "chunks_per_sec",
                "total_wall_sec",
            )
        },
        "status": "success" if tasks else "no_tasks",
    }
    out = settings.artifact_dir / "refresh_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_all(corpus_dir: str | Path = DEFAULT_CORPUS) -> dict[str, Any]:
    """Run every refresh task in DAG order and publish metrics (local runner)."""
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for name, fn in (
        ("generate_or_sync_corpus", generate_or_sync_corpus),
        ("deid_scan", deid_scan),
        ("chunk_and_embed", chunk_and_embed),
        ("upsert", upsert),
        ("smoke_query", smoke_query),
    ):
        results[name] = (
            fn(corpus_dir)
            if name in {"generate_or_sync_corpus", "deid_scan", "chunk_and_embed"}
            else fn()
        )
    metrics = publish_metrics()
    metrics["total_wall_sec"] = round(time.perf_counter() - started, 4)
    (get_settings().artifact_dir / "refresh_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


__all__ = [
    "DAG_ID",
    "chunk_and_embed",
    "deid_scan",
    "generate_or_sync_corpus",
    "publish_metrics",
    "run_all",
    "smoke_query",
    "upsert",
]
