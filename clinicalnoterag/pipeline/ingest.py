"""End-to-end ingestion: load -> de-identify -> chunk -> embed -> upsert."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from clinicalnoterag.config import Settings
from clinicalnoterag.data.synthea_gen import load_bundle
from clinicalnoterag.deid.engine import DeidEngine
from clinicalnoterag.embeddings.backends import (
    EmbeddingBackend,
    save_embedding_state,
    state_path_for,
)
from clinicalnoterag.pipeline.chunking import Chunk, chunk_bundle, chunk_note
from clinicalnoterag.vectorstore.base import Record, VectorStore


@dataclass
class PreparedIngest:
    """All chunks with vectors, ready for idempotent upsert."""

    records: list[Record]
    n_bundles: int
    n_notes: int
    n_clinical_notes: int
    n_payment_notes: int
    n_fact_chunks: int
    n_note_chunks: int
    deid_backend: str
    deid_entities_replaced: int
    embedding_backend: str
    embedding_dim: int
    timings_sec: dict[str, float] = field(default_factory=dict)

    def vectors(self) -> np.ndarray:
        """Stack record vectors into one matrix."""
        return np.vstack([r.vector for r in self.records]).astype(np.float32)

    def to_meta(self) -> dict[str, Any]:
        """Return metadata-only summary (no chunk text)."""
        data = asdict(self)
        data.pop("records")
        return data


def _iter_note_records(corpus_dir: Path):
    for path in sorted((corpus_dir / "notes").glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def save_prepared(prepared: PreparedIngest, directory: str | Path) -> None:
    """Persist prepared chunks + vectors so the DAG can split embedding and upsert."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", prepared.vectors())
    with (path / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for record in prepared.records:
            fh.write(
                json.dumps(
                    {
                        "id": record.id,
                        "chunk_id": record.chunk_id,
                        "text": record.text,
                        "metadata": record.metadata,
                    }
                )
                + "\n"
            )
    (path / "prepared_meta.json").write_text(
        json.dumps(prepared.to_meta(), indent=2), encoding="utf-8"
    )


def load_prepared(directory: str | Path) -> PreparedIngest:
    """Load a prepared ingest written by :func:`save_prepared`."""
    path = Path(directory)
    meta = json.loads((path / "prepared_meta.json").read_text(encoding="utf-8"))
    vectors = np.load(path / "vectors.npy")
    records: list[Record] = []
    with (path / "chunks.jsonl").open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            row = json.loads(line)
            records.append(
                Record(
                    id=row["id"],
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    metadata=row["metadata"],
                    vector=vectors[idx],
                )
            )
    return PreparedIngest(records=records, **meta)


def prepare_ingest(
    corpus_dir: str | Path,
    deid: DeidEngine,
    embedder: EmbeddingBackend,
    settings: Settings,
    deid_map_dir: str | Path | None = None,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
    batch_size: int = 256,
) -> PreparedIngest:
    """Load, de-identify, chunk and embed a corpus without touching a store.

    De-identification maps are written outside the vector store at
    ``deid_map_dir`` (default ``artifacts/deid_maps``) for authorized reversal.
    """
    corpus = Path(corpus_dir)
    manifest_path = corpus / "corpus_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing corpus manifest: {manifest_path}")
    maps_dir = Path(deid_map_dir) if deid_map_dir else settings.artifact_dir / "deid_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    target = target_tokens or settings.chunk_target_tokens
    overlap = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens

    timings = {"load_sec": 0.0, "deid_sec": 0.0, "chunk_sec": 0.0, "embed_sec": 0.0}
    chunks: list[Chunk] = []
    n_bundles = n_notes = n_clinical = n_payment = 0
    deid_backends: set[str] = set()
    deid_entities = 0

    start = time.perf_counter()
    for bundle_path in sorted((corpus / "bundles").glob("*.json")):
        bundle = load_bundle(bundle_path)
        chunks.extend(chunk_bundle(bundle))
        n_bundles += 1
    timings["load_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    for note in _iter_note_records(corpus):
        n_notes += 1
        if note.get("doc_type") == "clinical_note":
            n_clinical += 1
        else:
            n_payment += 1
        t_deid = time.perf_counter()
        result = deid.scrub(note["text"])
        deid_backends.add(result.backend_name)
        deid_entities += len(result.entities)
        timings["deid_sec"] += time.perf_counter() - t_deid
        map_path = maps_dir / f"{note['doc_id']}.json"
        map_path.write_text(
            json.dumps(
                {
                    "doc_id": note["doc_id"],
                    "patient_id": note.get("patient_id"),
                    "backend": result.backend_name,
                    "mapping": result.mapping,
                    "entities": [entity.to_dict() for entity in result.entities],
                    "warning": "PHI-bearing reversal map. Never index or commit this file.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        t_chunk = time.perf_counter()
        chunks.extend(chunk_note(note, result.text, target_tokens=target, overlap_tokens=overlap))
        timings["chunk_sec"] += time.perf_counter() - t_chunk
    timings["load_sec"] += time.perf_counter() - start - timings["deid_sec"] - timings["chunk_sec"]

    n_fact = sum(1 for chunk in chunks if chunk.metadata.get("doc_type") == "fhir_fact")
    n_note_chunks = len(chunks) - n_fact

    start = time.perf_counter()
    texts = [chunk.text for chunk in chunks]
    embedder.fit(texts)
    vectors = embedder.encode(texts)
    timings["embed_sec"] = time.perf_counter() - start
    if vectors.shape[0] != len(chunks):
        raise RuntimeError("embedding backend returned a different number of rows")

    records = [
        Record(
            id=chunk.storage_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=chunk.metadata,
            vector=vectors[i],
        )
        for i, chunk in enumerate(chunks)
    ]
    save_embedding_state(embedder, state_path_for(settings))
    return PreparedIngest(
        records=records,
        n_bundles=n_bundles,
        n_notes=n_notes,
        n_clinical_notes=n_clinical,
        n_payment_notes=n_payment,
        n_fact_chunks=n_fact,
        n_note_chunks=n_note_chunks,
        deid_backend=",".join(sorted(deid_backends)),
        deid_entities_replaced=deid_entities,
        embedding_backend=embedder.name,
        embedding_dim=embedder.dim,
        timings_sec={k: round(v, 4) for k, v in timings.items()},
    )


def run_ingest(
    corpus_dir: str | Path,
    store: VectorStore,
    deid: DeidEngine,
    embedder: EmbeddingBackend,
    settings: Settings,
    out_path: str | Path | None = None,
    deid_map_dir: str | Path | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run the full ingest and write a real ingest report.

    Args:
        corpus_dir: Directory containing ``bundles/``, ``notes/`` and
            ``corpus_manifest.json``.
        store: A FAISS or pgvector store.
        deid: De-identification engine.
        embedder: Embedding backend.
        settings: Application settings.
        out_path: Report path; defaults to ``artifacts/ingest_report.json``.
        deid_map_dir: Directory for reversible PHI maps (outside the store).
        batch_size: Batch size used for the upsert loop.

    Returns:
        The report dict written to disk.
    """
    started = time.perf_counter()
    prepared = prepare_ingest(
        corpus_dir,
        deid,
        embedder,
        settings,
        deid_map_dir=deid_map_dir,
        batch_size=batch_size,
    )
    upsert_start = time.perf_counter()
    inserted = 0
    for start in range(0, len(prepared.records), batch_size):
        inserted += store.upsert(prepared.records[start : start + batch_size])
    total = time.perf_counter() - started
    timings = {**prepared.timings_sec, "upsert_sec": round(time.perf_counter() - upsert_start, 4)}
    report = {
        "corpus": str(corpus_dir),
        "seed": settings.seed,
        "documents": prepared.n_notes + prepared.n_bundles,
        "bundles": prepared.n_bundles,
        "notes": prepared.n_notes,
        "clinical_notes": prepared.n_clinical_notes,
        "payment_notes": prepared.n_payment_notes,
        "chunks": len(prepared.records),
        "fact_chunks": prepared.n_fact_chunks,
        "note_chunks": prepared.n_note_chunks,
        "inserted": inserted,
        "store_count": store.count(),
        "store_backend": store.name,
        "embedding_backend": prepared.embedding_backend,
        "embedding_dim": prepared.embedding_dim,
        "deid_backend": prepared.deid_backend,
        "deid_entities_replaced": prepared.deid_entities_replaced,
        "deid_maps_dir": str(deid_map_dir or settings.artifact_dir / "deid_maps"),
        "timings_sec": timings,
        "total_wall_sec": round(total, 4),
        "chunks_per_sec": round(len(prepared.records) / total, 2) if total else 0.0,
        "embedding_chunks_per_sec": (
            round(len(prepared.records) / timings["embed_sec"], 2) if timings["embed_sec"] else 0.0
        ),
        "batch_size": batch_size,
        "filters_supported": [
            "patient_id",
            "encounter_id",
            "resource_type",
            "code_system",
            "code",
            "service_date_gte",
            "service_date_lte",
            "doc_type",
        ],
        "note": "Generated by clinicalnoterag.pipeline.ingest.run_ingest",
    }
    target = Path(out_path) if out_path else settings.artifact_dir / "ingest_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = [
    "PreparedIngest",
    "load_prepared",
    "prepare_ingest",
    "run_ingest",
    "save_prepared",
]
