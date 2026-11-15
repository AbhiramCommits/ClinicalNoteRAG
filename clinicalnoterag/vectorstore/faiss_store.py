"""FAISS vector store with a SQLite metadata sidecar."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

with contextlib.suppress(Exception):  # avoid OpenMP conflicts with PyTorch
    faiss.omp_set_num_threads(1)

from clinicalnoterag.vectorstore.base import (
    Record,
    SearchHit,
    normalize_filters,
    validate_vector,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    chunk_id TEXT NOT NULL,
    text TEXT NOT NULL,
    patient_id TEXT,
    encounter_id TEXT,
    resource_type TEXT,
    code_system TEXT,
    code TEXT,
    doc_type TEXT,
    service_date TEXT,
    claim_id TEXT,
    denied INTEGER,
    denial_code TEXT,
    title TEXT,
    section TEXT,
    chunk_index INTEGER,
    metadata TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_patient ON chunks(patient_id);
CREATE INDEX IF NOT EXISTS idx_chunks_encounter ON chunks(encounter_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_code ON chunks(code_system, code);
CREATE INDEX IF NOT EXISTS idx_chunks_service_date ON chunks(service_date);
"""


class FaissStore:
    """FAISS ``IndexFlatIP`` + SQLite sidecar for metadata and filtering.

    Vectors are normalized by the embedding backends, so inner product equals
    cosine similarity. Filtered searches scan only the SQL-selected rowids and
    score them exactly, which guarantees that a patient filter cannot return a
    chunk from another patient.
    """

    name = "faiss"

    def __init__(
        self,
        dim: int,
        index_path: str | Path,
        sqlite_path: str | Path,
        fresh: bool = False,
    ) -> None:
        self.dim = int(dim)
        self.index_path = Path(index_path)
        self.sqlite_path = Path(sqlite_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            self.index_path.unlink(missing_ok=True)
            self.sqlite_path.unlink(missing_ok=True)
        self._index = self._load_index()
        self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _load_index(self) -> faiss.Index:
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.d != self.dim:
                raise ValueError(
                    f"index dim {index.d} does not match requested dim {self.dim}; "
                    "use fresh=True or rebuild"
                )
            return index
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _row_from_record(self, record: Record) -> tuple[Any, ...]:
        meta = record.metadata
        vector = validate_vector(record.vector, self.dim)
        return (
            record.id,
            record.chunk_id,
            record.text,
            meta.get("patient_id"),
            meta.get("encounter_id"),
            meta.get("resource_type"),
            meta.get("code_system"),
            meta.get("code"),
            meta.get("doc_type"),
            meta.get("service_date"),
            meta.get("claim_id"),
            1 if meta.get("denied") else 0,
            meta.get("denial_code"),
            meta.get("title"),
            meta.get("section"),
            meta.get("chunk_index"),
            json.dumps(meta, sort_keys=True, default=str),
            vector.astype(np.float32).tobytes(),
        )

    def upsert(self, records: list[Record]) -> int:
        """Insert new content-hash ids; duplicates are skipped idempotently."""
        inserted = 0
        with self._lock:
            for record in records:
                cursor = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO chunks (
                        id, chunk_id, text, patient_id, encounter_id, resource_type,
                        code_system, code, doc_type, service_date, claim_id, denied,
                        denial_code, title, section, chunk_index, metadata, vector
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._row_from_record(record),
                )
                if cursor.rowcount == 0:
                    continue
                rowid = int(cursor.lastrowid)
                vector = validate_vector(record.vector, self.dim).reshape(1, -1)
                self._index.add_with_ids(vector, np.asarray([rowid], dtype=np.int64))
                inserted += 1
            self._conn.commit()
            self.flush()
        return inserted

    def flush(self) -> None:
        """Persist the FAISS index to disk."""
        with self._lock:
            faiss.write_index(self._index, str(self.index_path))

    def _hit_from_row(self, row: sqlite3.Row, score: float) -> SearchHit:
        return SearchHit(
            record_id=row["id"],
            chunk_id=row["chunk_id"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
            score=float(score),
        )

    def search(
        self, query_vec: np.ndarray, k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Search the index, applying exact SQL metadata filters when present."""
        if k <= 0:
            return []
        query = validate_vector(query_vec, self.dim)
        active = normalize_filters(filters)
        if active:
            from clinicalnoterag.vectorstore.base import sql_filter_clause

            clause, params = sql_filter_clause(active)
            clause = clause.replace("%s", "?")
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT rowid, id, chunk_id, text, metadata, vector FROM chunks{clause}",  # noqa: S608
                    params,
                ).fetchall()
            if not rows:
                return []
            matrix = np.vstack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
            scores = matrix @ query
            order = np.argsort(-scores)[:k]
            return [self._hit_from_row(rows[int(i)], float(scores[int(i)])) for i in order]

        distances, ids = self._index.search(query.reshape(1, -1), k)
        ordered_ids = [int(i) for i in ids[0] if i != -1]
        if not ordered_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT rowid, id, chunk_id, text, metadata FROM chunks WHERE rowid IN ({placeholders})",  # noqa: S608
                ordered_ids,
            ).fetchall()
        by_rowid = {int(row["rowid"]): row for row in rows}
        hits: list[SearchHit] = []
        for score, rowid in zip(distances[0], ids[0], strict=False):
            if int(rowid) == -1:
                continue
            row = by_rowid.get(int(rowid))
            if row is not None:
                hits.append(self._hit_from_row(row, float(score)))
        return hits

    def count(self) -> int:
        """Number of rows in the sidecar."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def all_records(self, include_vectors: bool = False) -> list[Record]:
        """Return every chunk; vectors are optional to keep BM25 memory low."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, chunk_id, text, metadata, vector FROM chunks ORDER BY rowid"
            ).fetchall()
        records: list[Record] = []
        for row in rows:
            vector = None
            if include_vectors:
                vector = np.frombuffer(row["vector"], dtype=np.float32).copy()
            records.append(
                Record(
                    id=row["id"],
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]),
                    vector=vector,
                )
            )
        return records

    def close(self) -> None:
        """Flush the index and close SQLite."""
        try:
            self.flush()
        finally:
            with self._lock:
                self._conn.close()

    def __enter__(self) -> FaissStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["FaissStore"]
