"""Postgres + pgvector store implementing the same interface as FAISS."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import numpy as np

from clinicalnoterag.vectorstore.base import (
    Record,
    SearchHit,
    normalize_filters,
    sql_filter_clause,
    validate_vector,
)


def _literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"


class PgVectorStore:
    """pgvector backend with SQL metadata filtering.

    Vectors are L2-normalized by the embedding backends and stored in a
    ``vector(dim)`` column. Search uses cosine distance and converts it to a
    similarity score (``1 - distance``).
    """

    name = "pgvector"

    def __init__(
        self,
        dsn: str,
        dim: int,
        table: str = "chunks",
        fresh: bool = False,
    ) -> None:
        import psycopg

        self.dsn = dsn
        self.dim = int(dim)
        self.table = table
        self._conn = psycopg.connect(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            if fresh:
                cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    patient_id TEXT,
                    encounter_id TEXT,
                    resource_type TEXT,
                    code_system TEXT,
                    code TEXT,
                    doc_type TEXT,
                    service_date DATE,
                    claim_id TEXT,
                    denied BOOLEAN,
                    denial_code TEXT,
                    title TEXT,
                    section TEXT,
                    chunk_index INTEGER,
                    metadata JSONB NOT NULL,
                    embedding vector({self.dim}) NOT NULL
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_patient_idx ON {self.table} (patient_id)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_encounter_idx ON {self.table} (encounter_id)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_doc_type_idx ON {self.table} (doc_type)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_code_idx ON {self.table} (code_system, code)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_service_date_idx ON {self.table} (service_date)"
            )
            with contextlib.suppress(Exception):
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
                    f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
                )

    def upsert(self, records: list[Record]) -> int:
        """Insert or update by content-hash id; returns rows actually inserted."""
        inserted = 0
        with self._conn.cursor() as cur:
            for record in records:
                vector = _literal(validate_vector(record.vector, self.dim))
                meta = record.metadata
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (
                        id, chunk_id, text, patient_id, encounter_id, resource_type,
                        code_system, code, doc_type, service_date, claim_id, denied,
                        denial_code, title, section, chunk_index, metadata, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s::jsonb, %s::vector
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    RETURNING (xmax = 0) AS inserted
                    """,
                    (
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
                        bool(meta.get("denied")),
                        meta.get("denial_code"),
                        meta.get("title"),
                        meta.get("section"),
                        meta.get("chunk_index"),
                        json.dumps(meta, default=str),
                        vector,
                    ),
                )
                row = cur.fetchone()
                if row and row[0]:
                    inserted += 1
        return inserted

    def search(
        self, query_vec: np.ndarray, k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Cosine-similarity search with SQL metadata filters."""
        if k <= 0:
            return []
        query = validate_vector(query_vec, self.dim)
        active = normalize_filters(filters)
        clause, params = sql_filter_clause(active)
        sql = (
            f"SELECT id, chunk_id, text, metadata, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table}{clause} ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        vector = _literal(query)
        with self._conn.cursor() as cur:
            cur.execute(sql, [vector, *params, vector, k])
            rows = cur.fetchall()
        return [
            SearchHit(
                record_id=row[0],
                chunk_id=row[1],
                text=row[2],
                metadata=row[3],
                score=float(row[4]),
            )
            for row in rows
        ]

    def count(self) -> int:
        """Number of rows in the table."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def all_records(self, include_vectors: bool = False) -> list[Record]:
        """Return every record; vectors are reconstructed when requested."""
        columns = "id, chunk_id, text, metadata" + (", embedding::text" if include_vectors else "")
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT {columns} FROM {self.table}")
            rows = cur.fetchall()
        records: list[Record] = []
        for row in rows:
            vector = None
            if include_vectors:
                vector = np.asarray(json.loads(row[4]), dtype=np.float32)
            records.append(
                Record(
                    id=row[0],
                    chunk_id=row[1],
                    text=row[2],
                    metadata=row[3],
                    vector=vector,
                )
            )
        return records

    def close(self) -> None:
        """Close the Postgres connection."""
        self._conn.close()

    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["PgVectorStore"]
