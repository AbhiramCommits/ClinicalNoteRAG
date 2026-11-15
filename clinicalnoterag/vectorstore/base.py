"""Vector store interface shared by FAISS and pgvector backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: Metadata keys accepted by every store. Unknown keys are rejected so a typo
#: cannot silently return unfiltered (and potentially cross-patient) results.
FILTER_KEYS = {
    "patient_id",
    "encounter_id",
    "resource_type",
    "code_system",
    "code",
    "doc_type",
    "service_date_gte",
    "service_date_lte",
    "claim_id",
    "denied",
}

_SQL_COLUMNS = {
    "patient_id": "patient_id",
    "encounter_id": "encounter_id",
    "resource_type": "resource_type",
    "code_system": "code_system",
    "code": "code",
    "doc_type": "doc_type",
    "claim_id": "claim_id",
}


@dataclass
class Record:
    """A chunk plus its embedding and metadata."""

    id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: np.ndarray | None = None

    @property
    def patient_id(self) -> str | None:
        """Convenience accessor for the patient id."""
        return self.metadata.get("patient_id")

    @property
    def encounter_id(self) -> str | None:
        """Convenience accessor for the encounter id."""
        return self.metadata.get("encounter_id")

    @property
    def doc_type(self) -> str | None:
        """Convenience accessor for the document type."""
        return self.metadata.get("doc_type")


@dataclass
class SearchHit:
    """One scored retrieval result."""

    record_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    vector: np.ndarray | None = None

    @property
    def patient_id(self) -> str | None:
        """Patient id copied from metadata."""
        return self.metadata.get("patient_id")

    @property
    def encounter_id(self) -> str | None:
        """Encounter id copied from metadata."""
        return self.metadata.get("encounter_id")

    @property
    def doc_type(self) -> str | None:
        """Document type copied from metadata."""
        return self.metadata.get("doc_type")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable hit (without the vector)."""
        return {
            "record_id": self.record_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


@runtime_checkable
class VectorStore(Protocol):
    """Contract implemented by FAISS and pgvector stores."""

    name: str
    dim: int

    def upsert(self, records: list[Record]) -> int:
        """Insert records idempotently; return the number newly inserted."""
        ...

    def search(
        self, query_vec: np.ndarray, k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return the top-k highest-scoring records satisfying ``filters``."""
        ...

    def count(self) -> int:
        """Number of stored records."""
        ...

    def all_records(self, include_vectors: bool = False) -> list[Record]:
        """Return every stored record (used to build BM25 and benchmarks)."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


def normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Validate filter keys and drop ``None`` values.

    Raises:
        ValueError: If a filter key is not part of the documented schema.
    """
    if not filters:
        return {}
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise ValueError(f"unsupported metadata filters: {sorted(unknown)}")
    return {key: value for key, value in filters.items() if value is not None and value != ""}


def sql_filter_clause(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build a parameterized SQL WHERE fragment for the canonical filter schema."""
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in normalize_filters(filters).items():
        if key == "service_date_gte":
            clauses.append("service_date >= %s")
            params.append(str(value))
        elif key == "service_date_lte":
            clauses.append("service_date <= %s")
            params.append(str(value))
        elif key == "denied":
            clauses.append("denied = %s")
            params.append(1 if value else 0)
        else:
            column = _SQL_COLUMNS[key]
            clauses.append(f"{column} = %s")
            params.append(str(value))
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def validate_vector(vector: np.ndarray | None, dim: int) -> np.ndarray:
    """Validate and normalize a query/record vector to float32."""
    if vector is None:
        raise ValueError("vector is required")
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] != dim:
        raise ValueError(f"vector dimension {array.shape[0]} does not match store dim {dim}")
    return array


class InMemoryVectorStore:
    """Small exact store used by tests and examples.

    It implements the same contract as the production stores, including
    metadata filters, so it is useful for contract tests that must run without
    FAISS or Postgres.
    """

    name = "in-memory"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._records: dict[str, Record] = {}

    def upsert(self, records: list[Record]) -> int:
        """Insert new ids; existing ids are left unchanged."""
        inserted = 0
        for record in records:
            if record.id in self._records:
                continue
            vector = validate_vector(record.vector, self.dim)
            record.vector = vector
            self._records[record.id] = record
            inserted += 1
        return inserted

    def search(
        self, query_vec: np.ndarray, k: int, filters: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return top-k exact inner-product hits after metadata filtering."""
        query = validate_vector(query_vec, self.dim)
        active = normalize_filters(filters)
        hits: list[SearchHit] = []
        for record in self._records.values():
            if not _matches(record.metadata, active):
                continue
            score = float(np.dot(query, record.vector))
            hits.append(
                SearchHit(
                    record.id, record.chunk_id, record.text, record.metadata, score, record.vector
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        """Number of stored records."""
        return len(self._records)

    def all_records(self, include_vectors: bool = False) -> list[Record]:
        """Return all records, optionally with vectors."""
        return [
            Record(
                r.id, r.chunk_id, r.text, dict(r.metadata), r.vector if include_vectors else None
            )
            for r in self._records.values()
        ]

    def close(self) -> None:
        """No-op for the in-memory store."""
        return


def _matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if key == "service_date_gte":
            if str(metadata.get("service_date") or "") < str(value):
                return False
        elif key == "service_date_lte":
            if str(metadata.get("service_date") or "") > str(value):
                return False
        elif key == "denied":
            if bool(metadata.get("denied")) != bool(value):
                return False
        elif str(metadata.get(key)) != str(value):
            return False
    return True


__all__ = [
    "FILTER_KEYS",
    "InMemoryVectorStore",
    "Record",
    "SearchHit",
    "VectorStore",
    "normalize_filters",
    "sql_filter_clause",
    "validate_vector",
]
