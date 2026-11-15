"""Hybrid dense + BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from clinicalnoterag.config import Settings
from clinicalnoterag.embeddings.backends import EmbeddingBackend
from clinicalnoterag.vectorstore.base import Record, VectorStore, normalize_filters

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that keeps codes such as ``e11.9`` and ``co-16`` intact."""
    return _TOKEN_RE.findall(text.lower())


def metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Apply the canonical metadata filter schema in Python (for BM25 masking)."""
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


@dataclass
class RetrievedChunk:
    """One retrieval result with dense, BM25 and fused scores."""

    record_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rank: int = 0

    @property
    def patient_id(self) -> str | None:
        """Patient id from metadata."""
        return self.metadata.get("patient_id")

    @property
    def encounter_id(self) -> str | None:
        """Encounter id from metadata."""
        return self.metadata.get("encounter_id")

    @property
    def doc_type(self) -> str | None:
        """Document type from metadata."""
        return self.metadata.get("doc_type")

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        """Serialize retrieval scores and metadata for artifacts."""
        data = {
            "record_id": self.record_id,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "score": self.score,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rank": self.rank,
        }
        if include_text:
            data["text"] = self.text
        return data


class HybridRetriever:
    """Dense FAISS/pgvector search plus BM25, fused with reciprocal rank fusion."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingBackend,
        settings: Settings,
        records: list[Record] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.records = records if records is not None else store.all_records(include_vectors=False)
        self._by_id = {record.id: record for record in self.records}
        self._bm25 = (
            BM25Okapi([tokenize(record.text) for record in self.records]) if self.records else None
        )

    def _dense(self, query: str, k: int, filters: dict[str, Any]) -> list[tuple[str, float]]:
        vector = self.embedder.encode([query])[0]
        hits = self.store.search(vector, k=k, filters=filters)
        return [(hit.record_id, hit.score) for hit in hits]

    def _sparse(self, query: str, k: int, filters: dict[str, Any]) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        candidates: list[tuple[str, float]] = []
        for idx, record in enumerate(self.records):
            if not metadata_matches(record.metadata, filters):
                continue
            score = float(scores[idx])
            if score > 0:
                candidates.append((record.id, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

    def retrieve(
        self,
        question: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        """Retrieve chunks.

        Args:
            question: Natural-language query.
            k: Number of chunks to return (default from settings).
            filters: Canonical metadata filters (patient, encounter, code, ...).
            mode: ``hybrid``, ``dense`` or ``bm25``.

        Returns:
            Chunks sorted by fused score, with per-channel scores retained.
        """
        if mode not in {"hybrid", "dense", "bm25"}:
            raise ValueError(f"unknown retrieval mode: {mode}")
        top_k = k or self.settings.retrieval_k
        active = normalize_filters(filters)
        candidate_k = max(top_k * 4, 20)
        dense = self._dense(question, candidate_k, active) if mode in {"hybrid", "dense"} else []
        sparse = self._sparse(question, candidate_k, active) if mode in {"hybrid", "bm25"} else []
        dense_scores = dict(dense)
        sparse_scores = dict(sparse)
        fused: dict[str, float] = {}
        for ranked in (dense, sparse):
            for rank, (record_id, _) in enumerate(ranked, start=1):
                fused[record_id] = fused.get(record_id, 0.0) + 1.0 / (self.settings.rrf_k + rank)
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        results: list[RetrievedChunk] = []
        for rank, (record_id, score) in enumerate(ordered, start=1):
            record = self._by_id.get(record_id)
            if record is None:
                continue
            results.append(
                RetrievedChunk(
                    record_id=record.id,
                    chunk_id=record.chunk_id,
                    text=record.text,
                    metadata=record.metadata,
                    score=score,
                    dense_score=float(dense_scores.get(record_id, 0.0)),
                    bm25_score=float(sparse_scores.get(record_id, 0.0)),
                    rank=rank,
                )
            )
        return results


__all__ = [
    "HybridRetriever",
    "RetrievedChunk",
    "metadata_matches",
    "tokenize",
]
