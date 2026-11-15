"""Citation-grounded, abstaining clinical RAG answerer."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from clinicalnoterag.config import Settings
from clinicalnoterag.rag.codes import extract_codes
from clinicalnoterag.rag.generators import ABSTENTION_TEXT, Generator, get_generator
from clinicalnoterag.rag.retriever import HybridRetriever, RetrievedChunk
from clinicalnoterag.rag.verify import (
    CITATION_RE,
    content_tokens,
    split_sentences,
    verify_answer,
)
from clinicalnoterag.telemetry import TelemetryRecorder

HARD_ID_RE = re.compile(r"\b(?:PAT-\d{4}|ENC-\d{4}-\d{2}|CLM-\d{4}-\d{2}|(?:CO|PR|OA)-\d{1,3})\b")
CPT_QUESTION_RE = re.compile(r"\bCPT\s*[:#]?\s*(\d{5})\b", re.IGNORECASE)


@dataclass
class Citation:
    """A citation to a retrieved chunk with the quoted supporting span."""

    index: int
    chunk_id: str
    patient_id: str | None
    encounter_id: str | None
    doc_type: str | None
    quote: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable citation."""
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "patient_id": self.patient_id,
            "encounter_id": self.encounter_id,
            "doc_type": self.doc_type,
            "quote": self.quote,
            "span": {"start": self.start, "end": self.end},
        }


@dataclass
class Answer:
    """A grounded answer or an explicit abstention."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    abstention_reason: str | None = None
    unsupported_claim: bool = False
    verification: dict[str, Any] = field(default_factory=dict)
    retrieval_scores: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    token_counts: dict[str, int] = field(default_factory=dict)
    est_cost_usd: float = 0.0
    backend_name: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    mode: str = "hybrid"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable answer for API and artifacts."""
        return {
            "question": self.question,
            "text": self.text,
            "citations": [citation.to_dict() for citation in self.citations],
            "confidence": self.confidence,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "unsupported_claim": self.unsupported_claim,
            "verification": self.verification,
            "retrieval_scores": self.retrieval_scores,
            "latency_ms": self.latency_ms,
            "token_counts": self.token_counts,
            "est_cost_usd": self.est_cost_usd,
            "backend_name": self.backend_name,
            "filters": self.filters,
            "mode": self.mode,
        }


def _quote_span(chunk_text: str, sentence: str) -> tuple[str, int, int]:
    """Pick the chunk sentence with the highest overlap and return its offsets."""
    best_sentence = ""
    best_overlap = 0.0
    sentence_tokens = content_tokens(sentence)
    for candidate in split_sentences(chunk_text):
        candidate_tokens = content_tokens(candidate)
        if not candidate_tokens:
            continue
        overlap = len(sentence_tokens & candidate_tokens) / max(1, len(sentence_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = candidate
    if not best_sentence:
        best_sentence = chunk_text[:300]
    start = chunk_text.find(best_sentence)
    if start < 0:
        start = 0
        best_sentence = chunk_text[:300]
    return best_sentence, start, start + len(best_sentence)


def _context_supports_entity(entity: str, contexts: list[RetrievedChunk]) -> bool:
    entity_lower = entity.lower()
    for chunk in contexts:
        if entity_lower in chunk.text.lower():
            return True
        for key in ("patient_id", "encounter_id", "claim_id", "code", "denial_code"):
            value = str(chunk.metadata.get(key) or "")
            if value and value.lower() == entity_lower:
                return True
    return False


def key_entities(question: str) -> list[str]:
    """Extract hard identifiers/codes that a grounded answer must find in context."""
    entities = {match.group(0) for match in HARD_ID_RE.finditer(question)}
    for code in extract_codes(question):
        if code.system in {"ICD-10", "CPT", "CARC"}:
            entities.add(code.code)
    for match in CPT_QUESTION_RE.finditer(question):
        entities.add(match.group(1))
    return sorted(entities)


class ClinicalAnswerer:
    """Retrieve, generate, cite, verify, and abstain when evidence is missing."""

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator | None = None,
        settings: Settings | None = None,
        telemetry: TelemetryRecorder | None = None,
        system_name: str = "rag-zero-shot",
        reranker: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.settings = settings or retriever.settings
        self.generator = generator or get_generator(self.settings)
        self.telemetry = telemetry or TelemetryRecorder(self.settings.telemetry_path, enabled=False)
        self.system_name = system_name
        self.reranker = reranker
        self.reranker_name = getattr(reranker, "name", None)

    def _abstain(
        self, question: str, contexts: list[RetrievedChunk], reason: str, mode: str
    ) -> Answer:
        return Answer(
            question=question,
            text=ABSTENTION_TEXT,
            citations=[],
            confidence=0.0,
            abstained=True,
            abstention_reason=reason,
            unsupported_claim=False,
            retrieval_scores=[chunk.to_dict() for chunk in contexts],
            backend_name=self.generator.name,
            filters={},
            mode=mode,
        )

    def answer(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        k: int | None = None,
        mode: str = "hybrid",
    ) -> Answer:
        """Answer a question with citations, or abstain with zero fabricated content.

        Abstention is triggered when retrieval is empty, when the fused score is
        below the configured threshold, when any hard identifier in the question
        is absent from retrieved evidence, or when the generator reports that
        the requested evidence is absent.
        """
        started = time.perf_counter()
        active_filters = dict(filters or {})
        with self.telemetry.query(
            system=self.system_name,
            backend=self.generator.name,
            question=question,
            filters=active_filters,
        ) as record:
            requested_k = k or self.settings.retrieval_k
            if self.reranker is not None:
                candidates = self.retriever.retrieve(
                    question,
                    k=max(20, requested_k * 4),
                    filters=active_filters,
                    mode=mode,
                )
                contexts = self.reranker.rerank(question, candidates, top_k=requested_k)
            else:
                contexts = self.retriever.retrieve(
                    question, k=requested_k, filters=active_filters, mode=mode
                )
            if not contexts:
                answer = self._abstain(question, contexts, "no_retrieval", mode)
            elif contexts[0].score < self.settings.abstention_threshold:
                answer = self._abstain(question, contexts, "low_retrieval_score", mode)
            else:
                missing = [
                    entity
                    for entity in key_entities(question)
                    if not _context_supports_entity(entity, contexts)
                ]
                if missing:
                    answer = self._abstain(
                        question, contexts, f"missing_key_entity:{','.join(missing)}", mode
                    )
                else:
                    result = self.generator.generate(
                        question,
                        contexts,
                        filters=active_filters,
                        max_new_tokens=self.settings.max_new_tokens,
                    )
                    if result.insufficient_evidence or result.text.strip() == ABSTENTION_TEXT:
                        answer = self._abstain(question, contexts, "generator_insufficient", mode)
                        answer.token_counts = {
                            "input": result.input_tokens,
                            "output": result.output_tokens,
                        }
                    elif not result.text.strip():
                        answer = self._abstain(question, contexts, "empty_generation", mode)
                    else:
                        citation_map = dict(enumerate(contexts, start=1))
                        verification = verify_answer(result.text, citation_map)
                        citations = self._build_citations(result.text, contexts)
                        top_possible = 2.0 / (self.settings.rrf_k + 1)
                        confidence = min(1.0, contexts[0].score / top_possible)
                        if verification.unsupported_claim:
                            confidence *= 0.5
                        answer = Answer(
                            question=question,
                            text=result.text,
                            citations=citations,
                            confidence=round(confidence, 4),
                            abstained=False,
                            unsupported_claim=verification.unsupported_claim,
                            verification=verification.to_dict(),
                            retrieval_scores=[chunk.to_dict() for chunk in contexts],
                            token_counts={
                                "input": result.input_tokens,
                                "output": result.output_tokens,
                            },
                            est_cost_usd=result.est_cost_usd,
                            backend_name=result.backend_name,
                            filters=active_filters,
                            mode=mode,
                        )
            answer.filters = active_filters
            answer.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            record.abstained = answer.abstained
            record.confidence = answer.confidence
            record.n_retrieved = len(contexts)
            record.input_tokens = answer.token_counts.get("input", 0)
            record.output_tokens = answer.token_counts.get("output", 0)
            record.est_cost_usd = answer.est_cost_usd
            record.unsupported_claim = answer.unsupported_claim
        answer.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return answer

    @staticmethod
    def _build_citations(text: str, contexts: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[int] = set()
        for sentence in split_sentences(text):
            for match in CITATION_RE.finditer(sentence):
                index = int(match.group(1))
                if index in seen or not (1 <= index <= len(contexts)):
                    continue
                seen.add(index)
                chunk = contexts[index - 1]
                quote, start, end = _quote_span(chunk.text, sentence)
                citations.append(
                    Citation(
                        index=index,
                        chunk_id=chunk.chunk_id,
                        patient_id=chunk.patient_id,
                        encounter_id=chunk.encounter_id,
                        doc_type=chunk.doc_type,
                        quote=quote,
                        start=start,
                        end=end,
                    )
                )
        return citations


__all__ = [
    "Answer",
    "Citation",
    "ClinicalAnswerer",
    "key_entities",
]
