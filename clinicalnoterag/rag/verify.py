"""Post-hoc citation support verification (token-overlap / NLI-lite)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from clinicalnoterag.rag.codes import extract_codes

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|(?<=[.!?])$")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-]*")

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "was",
    "were",
    "is",
    "are",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "with",
    "by",
    "as",
    "at",
    "from",
    "it",
    "its",
    "patient",
    "patients",
    "documented",
    "record",
    "records",
    "available",
    "according",
    "based",
    "claim",
    "claims",
    "there",
    "their",
    "has",
    "have",
    "had",
    "not",
    "no",
    "which",
    "what",
    "who",
}


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences without breaking on decimals or codes."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", cleaned) if part.strip()
    ]
    return sentences or [cleaned]


def content_tokens(text: str) -> set[str]:
    """Return lowercase, lightly stemmed content tokens used for support scoring."""
    raw = (match.group(0).lower().rstrip(".,;:!?)]}") for match in TOKEN_RE.finditer(text))
    tokens: set[str] = set()
    for token in raw:
        if not token or len(token) <= 1:
            continue
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        if token not in STOPWORDS:
            tokens.add(token)
    return tokens


@dataclass
class SentenceVerification:
    """Support status for one answer sentence."""

    sentence: str
    citation_indices: list[int]
    supported: bool
    best_overlap: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "sentence": self.sentence,
            "citation_indices": self.citation_indices,
            "supported": self.supported,
            "best_overlap": round(self.best_overlap, 4),
            "reason": self.reason,
        }


@dataclass
class VerificationResult:
    """Aggregate verifier result for one answer."""

    sentences: list[SentenceVerification] = field(default_factory=list)
    unsupported_claim: bool = False

    @property
    def unsupported_sentences(self) -> list[SentenceVerification]:
        """Sentences that could not be attributed to any cited chunk."""
        return [sentence for sentence in self.sentences if not sentence.supported]

    @property
    def citation_coverage(self) -> float:
        """Fraction of sentences that are supported by at least one cited chunk."""
        if not self.sentences:
            return 1.0
        return sum(1 for sentence in self.sentences if sentence.supported) / len(self.sentences)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "unsupported_claim": self.unsupported_claim,
            "citation_coverage": round(self.citation_coverage, 4),
            "n_sentences": len(self.sentences),
            "unsupported_sentences": [s.to_dict() for s in self.unsupported_sentences],
            "sentences": [s.to_dict() for s in self.sentences],
        }


def verify_answer(
    text: str,
    contexts: dict[int, Any],
    overlap_threshold: float = 0.30,
) -> VerificationResult:
    """Check that every sentence is supported by at least one cited chunk.

    A sentence is supported when it carries at least one ``[n]`` marker, the
    cited chunk has token overlap of at least ``overlap_threshold`` with the
    sentence's content tokens, and any code in the sentence appears in that
    chunk. This is a deterministic NLI-lite check, not a learned entailment
    model; it is intentionally conservative about fabricated codes.

    Args:
        text: Generated answer, including inline citation markers.
        contexts: Mapping from 1-based citation index to retrieved chunk.
        overlap_threshold: Minimum content-token containment.

    Returns:
        VerificationResult with per-sentence support and the aggregate flag.
    """
    if not text.strip() or text.strip().startswith("Insufficient evidence"):
        return VerificationResult(sentences=[], unsupported_claim=False)
    result = VerificationResult()
    for sentence in split_sentences(text):
        indices = [int(match.group(1)) for match in CITATION_RE.finditer(sentence)]
        body = CITATION_RE.sub("", sentence).strip()
        body_tokens = content_tokens(body)
        sentence_codes = {(code.system, code.code) for code in extract_codes(body)}
        best_overlap = 0.0
        supported = False
        reason = "no citation marker"
        for index in indices:
            context = contexts.get(index)
            if context is None:
                reason = f"citation [{index}] does not map to a retrieved chunk"
                continue
            chunk_tokens = content_tokens(getattr(context, "text", ""))
            if not body_tokens:
                overlap = 0.0
            else:
                overlap = len(body_tokens & chunk_tokens) / len(body_tokens)
            best_overlap = max(best_overlap, overlap)
            if overlap < overlap_threshold:
                reason = f"token overlap {overlap:.2f} below {overlap_threshold:.2f}"
                continue
            if sentence_codes:
                chunk_codes = {
                    (code.system, code.code) for code in extract_codes(getattr(context, "text", ""))
                }
                if not (sentence_codes & chunk_codes):
                    reason = "cited chunk does not contain any code in the sentence"
                    continue
            supported = True
            reason = "supported by cited chunk"
            break
        verification = SentenceVerification(
            sentence=sentence,
            citation_indices=indices,
            supported=supported,
            best_overlap=best_overlap,
            reason=reason,
        )
        result.sentences.append(verification)
        if not supported:
            result.unsupported_claim = True
    return result


__all__ = [
    "CITATION_RE",
    "SentenceVerification",
    "VerificationResult",
    "content_tokens",
    "split_sentences",
    "verify_answer",
]
