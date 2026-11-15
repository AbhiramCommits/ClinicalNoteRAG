"""Metric primitives for retrieval, QA, hallucination and code extraction."""

from __future__ import annotations

import re
from collections.abc import Iterable

from clinicalnoterag.rag.codes import Code, extract_codes

CITATION_RE = re.compile(r"\[\d+\]")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    """Normalize an answer for exact match: strip citations, case, final periods."""
    cleaned = CITATION_RE.sub("", text or "")
    cleaned = " ".join(cleaned.lower().split())
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.rstrip(" .")


def answer_tokens(text: str) -> list[str]:
    """Tokenize an answer for token-F1 (citations removed, codes kept intact)."""
    return TOKEN_RE.findall(CITATION_RE.sub("", text or "").lower())


def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between a prediction and gold answer."""
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 when normalized answers match exactly."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text, supporting comma thousands separators."""
    values: list[float] = []
    for match in NUMBER_RE.finditer(text or ""):
        raw = match.group(0).replace(",", "")
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def numeric_match(prediction: str, gold: float, rel_tol: float = 1e-6) -> float:
    """Return 1.0 if any number in the prediction matches ``gold``."""
    tolerance = max(1e-9, abs(float(gold)) * rel_tol)
    return (
        1.0
        if any(abs(value - float(gold)) <= tolerance for value in extract_numbers(prediction))
        else 0.0
    )


def retrieval_recall(retrieved: Iterable[str], gold: Iterable[str]) -> float | None:
    """Fraction of gold chunks retrieved at the evaluated cutoff."""
    gold_set = set(gold)
    if not gold_set:
        return None
    return len(set(retrieved) & gold_set) / len(gold_set)


def reciprocal_rank(retrieved: list[str], gold: Iterable[str]) -> float | None:
    """Reciprocal rank of the first gold chunk (0 when none is retrieved)."""
    gold_set = set(gold)
    if not gold_set:
        return None
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def code_set(codes: Iterable[Code] | Iterable[dict[str, str]]) -> set[tuple[str, str]]:
    """Normalize codes to a set of ``(system, code)`` tuples."""
    result: set[tuple[str, str]] = set()
    for code in codes:
        if isinstance(code, Code):
            result.add((code.system, code.code))
        else:
            result.add((str(code["system"]), str(code["code"])))
    return result


def predicted_codes(text: str) -> set[tuple[str, str]]:
    """Extract codes from generated text as a normalized set."""
    return code_set(extract_codes(text))


def code_counts(
    predicted: set[tuple[str, str]], gold: set[tuple[str, str]]
) -> tuple[int, int, int]:
    """Return ``(true_positive, false_positive, false_negative)`` code counts."""
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return precision, recall and F1 from counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


__all__ = [
    "answer_tokens",
    "code_counts",
    "code_set",
    "exact_match",
    "extract_numbers",
    "normalize_answer",
    "numeric_match",
    "prf",
    "predicted_codes",
    "reciprocal_rank",
    "retrieval_recall",
    "token_f1",
]
