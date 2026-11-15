"""Small text utilities shared by generators and metrics."""

from __future__ import annotations


def count_tokens(text: str) -> int:
    """Approximate token count with whitespace tokenization.

    This is used only for cost accounting on the deterministic backend; model
    backends report their tokenizer's real counts.
    """
    return len(text.split())


__all__ = ["count_tokens"]
