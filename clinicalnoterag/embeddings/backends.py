"""Pluggable embedding backends with a deterministic offline fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from clinicalnoterag.config import Settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface implemented by every embedding backend."""

    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a float32 matrix of shape ``(len(texts), dim)``."""
        ...

    def fit(self, texts: list[str]) -> None:
        """Optionally fit corpus statistics (no-op for pretrained backends)."""
        ...

    def state(self) -> dict:
        """Return JSON-serializable state needed to reproduce the encoder."""
        ...


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


class HashingTfidfBackend:
    """Deterministic hashed TF-IDF over word, bigram and character n-grams.

    This backend needs no model download and produces identical vectors across
    processes and machines. It is the documented offline fallback for dense
    retrieval when a sentence-transformer cannot be loaded.
    """

    name = "hashing-tfidf"

    def __init__(self, dim: int = 512, idf: np.ndarray | None = None) -> None:
        if dim < 32:
            raise ValueError("dim must be >= 32")
        self.dim = dim
        self._idf = idf.astype(np.float32) if idf is not None else np.ones(dim, dtype=np.float32)
        if self._idf.shape[0] != dim:
            raise ValueError("idf length must equal dim")
        self.name = f"hashing-tfidf-{dim}"

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        features: list[str] = []
        for token in tokens:
            features.append(f"w:{token}")
            padded = f"<{token}>"
            for n in (3, 4, 5):
                for i in range(max(0, len(padded) - n + 1)):
                    features.append(f"c:{padded[i : i + n]}")
        for first, second in zip(tokens, tokens[1:], strict=False):
            features.append(f"b:{first}_{second}")
        return features

    def _counts(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in self._features(text):
            vector[_stable_hash(feature) % self.dim] += 1.0
        return vector

    def fit(self, texts: list[str]) -> None:
        """Fit inverse document frequency weights over a corpus."""
        n_docs = max(1, len(texts))
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            df += (self._counts(text) > 0).astype(np.float32)
        self._idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts with sublinear TF-IDF and L2 normalization."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = self._counts(text)
            weights = (1.0 + np.log1p(counts)) * self._idf
            weights[counts == 0] = 0.0
            norm = float(np.linalg.norm(weights))
            matrix[i] = weights / norm if norm > 0 else weights
        return matrix.astype(np.float32)

    def state(self) -> dict:
        """Return JSON state including fitted IDF weights."""
        return {"backend": "hashing", "name": self.name, "dim": self.dim, "idf": self._idf.tolist()}


class SentenceTransformersBackend:
    """Sentence-Transformers backend (default model: all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{model_name}"

    def fit(self, texts: list[str]) -> None:
        """Pretrained encoders need no fitting."""
        return

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode and L2-normalize texts."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return np.asarray(vectors, dtype=np.float32)

    def state(self) -> dict:
        """Return the model identity needed to reproduce the encoder."""
        return {
            "backend": "sentence-transformers",
            "name": self.name,
            "dim": self.dim,
            "model": self.model_name,
        }


def get_embedding_backend(settings: Settings) -> EmbeddingBackend:
    """Resolve the embedding backend, falling back to deterministic hashing.

    ``embedding_backend=auto`` first tries Sentence-Transformers and logs the
    fallback. The returned object exposes ``name`` so callers record exactly
    which backend ran.
    """
    choice = settings.embedding_backend
    if choice == "hashing":
        return HashingTfidfBackend(dim=settings.embedding_dim)
    if choice in {"auto", "sentence-transformers"}:
        try:
            backend = SentenceTransformersBackend(settings.embedding_model)
            logger.info("Embedding backend: %s (dim=%d)", backend.name, backend.dim)
            return backend
        except Exception as exc:  # noqa: BLE001 - any load failure triggers fallback
            if choice == "sentence-transformers":
                raise RuntimeError(f"sentence-transformers backend failed: {exc}") from exc
            logger.warning(
                "Sentence-Transformers unavailable (%s: %s); using deterministic hashing fallback",
                type(exc).__name__,
                exc,
            )
            return HashingTfidfBackend(dim=settings.embedding_dim)
    raise ValueError(f"unknown embedding backend: {choice}")


def state_path_for(settings: Settings) -> Path:
    """Path where the ingest-time embedding state is persisted."""
    return settings.artifact_dir / "embedder" / "state.json"


def save_embedding_state(embedder: EmbeddingBackend, path: Path) -> None:
    """Persist the encoder state so query-time vectors match ingest-time vectors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embedder.state()), encoding="utf-8")


def load_embedding_state(settings: Settings) -> tuple[EmbeddingBackend, dict]:
    """Load the ingest-time encoder, returning the backend and its state."""
    path = state_path_for(settings)
    if not path.exists():
        backend = get_embedding_backend(settings)
        return backend, backend.state()
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("backend") == "hashing":
        backend: EmbeddingBackend = HashingTfidfBackend(
            dim=int(state["dim"]), idf=np.asarray(state["idf"], dtype=np.float32)
        )
    elif state.get("backend") == "sentence-transformers":
        backend = SentenceTransformersBackend(str(state["model"]))
    else:
        raise ValueError(f"unknown embedding state: {state.get('backend')}")
    return backend, state


__all__ = [
    "EmbeddingBackend",
    "HashingTfidfBackend",
    "SentenceTransformersBackend",
    "get_embedding_backend",
    "load_embedding_state",
    "save_embedding_state",
    "state_path_for",
]
