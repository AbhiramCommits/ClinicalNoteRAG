"""Vector store construction from settings."""

from __future__ import annotations

from clinicalnoterag.config import Settings
from clinicalnoterag.vectorstore.base import VectorStore


def get_store(
    settings: Settings,
    dim: int,
    store_name: str | None = None,
    fresh: bool = False,
) -> VectorStore:
    """Build the configured vector store.

    Args:
        settings: Application settings.
        dim: Embedding dimension.
        store_name: ``faiss`` or ``pgvector``; defaults to settings.
        fresh: Drop/recreate persisted state.

    Returns:
        A store implementing :class:`clinicalnoterag.vectorstore.base.VectorStore`.
    """
    name = store_name or settings.vector_backend
    if name == "faiss":
        from clinicalnoterag.vectorstore.faiss_store import FaissStore

        return FaissStore(
            dim=dim,
            index_path=settings.faiss_index_path,
            sqlite_path=settings.sqlite_path,
            fresh=fresh,
        )
    if name == "pgvector":
        from clinicalnoterag.vectorstore.pgvector_store import PgVectorStore

        return PgVectorStore(settings.postgres_dsn, dim=dim, fresh=fresh)
    raise ValueError(f"unknown vector store: {name}")


__all__ = ["get_store"]
