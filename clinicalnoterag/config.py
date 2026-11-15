"""Typed application configuration.

All values are environment-overridable and have offline-safe defaults. The
default backends deliberately prefer deterministic local components so that a
fresh checkout can run every command without a network connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingBackendName = Literal["auto", "sentence-transformers", "hashing"]
LLMBackendName = Literal["auto", "hf-local", "extractive", "openai"]
VectorBackendName = Literal["faiss", "pgvector"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- reproducibility -------------------------------------------------
    seed: int = Field(default=42, description="Global RNG seed (hard rule: 42).")

    # --- paths -----------------------------------------------------------
    artifact_dir: Path = Field(default=Path("artifacts"))
    data_dir: Path = Field(default=Path("data"))
    sample_dir: Path = Field(default=Path("data/sample"))

    # --- pluggable backends ---------------------------------------------
    embedding_backend: EmbeddingBackendName = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=512, ge=32, le=4096)
    llm_backend: LLMBackendName = "auto"
    llm_primary_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_fallback_model: str = "google/flan-t5-base"
    vector_backend: VectorBackendName = "faiss"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/clinicalnoterag"

    # --- retrieval -------------------------------------------------------
    chunk_target_tokens: int = Field(default=256, ge=32)
    chunk_overlap_tokens: int = Field(default=32, ge=0)
    retrieval_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal-rank-fusion constant.")
    abstention_threshold: float = Field(
        default=0.012,
        ge=0.0,
        description="Minimum fused top score required to answer.",
    )

    # --- generation / cost model ----------------------------------------
    max_new_tokens: int = Field(default=256, ge=16)
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    cost_per_1k_input_tokens_usd: float = Field(default=0.0, ge=0.0)
    cost_per_1k_output_tokens_usd: float = Field(default=0.0, ge=0.0)

    # --- service ---------------------------------------------------------
    faiss_index_path: Path = Field(default=Path("artifacts/index/clinicalnoterag.faiss"))
    sqlite_path: Path = Field(default=Path("artifacts/index/clinicalnoterag.sqlite3"))
    telemetry_path: Path = Field(default=Path("artifacts/query_telemetry.jsonl"))

    # --- training --------------------------------------------------------
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: str = "q_proj,v_proj"
    max_steps: int = Field(default=20, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)

    def ensure_dirs(self) -> None:
        """Create artifact and data directories if they do not exist."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides: object) -> Settings:
    """Return settings, optionally with explicit overrides (useful in tests)."""
    return Settings(**overrides)  # type: ignore[arg-type]


__all__ = [
    "EmbeddingBackendName",
    "LLMBackendName",
    "Settings",
    "VectorBackendName",
    "get_settings",
]
