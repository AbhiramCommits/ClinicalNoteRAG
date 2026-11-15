"""FastAPI service exposing citation-grounded clinical QA."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from clinicalnoterag import __version__
from clinicalnoterag.config import Settings, get_settings
from clinicalnoterag.embeddings.backends import load_embedding_state
from clinicalnoterag.rag.answerer import Answer, ClinicalAnswerer
from clinicalnoterag.rag.generators import get_generator
from clinicalnoterag.rag.retriever import HybridRetriever
from clinicalnoterag.telemetry import TelemetryRecorder, load_telemetry, telemetry_summary
from clinicalnoterag.vectorstore.base import FILTER_KEYS
from clinicalnoterag.vectorstore.factory import get_store

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(min_length=3, max_length=2000)
    filters: dict[str, Any] | None = None
    k: int = Field(default=5, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|bm25)$")

    @field_validator("filters")
    @classmethod
    def _validate_filters(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value:
            unknown = set(value) - FILTER_KEYS
            if unknown:
                raise ValueError(f"unsupported filters: {sorted(unknown)}")
        return value


class CitationResponse(BaseModel):
    """A supporting quote from a retrieved chunk."""

    index: int
    chunk_id: str
    patient_id: str | None = None
    encounter_id: str | None = None
    doc_type: str | None = None
    quote: str
    span: dict[str, int]


class AnswerResponse(BaseModel):
    """Response body for grounded answers."""

    question: str
    text: str
    citations: list[CitationResponse]
    confidence: float
    abstained: bool
    abstention_reason: str | None = None
    unsupported_claim: bool
    verification: dict[str, Any]
    retrieval_scores: list[dict[str, Any]]
    latency_ms: float
    token_counts: dict[str, int]
    est_cost_usd: float
    backend_name: str
    filters: dict[str, Any]
    mode: str


class ClaimsDenialRequest(BaseModel):
    """Request body for a claim denial / appeal-support query."""

    claim_id: str = Field(min_length=3, max_length=64)
    patient_id: str | None = None
    k: int = Field(default=10, ge=1, le=50)


class ClaimsDenialResponse(BaseModel):
    """Denial rationale plus appeal support, both citation-grounded."""

    claim_id: str
    answer: AnswerResponse


class HealthResponse(BaseModel):
    """Service health and active-backend summary."""

    status: str
    version: str
    store_backend: str
    store_count: int
    embedding_backend: str
    generator_backend: str
    deid_backend: str


class MetricsResponse(BaseModel):
    """Service metrics derived from real query telemetry."""

    n_queries: int
    p50_latency_ms: float
    p95_latency_ms: float
    abstention_rate: float
    mean_cost_usd: float
    mean_confidence: float
    unsupported_claim_rate: float


def build_answerer(settings: Settings) -> ClinicalAnswerer:
    """Construct the production answerer from persisted ingest state."""
    embedder, state = load_embedding_state(settings)
    store = get_store(settings, embedder.dim, fresh=False)
    retriever = HybridRetriever(store, embedder, settings)
    generator = get_generator(settings)
    telemetry = TelemetryRecorder(settings.telemetry_path, enabled=True)
    logger.info(
        "API answerer ready: store=%s count=%d embedding=%s generator=%s",
        store.name,
        store.count(),
        getattr(embedder, "name", state.get("name")),
        generator.name,
    )
    return ClinicalAnswerer(
        retriever,
        generator=generator,
        settings=settings,
        telemetry=telemetry,
        system_name="api",
    )


def create_app(
    settings: Settings | None = None, answerer: ClinicalAnswerer | None = None
) -> FastAPI:
    """Create the FastAPI app, optionally with an injected answerer (for tests)."""
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.answerer is None:
            app.state.answerer = build_answerer(active_settings)
        yield
        current = app.state.answerer
        if current is not None and hasattr(current.retriever, "store"):
            current.retriever.store.close()

    application = FastAPI(
        title="ClinicalNoteRAG API",
        version=__version__,
        description=(
            "Citation-grounded, abstaining QA over synthetic FHIR bundles and "
            "clinical/payment notes. All data is synthetic."
        ),
        lifespan=lifespan,
    )
    application.state.answerer = answerer
    application.state.settings = active_settings

    def _answerer(request: Request) -> ClinicalAnswerer:
        current = request.app.state.answerer
        if current is None:
            raise HTTPException(status_code=503, detail="answerer is not initialized")
        return current

    @application.get("/healthz", response_model=HealthResponse)
    def healthz(request: Request) -> HealthResponse:
        current = _answerer(request)
        store = current.retriever.store
        return HealthResponse(
            status="ok",
            version=__version__,
            store_backend=store.name,
            store_count=store.count(),
            embedding_backend=current.retriever.embedder.name,
            generator_backend=current.generator.name,
            deid_backend="regex",
        )

    @application.post("/ask", response_model=AnswerResponse)
    def ask(payload: AskRequest, request: Request) -> AnswerResponse:
        current = _answerer(request)
        answer: Answer = current.answer(
            payload.question,
            filters=payload.filters,
            k=payload.k,
            mode=payload.mode,
        )
        return AnswerResponse(**answer.to_dict())

    @application.post("/ask/claims-denial", response_model=ClaimsDenialResponse)
    def claims_denial(payload: ClaimsDenialRequest, request: Request) -> ClaimsDenialResponse:
        current = _answerer(request)
        filters = {"patient_id": payload.patient_id} if payload.patient_id else {}
        question = f"Why was claim {payload.claim_id} denied and what supports an appeal?"
        answer = current.answer(question, filters=filters, k=payload.k)
        return ClaimsDenialResponse(
            claim_id=payload.claim_id, answer=AnswerResponse(**answer.to_dict())
        )

    @application.get("/metrics", response_model=MetricsResponse)
    def metrics(request: Request) -> MetricsResponse:
        settings_obj = request.app.state.settings
        events = load_telemetry(settings_obj.telemetry_path, system="api")
        summary = telemetry_summary(events)
        return MetricsResponse(**summary)

    return application


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("clinicalnoterag.api.app:app", host="0.0.0.0", port=8000, reload=False)
