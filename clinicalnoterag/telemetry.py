"""Per-query telemetry and a configurable token cost model."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clinicalnoterag.config import Settings


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
) -> float:
    """Estimate USD cost from token counts and per-1k-token prices."""
    return (input_tokens / 1000.0) * cost_per_1k_input + (
        output_tokens / 1000.0
    ) * cost_per_1k_output


@dataclass
class QueryRecord:
    """One query telemetry event."""

    system: str
    backend: str
    question: str
    filters: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    abstained: bool = False
    confidence: float = 0.0
    n_retrieved: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    unsupported_claim: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event."""
        return asdict(self)


class TelemetryRecorder:
    """Append-only JSONL telemetry writer with a latency context manager."""

    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def query(
        self, system: str, backend: str, question: str, filters: dict[str, Any] | None = None
    ) -> Iterator[QueryRecord]:
        """Time a query and append the completed record on exit."""
        record = QueryRecord(
            system=system,
            backend=backend,
            question=question,
            filters=filters or {},
            timestamp=datetime.now(UTC).isoformat(),
        )
        started = time.perf_counter()
        try:
            yield record
        finally:
            record.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            if self.enabled:
                self.write(record)

    def write(self, record: QueryRecord) -> None:
        """Append one record as JSONL (thread-safe within one process)."""
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")


def load_telemetry(path: str | Path, system: str | None = None) -> list[dict[str, Any]]:
    """Load telemetry events, optionally filtered by system name."""
    file = Path(path)
    if not file.exists():
        return []
    events: list[dict[str, Any]] = []
    with file.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            event = json.loads(line)
            if system is None or event.get("system") == system:
                events.append(event)
    return events


def telemetry_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute p50/p95 latency, abstention rate and mean cost from events."""
    if not events:
        return {
            "n_queries": 0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "abstention_rate": 0.0,
            "mean_cost_usd": 0.0,
            "mean_confidence": 0.0,
            "unsupported_claim_rate": 0.0,
        }
    latencies = sorted(float(event.get("latency_ms", 0.0)) for event in events)
    n = len(latencies)
    p50 = latencies[int(0.50 * (n - 1))]
    p95 = latencies[int(0.95 * (n - 1))]
    return {
        "n_queries": n,
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "abstention_rate": round(sum(1 for event in events if event.get("abstained")) / n, 6),
        "mean_cost_usd": round(
            sum(float(event.get("est_cost_usd", 0.0)) for event in events) / n, 8
        ),
        "mean_confidence": round(
            sum(float(event.get("confidence", 0.0)) for event in events) / n, 6
        ),
        "unsupported_claim_rate": round(
            sum(1 for event in events if event.get("unsupported_claim")) / n, 6
        ),
    }


def default_recorder(settings: Settings, system: str = "api") -> TelemetryRecorder:
    """Build a recorder at the configured telemetry path."""
    return TelemetryRecorder(settings.telemetry_path, enabled=True)


__all__ = [
    "QueryRecord",
    "TelemetryRecorder",
    "default_recorder",
    "estimate_cost_usd",
    "load_telemetry",
    "telemetry_summary",
]
