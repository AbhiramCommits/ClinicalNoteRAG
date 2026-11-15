"""PHI leak auditing over de-identified documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from clinicalnoterag.deid.engine import DeidEngine


@dataclass
class LeakRecord:
    """Leak status for one document."""

    doc_id: str
    checked: dict[str, int] = field(default_factory=dict)
    leaked: dict[str, int] = field(default_factory=dict)
    leaked_values: dict[str, list[str]] = field(default_factory=dict)

    @property
    def any_leak(self) -> bool:
        """True if any injected PHI string survived scrubbing."""
        return any(self.leaked.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize without echoing the leaked PHI values in aggregate output."""
        return {
            "doc_id": self.doc_id,
            "checked": self.checked,
            "leaked": self.leaked,
            "leaked_value_count": {k: len(v) for k, v in self.leaked_values.items()},
            "any_leak": self.any_leak,
        }


@dataclass
class LeakReport:
    """Aggregate PHI leak audit result."""

    total_documents: int
    leaked_documents: int
    phi_leak_rate: float
    per_entity_checked: dict[str, int]
    per_entity_leaked: dict[str, int]
    per_entity_recall: dict[str, float]
    backend_name: str
    backend_selection_reason: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable audit artifact."""
        data = asdict(self)
        return data


def _iter_injected(record: Mapping[str, Any]) -> dict[str, list[str]]:
    injected = record.get("injected_phi") or {}
    out: dict[str, list[str]] = {}
    for entity_type, values in injected.items():
        if isinstance(values, str):
            out[entity_type] = [values]
        else:
            out[entity_type] = [str(v) for v in values]
    return out


def scan_corpus(
    records: Iterable[Mapping[str, Any]],
    engine: DeidEngine | None = None,
) -> LeakReport:
    """Re-scan scrubbed documents for the injected PHI strings.

    Each record must provide ``doc_id``, ``injected_phi`` and either
    ``scrubbed_text`` (already de-identified) or ``text`` plus an ``engine``.
    A document counts as leaked if any known injected PHI string appears in the
    scrubbed text. Per-entity recall is ``1 - leaked_values / checked_values``.

    Args:
        records: Note records (or any mappings with the required keys).
        engine: De-identification engine used when ``scrubbed_text`` is absent.

    Returns:
        LeakReport with real measured rates and the backend that ran.
    """
    per_checked: dict[str, int] = {}
    per_leaked: dict[str, int] = {}
    backend_names: set[str] = set()
    records_out: list[dict[str, Any]] = []
    total = leaked_docs = 0

    for record in records:
        total += 1
        doc_id = str(record.get("doc_id", f"doc-{total}"))
        injected = _iter_injected(record)
        if "scrubbed_text" in record:
            scrubbed = str(record["scrubbed_text"])
            backend_name = str(record.get("deid_backend", "provided"))
        else:
            if engine is None:
                raise ValueError("engine is required when records do not contain scrubbed_text")
            result = engine.scrub(str(record["text"]))
            scrubbed = result.text
            backend_name = result.backend_name
        backend_names.add(backend_name)

        leak_record = LeakRecord(doc_id=doc_id)
        for entity_type, values in injected.items():
            checked = per_checked.setdefault(entity_type, 0)
            leaked = per_leaked.setdefault(entity_type, 0)
            for value in values:
                checked += 1
                if value and value in scrubbed:
                    leaked += 1
                    leak_record.leaked_values.setdefault(entity_type, []).append(value)
                leak_record.checked[entity_type] = leak_record.checked.get(entity_type, 0) + 1
                leak_record.leaked[entity_type] = leak_record.leaked.get(entity_type, 0)
            per_checked[entity_type] = checked
            per_leaked[entity_type] = leaked
        if leak_record.any_leak:
            leaked_docs += 1
        records_out.append(leak_record.to_dict())

    recall = {
        entity_type: (1.0 - (per_leaked[entity_type] / checked) if checked else 1.0)
        for entity_type, checked in per_checked.items()
    }
    backend_name = ",".join(sorted(backend_names)) if backend_names else "none"
    return LeakReport(
        total_documents=total,
        leaked_documents=leaked_docs,
        phi_leak_rate=(leaked_docs / total) if total else 0.0,
        per_entity_checked=dict(sorted(per_checked.items())),
        per_entity_leaked=dict(sorted(per_leaked.items())),
        per_entity_recall=dict(sorted(recall.items())),
        backend_name=backend_name,
        backend_selection_reason=engine.selection_reason if engine else "provided",
        records=records_out,
    )


__all__ = ["LeakRecord", "LeakReport", "scan_corpus"]
