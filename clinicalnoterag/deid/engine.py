"""PHI de-identification engine with a Presidio backend and a regex fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clinicalnoterag.deid.patterns import PRIORITY, PatternEntity, find_entities

logger = logging.getLogger(__name__)


@dataclass
class DeidEntity:
    """One de-identified span and the reversible placeholder that replaced it."""

    entity_type: str
    start: int
    end: int
    original: str
    placeholder: str
    backend: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation (contains the original PHI)."""
        return asdict(self)


@dataclass
class DeidResult:
    """Result of scrubbing one document."""

    text: str
    entities: list[DeidEntity]
    backend_name: str
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def entity_counts(self) -> dict[str, int]:
        """Count replaced entities by type."""
        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        """Serialize the result. The mapping is PHI-bearing and must be stored separately."""
        payload: dict[str, Any] = {
            "backend_name": self.backend_name,
            "entities": [e.to_dict() for e in self.entities],
            "entity_counts": self.entity_counts,
            "mapping": self.mapping,
        }
        if include_text:
            payload["text"] = self.text
        return payload


_PRESIDIO_MAP = {
    "PERSON": "NAME",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "US_SSN": "SSN",
    "DATE_TIME": "DATE",
    "LOCATION": "ADDRESS",
    "US_ZIP_CODE": "ZIP",
}


class DeidEngine:
    """Scrub PHI with typed, reversible placeholders.

    Backend selection:
      * ``presidio``: Microsoft Presidio ``AnalyzerEngine`` (plus an
        ``AnonymizerEngine`` residual-leak pass) when the package and its spaCy
        model load successfully.
      * ``regex``: deterministic gazetteer + pattern fallback. This is the
        offline-safe default.

    The reversible mapping is returned to the caller and is never written into
    the vector store by the ingestion pipeline.
    """

    def __init__(
        self,
        backend: str = "auto",
        known_names: list[str] | None = None,
    ) -> None:
        if backend not in {"auto", "presidio", "regex"}:
            raise ValueError(f"unknown de-id backend: {backend}")
        self.known_names = known_names
        self._analyzer: Any = None
        self._anonymizer: Any = None
        self.backend_name = "regex"
        self.selection_reason = "regex fallback selected"
        if backend in {"auto", "presidio"}:
            self._try_presidio()
        if backend == "presidio" and self.backend_name != "presidio":
            raise RuntimeError(
                "Presidio was requested but could not be initialized: " + self.selection_reason
            )

    def _try_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self.backend_name = "presidio"
            self.selection_reason = "Presidio AnalyzerEngine + AnonymizerEngine loaded"
        except Exception as exc:  # noqa: BLE001 - any import/model failure means fallback
            self.selection_reason = f"Presidio unavailable ({type(exc).__name__}: {exc})"
            logger.info("Falling back to regex de-identification: %s", self.selection_reason)

    def _presidio_entities(self, text: str) -> list[PatternEntity]:
        if self._analyzer is None:
            return []
        try:
            results = self._analyzer.analyze(text=text, language="en")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Presidio analysis failed; using regex only for this document: %s", exc)
            return []
        out: list[PatternEntity] = []
        for result in results:
            entity_type = _PRESIDIO_MAP.get(result.entity_type)
            if entity_type is None or not (0 <= result.start < result.end <= len(text)):
                continue
            out.append(
                PatternEntity(
                    entity_type, result.start, result.end, text[result.start : result.end]
                )
            )
        return out

    def _residual_presidio_redaction(self, text: str) -> str:
        """Use the Presidio anonymizer on any residual detections (safety net)."""
        if self._analyzer is None or self._anonymizer is None:
            return text
        try:
            residual = self._analyzer.analyze(text=text, language="en")
            if not residual:
                return text
            from presidio_anonymizer.entities import OperatorConfig

            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=residual,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})},
            )
            return anonymized.text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Presidio anonymizer residual pass failed: %s", exc)
            return text

    @staticmethod
    def _merge_spans(
        primary: list[PatternEntity], fallback: list[PatternEntity]
    ) -> list[PatternEntity]:
        candidates = [*primary, *fallback]
        candidates.sort(
            key=lambda e: (e.start, -(e.end - e.start), -PRIORITY.get(e.entity_type, 1))
        )
        selected: list[PatternEntity] = []
        for candidate in candidates:
            if candidate.start == candidate.end:
                continue
            if any(
                candidate.start < chosen.end and chosen.start < candidate.end for chosen in selected
            ):
                continue
            selected.append(candidate)
        return sorted(selected, key=lambda e: e.start)

    def scrub(self, text: str) -> DeidResult:
        """Replace PHI in ``text`` with typed placeholders such as ``[NAME_1]``.

        Args:
            text: A single note or document.

        Returns:
            DeidResult with the scrubbed text, entity spans, selected backend
            name, and a placeholder -> original reversible mapping.
        """
        regex_entities = find_entities(text, extra_names=self.known_names)
        if self.backend_name == "presidio":
            entities = self._merge_spans(self._presidio_entities(text), regex_entities)
        else:
            entities = regex_entities

        counters: dict[str, int] = {}
        assignments: list[tuple[PatternEntity, str]] = []
        for entity in entities:
            counters[entity.entity_type] = counters.get(entity.entity_type, 0) + 1
            placeholder = f"[{entity.entity_type}_{counters[entity.entity_type]}]"
            assignments.append((entity, placeholder))

        pieces: list[str] = []
        cursor = 0
        for entity, placeholder in assignments:
            pieces.append(text[cursor : entity.start])
            pieces.append(placeholder)
            cursor = entity.end
        pieces.append(text[cursor:])
        scrubbed = "".join(pieces)

        if self.backend_name == "presidio":
            scrubbed = self._residual_presidio_redaction(scrubbed)

        deid_entities = [
            DeidEntity(
                entity_type=entity.entity_type,
                start=entity.start,
                end=entity.end,
                original=entity.text,
                placeholder=placeholder,
                backend=self.backend_name,
            )
            for entity, placeholder in assignments
        ]
        mapping = {placeholder: entity.text for entity, placeholder in assignments}
        return DeidResult(
            text=scrubbed,
            entities=deid_entities,
            backend_name=self.backend_name,
            mapping=mapping,
        )


def load_known_names(manifest_path: str | Path) -> list[str]:
    """Load the synthetic-name gazetteer from a corpus manifest."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return list(data.get("known_names", []))


__all__ = ["DeidEngine", "DeidEntity", "DeidResult", "load_known_names"]
