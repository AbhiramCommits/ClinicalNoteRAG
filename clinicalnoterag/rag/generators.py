"""Generator backends: local HF, commercial API, and deterministic extractive fallback."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from clinicalnoterag.config import Settings
from clinicalnoterag.embedding_utils import count_tokens
from clinicalnoterag.rag.retriever import RetrievedChunk
from clinicalnoterag.telemetry import estimate_cost_usd

logger = logging.getLogger(__name__)

ABSTENTION_TEXT = "Insufficient evidence in the available record"

CLAIM_RE = re.compile(r"\bCLM-\d{4}-\d{2}\b")
ENCOUNTER_RE = re.compile(r"\bENC-\d{4}-\d{2}\b")
PATIENT_RE = re.compile(r"\bPAT-\d{4}\b")
CARC_RE = re.compile(r"\b(?:CO|PR|OA)-\d{1,3}\b")
ICD_RE = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?\b")
CPT_RE = re.compile(r"\b\d{5}\b")


@dataclass
class GenerationResult:
    """One generator call's text and token/cost accounting."""

    text: str
    backend_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    est_cost_usd: float = 0.0
    insufficient_evidence: bool = False


@runtime_checkable
class Generator(Protocol):
    """Interface for answer generators."""

    name: str

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        filters: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate an answer grounded in ``contexts``."""
        ...


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    """Build the grounded-answer prompt shared by local and API generators."""
    context_block = "\n".join(
        f"[{index}] ({chunk.metadata.get('doc_type', 'record')}; "
        f"patient {chunk.metadata.get('patient_id', 'unknown')}; "
        f"encounter {chunk.metadata.get('encounter_id', 'unknown')}) {chunk.text}"
        for index, chunk in enumerate(contexts, start=1)
    )
    return (
        "You are a clinical records assistant supporting treatment, payment and "
        "operations review. Answer the question using ONLY the numbered context "
        "below. Cite every sentence with inline markers like [1] or [2] that map "
        "to the context numbers. Do not use outside knowledge. If the context "
        f'does not contain the answer, respond exactly: "{ABSTENTION_TEXT}"\n\n'
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )


def _cite(index: int) -> str:
    return f"[{index}]"


def _sentence(text: str, indices: list[int]) -> str:
    marks = "".join(_cite(index) for index in indices)
    return f"{text.rstrip('.')} {marks}."


class ExtractiveGenerator:
    """Deterministic offline generator that stitches supporting context sentences.

    It never invents content: every statement is a rendered FHIR fact or a
    verbatim sentence from a retrieved chunk. It is the reproducible fallback
    used for the full evaluation when a local instruct model is not enabled.
    """

    name = "extractive-offline"

    def __init__(self) -> None:
        self.selection_reason = "deterministic extractive backend"

    @staticmethod
    def _facts(
        contexts: list[RetrievedChunk], resource_type: str
    ) -> list[tuple[int, RetrievedChunk]]:
        facts = [
            (index, chunk)
            for index, chunk in enumerate(contexts, start=1)
            if chunk.metadata.get("doc_type") == "fhir_fact"
            and chunk.metadata.get("resource_type") == resource_type
        ]
        return sorted(
            facts, key=lambda item: (str(item[1].metadata.get("code") or ""), item[1].chunk_id)
        )

    @staticmethod
    def _claim(
        contexts: list[RetrievedChunk], claim_id: str | None
    ) -> tuple[int, RetrievedChunk] | None:
        for index, chunk in enumerate(contexts, start=1):
            if chunk.metadata.get("doc_type") != "fhir_fact":
                continue
            if chunk.metadata.get("resource_type") != "Claim":
                continue
            if chunk.metadata.get("claim_id") == claim_id or (claim_id and claim_id in chunk.text):
                return index, chunk
        return None

    @staticmethod
    def _claim_response(
        contexts: list[RetrievedChunk], claim_id: str | None
    ) -> tuple[int, RetrievedChunk] | None:
        for index, chunk in enumerate(contexts, start=1):
            if chunk.metadata.get("resource_type") == "ClaimResponse" and (
                chunk.metadata.get("claim_id") == claim_id or (claim_id and claim_id in chunk.text)
            ):
                return index, chunk
        return None

    @staticmethod
    def _parse_denial_reason(chunk: RetrievedChunk) -> str:
        match = re.search(
            r"Denial reason code: (?:CO|PR|OA)-\d+ \((.*)\)\. Disposition:", chunk.text
        )
        if not match:
            return ""
        return match.group(1).strip().replace(". ", "; ").rstrip(".")

    @staticmethod
    def _parse_observation(chunk: RetrievedChunk) -> tuple[str, str, str]:
        value_match = re.search(r"Value: (.*?)\. Date:", chunk.text)
        date_match = re.search(r"Date: ([^.]*)\.", chunk.text)
        display = str(chunk.metadata.get("code_display") or "")
        return (
            display,
            value_match.group(1).strip() if value_match else "",
            date_match.group(1).strip()
            if date_match
            else str(chunk.metadata.get("service_date") or ""),
        )

    @staticmethod
    def _parse_claim_amount(chunk: RetrievedChunk, key: str) -> float:
        metadata_key = {
            "billed": "billed_amount",
            "allowed": "allowed_amount",
            "paid": "paid_amount",
        }[key]
        return float(chunk.metadata.get(metadata_key) or 0.0)

    @staticmethod
    def _parse_primary_dx(chunk: RetrievedChunk) -> tuple[str, str]:
        match = re.search(
            r"Supporting diagnosis: ICD-10 ([A-Z0-9.]+) \((.*)\)\. All procedure codes:",
            chunk.text,
        )
        if not match:
            match = re.search(r"Supporting diagnosis: ICD-10 ([A-Z0-9.]+) \(([^)]+)\)", chunk.text)
        if match:
            return match.group(1), match.group(2)
        return "", ""

    @staticmethod
    def _parse_primary_cpt(chunk: RetrievedChunk) -> tuple[str, str]:
        match = re.search(r"Primary procedure: CPT (\d{5}) \(([^)]+)\)", chunk.text)
        if match:
            return match.group(1), match.group(2)
        return str(chunk.metadata.get("code") or ""), str(chunk.metadata.get("code_display") or "")

    @staticmethod
    def _count_sentence(
        count: int, singular: str, plural: str, patient: str, indices: list[int]
    ) -> str:
        verb = "is" if count == 1 else "are"
        label = singular if count == 1 else plural
        return _sentence(f"There {verb} {count} {label} for patient {patient}", indices)

    @staticmethod
    def _patient_id(question: str, filters: dict[str, Any] | None) -> str:
        match = PATIENT_RE.search(question)
        if match:
            return match.group(0)
        return str((filters or {}).get("patient_id") or "the patient")

    def _insufficient(self, question: str) -> GenerationResult:
        return GenerationResult(
            text=ABSTENTION_TEXT,
            backend_name=self.name,
            input_tokens=count_tokens(question),
            output_tokens=count_tokens(ABSTENTION_TEXT),
            insufficient_evidence=True,
        )

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        filters: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Answer by selecting facts and stitching cited sentences."""
        if not contexts:
            return self._insufficient(question)
        q = question.lower()
        clauses = self._generate_core(question, q, contexts, filters)
        if clauses is None:
            return self._insufficient(question)
        text = " ".join(clauses)
        return GenerationResult(
            text=text,
            backend_name=self.name,
            input_tokens=count_tokens(question) + sum(count_tokens(c.text) for c in contexts),
            output_tokens=count_tokens(text),
            est_cost_usd=0.0,
            insufficient_evidence=False,
        )

    def _generate_core(
        self,
        question: str,
        q: str,
        contexts: list[RetrievedChunk],
        filters: dict[str, Any] | None,
    ) -> list[str] | None:
        claim_match = CLAIM_RE.search(question)
        claim_id = claim_match.group(0) if claim_match else None
        patient = self._patient_id(question, filters)
        condition_facts = self._facts(contexts, "Condition")
        procedure_facts = self._facts(contexts, "Procedure")
        claim_facts = self._facts(contexts, "Claim")
        encounter_facts = self._facts(contexts, "Encounter")
        observation_facts = self._facts(contexts, "Observation")
        medication_facts = self._facts(contexts, "MedicationRequest")

        # --- claims denial / payment operations -------------------------
        if "appeal" in q or ("why" in q and "denied" in q):
            claim = self._claim(contexts, claim_id)
            if claim is None:
                return None
            claim_index, claim_chunk = claim
            if not claim_chunk.metadata.get("denied"):
                return [
                    _sentence(
                        f"Claim {claim_id} was not denied; it was paid "
                        f"${self._parse_claim_amount(claim_chunk, 'paid'):,.2f}",
                        [claim_index],
                    )
                ]
            response = self._claim_response(contexts, claim_id)
            reason = self._parse_denial_reason(response[1]) if response else ""
            code = str(claim_chunk.metadata.get("denial_code") or "")
            reason_text = f"{code}: {reason}" if reason else code
            clauses = [
                _sentence(
                    f"Claim {claim_id} was denied with CARC reason {reason_text}",
                    [response[0] if response else claim_index],
                )
            ]
            cpt_code, cpt_display = self._parse_primary_cpt(claim_chunk)
            if cpt_code:
                clauses.append(
                    _sentence(
                        f"The primary procedure associated with the denial was CPT {cpt_code} ({cpt_display})",
                        [claim_index],
                    )
                )
            dx_code, dx_display = self._parse_primary_dx(claim_chunk)
            if dx_code:
                clauses.append(
                    _sentence(
                        f"The supporting diagnosis for the appeal was ICD-10 {dx_code} ({dx_display})",
                        [claim_index],
                    )
                )
            return clauses

        if "denied" in q and ("was claim" in q or claim_id):
            claim = self._claim(contexts, claim_id)
            if claim is None:
                return None
            index, chunk = claim
            denied = bool(chunk.metadata.get("denied"))
            code = str(chunk.metadata.get("denial_code") or "")
            if denied:
                response = self._claim_response(contexts, claim_id)
                reason = self._parse_denial_reason(response[1]) if response else ""
                detail = f" with CARC {code}" + (f": {reason}" if reason else "")
                return [_sentence(f"Claim {claim_id} was denied{detail}", [index])]
            return [
                _sentence(
                    f"Claim {claim_id} was not denied; the paid amount was "
                    f"${self._parse_claim_amount(chunk, 'paid'):,.2f}",
                    [index],
                )
            ]

        if "code" in q and ("denial" in q or "drove" in q):
            claim = self._claim(contexts, claim_id)
            if claim is None:
                return None
            index, chunk = claim
            cpt_code, _ = self._parse_primary_cpt(chunk)
            carc = str(chunk.metadata.get("denial_code") or "")
            if not carc:
                return [_sentence(f"Claim {claim_id} was not denied", [index])]
            return [
                _sentence(
                    f"The denial reason code was {carc} and the primary procedure code was CPT {cpt_code}",
                    [index],
                )
            ]

        if "how many" in q:
            if "denied" in q and "claim" in q:
                if not claim_facts:
                    return None
                denied = [(i, c) for i, c in claim_facts if c.metadata.get("denied")]
                count = len({c.metadata.get("claim_id") or c.chunk_id for _, c in denied})
                cites = [i for i, _ in (denied or claim_facts)]
                return [
                    self._count_sentence(count, "denied claim", "denied claims", patient, cites)
                ]
            if "claim" in q:
                if not claim_facts:
                    return None
                count = len({c.metadata.get("claim_id") or c.chunk_id for _, c in claim_facts})
                return [
                    self._count_sentence(
                        count, "claim", "claims", patient, [i for i, _ in claim_facts]
                    )
                ]
            if "encounter" in q:
                if not encounter_facts:
                    return None
                count = len(
                    {c.metadata.get("encounter_id") or c.chunk_id for _, c in encounter_facts}
                )
                return [
                    self._count_sentence(
                        count,
                        "encounter",
                        "encounters",
                        patient,
                        [i for i, _ in encounter_facts],
                    )
                ]
            if "procedure" in q:
                if not procedure_facts:
                    return None
                count = len(procedure_facts)
                return [
                    self._count_sentence(
                        count,
                        "procedure",
                        "procedures",
                        patient,
                        [i for i, _ in procedure_facts],
                    )
                ]
            if "diagnos" in q:
                if not condition_facts:
                    return None
                count = len(condition_facts)
                return [
                    self._count_sentence(
                        count,
                        "diagnosis",
                        "diagnoses",
                        patient,
                        [i for i, _ in condition_facts],
                    )
                ]
            if "medication" in q:
                if not medication_facts:
                    return None
                count = len(medication_facts)
                return [
                    self._count_sentence(
                        count,
                        "medication order",
                        "medication orders",
                        patient,
                        [i for i, _ in medication_facts],
                    )
                ]
            return None

        if "total" in q and ("billed" in q or "allowed" in q or "paid" in q):
            if not claim_facts:
                return None
            if "allowed" in q:
                key, label = "allowed", "allowed"
            elif "paid" in q:
                key, label = "paid", "paid"
            else:
                key, label = "billed", "billed"
            total = sum(self._parse_claim_amount(c, key) for _, c in claim_facts)
            count = len(claim_facts)
            claim_label = "claim" if count == 1 else "claims"
            return [
                _sentence(
                    f"The total {label} amount for patient {patient} is ${total:,.2f} "
                    f"across {count} {claim_label}",
                    [i for i, _ in claim_facts],
                )
            ]

        if "date of service" in q and claim_id:
            claim = self._claim(contexts, claim_id)
            if claim is None:
                return None
            index, chunk = claim
            return [
                _sentence(
                    f"The date of service for claim {claim_id} was {chunk.metadata.get('service_date')}",
                    [index],
                )
            ]

        if "list" in q and "denial" in q:
            if not claim_facts:
                return None
            codes = sorted(
                {
                    str(c.metadata.get("denial_code"))
                    for _, c in claim_facts
                    if c.metadata.get("denial_code")
                }
            )
            if not codes:
                return [
                    _sentence(
                        f"No denial reason codes were recorded for patient {patient}",
                        [i for i, _ in claim_facts],
                    )
                ]
            return [
                _sentence(
                    f"The denial reason codes for patient {patient} were {', '.join(codes)}",
                    [i for i, chunk in claim_facts if chunk.metadata.get("denial_code")],
                )
            ]

        # --- treatment / clinical facts ---------------------------------
        if "diagnos" in q:
            if not condition_facts:
                return None
            if "primary" in q or "main" in q:
                index, chunk = sorted(
                    condition_facts, key=lambda item: str(item[1].metadata.get("code") or "")
                )[0]
                code = str(chunk.metadata.get("code") or "")
                display = str(chunk.metadata.get("code_display") or "")
                return [
                    _sentence(
                        f"The primary diagnosis was {display} (ICD-10 {code})",
                        [index],
                    )
                ]
            return [
                _sentence(
                    f"{chunk.metadata.get('code_display')} was documented "
                    f"(ICD-10 {chunk.metadata.get('code')})",
                    [index],
                )
                for index, chunk in condition_facts
            ]

        if "procedure" in q:
            if not procedure_facts:
                return None
            return [
                _sentence(
                    f"{chunk.metadata.get('code_display')} was performed "
                    f"(CPT {chunk.metadata.get('code')})",
                    [index],
                )
                for index, chunk in procedure_facts
            ]

        if "medication" in q:
            if not medication_facts:
                return None
            return [
                _sentence(
                    f"{chunk.metadata.get('code_display')} was ordered "
                    f"(RxNorm {chunk.metadata.get('code')})",
                    [index],
                )
                for index, chunk in medication_facts
            ]

        if observation_facts and ("what was the" in q or "what is the" in q or "most recent" in q):
            question_terms = {
                token
                for token in re.findall(r"[a-z]+", q)
                if token
                not in {
                    "what",
                    "was",
                    "the",
                    "is",
                    "for",
                    "patient",
                    "most",
                    "recent",
                    "recorded",
                    "value",
                    "of",
                    "a",
                    "an",
                    "on",
                    "during",
                    "at",
                }
            }
            matches: list[tuple[int, RetrievedChunk, float]] = []
            for index, chunk in observation_facts:
                display = str(chunk.metadata.get("code_display") or "").lower()
                display_terms = set(re.findall(r"[a-z]+", display))
                overlap = len(question_terms & display_terms)
                if overlap:
                    matches.append((index, chunk, overlap))
            if not matches:
                return None
            best_overlap = max(match[2] for match in matches)
            best = [match for match in matches if match[2] == best_overlap]
            best.sort(
                key=lambda item: str(item[1].metadata.get("service_date") or ""), reverse=True
            )
            index, chunk, _ = best[0]
            display, value, date = self._parse_observation(chunk)
            if not value:
                return None
            code = str(chunk.metadata.get("code") or "")
            return [_sentence(f"{display} was {value} on {date} (SNOMED {code})", [index])]

        if ("summarize" in q or "summary" in q or "history" in q) and contexts:
            clauses: list[str] = []
            for index, chunk in enumerate(contexts[:3], start=1):
                snippet = chunk.text.split(". ")[0].strip()
                if snippet:
                    clauses.append(_sentence(snippet, [index]))
            return clauses or None

        # Generic fallback: quote the best overlapping sentence from a retrieved chunk.
        question_tokens = set(re.findall(r"[a-z0-9]{2,}", q))
        ranked: list[tuple[int, float, str]] = []
        for index, chunk in enumerate(contexts, start=1):
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                sentence_tokens = set(re.findall(r"[a-z0-9]{2,}", sentence.lower()))
                overlap = len(question_tokens & sentence_tokens)
                ranked.append((index, float(overlap), sentence.strip()))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if ranked and ranked[0][1] > 0:
            index, _, sentence = ranked[0]
            return [_sentence(sentence, [index])]
        return None


class HFLocalGenerator:
    """Small local instruct generator (Qwen2.5-0.5B-Instruct, flan-t5-base fallback)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._is_seq2seq = False
        self.selection_reason = ""
        self.name = ""
        self._load()

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        last_error: Exception | None = None
        for model_name in (self.settings.llm_primary_model, self.settings.llm_fallback_model):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if "t5" in model_name.lower():
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    self._is_seq2seq = True
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    self._is_seq2seq = False
                model.eval()
                self._model = model
                self._tokenizer = tokenizer
                self.name = f"hf-local:{model_name}"
                self.selection_reason = f"loaded {model_name} on CPU"
                return
            except Exception as exc:  # noqa: BLE001 - try the configured fallback
                last_error = exc
                logger.warning("Could not load %s: %s", model_name, exc)
        raise RuntimeError(f"no local HF model could be loaded: {last_error}")

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        filters: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Run a local instruct model with a strictly grounded prompt."""
        import torch

        assert self._model is not None and self._tokenizer is not None
        prompt = build_prompt(question, contexts)
        max_tokens = max_new_tokens or self.settings.max_new_tokens
        if self._is_seq2seq:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=2
                )
            text = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
            input_tokens = int(inputs["input_ids"].shape[1])
            output_tokens = int(output.shape[1])
        else:
            messages = [
                {
                    "role": "system",
                    "content": "Answer only from the supplied clinical context and cite every sentence.",
                },
                {"role": "user", "content": prompt},
            ]
            if hasattr(self._tokenizer, "apply_chat_template"):
                inputs = self._tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )
                input_ids = inputs
            else:
                input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                output = self._model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = output[0][input_ids.shape[1] :]
            text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
            input_tokens = int(input_ids.shape[1])
            output_tokens = int(generated.shape[0])
        return GenerationResult(
            text=text,
            backend_name=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            est_cost_usd=0.0,
            insufficient_evidence=text.startswith("Insufficient evidence"),
        )


class OpenAIGenerator:
    """Commercial API baseline; only constructible when OPENAI_API_KEY is set."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.settings = settings
        self.name = f"openai:{settings.openai_model}"
        self.selection_reason = "OPENAI_API_KEY present"

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        filters: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Call the OpenAI chat completions API with the grounded prompt."""
        prompt = build_prompt(question, contexts)
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.openai_model,
                "messages": [
                    {"role": "system", "content": "Answer only from the supplied context."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_new_tokens or self.settings.max_new_tokens,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"].strip()
        usage = payload.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", count_tokens(prompt)))
        output_tokens = int(usage.get("completion_tokens", count_tokens(text)))
        return GenerationResult(
            text=text,
            backend_name=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            est_cost_usd=estimate_cost_usd(
                input_tokens,
                output_tokens,
                self.settings.cost_per_1k_input_tokens_usd,
                self.settings.cost_per_1k_output_tokens_usd,
            ),
            insufficient_evidence=text.startswith("Insufficient evidence"),
        )


def get_generator(settings: Settings) -> Generator:
    """Resolve the generator backend and record exactly which one ran.

    ``auto`` uses the deterministic extractive backend so every offline build
    is reproducible. Set ``LLM_BACKEND=hf-local`` (or ``openai``) to opt into a
    model-backed generator; ``CLINICAL_RAG_ENABLE_HF=1`` enables HF under auto.
    """
    choice = settings.llm_backend
    if choice == "hf-local" or (
        choice == "auto" and os.environ.get("CLINICAL_RAG_ENABLE_HF") == "1"
    ):
        try:
            return HFLocalGenerator(settings)
        except Exception as exc:  # noqa: BLE001 - the offline fallback is the safety net
            if choice == "hf-local":
                raise
            logger.warning("HF generator unavailable (%s); using extractive fallback", exc)
            return ExtractiveGenerator()
    if choice == "openai":
        return OpenAIGenerator(settings)
    if (
        choice == "auto"
        and settings.openai_api_key
        and os.environ.get("CLINICAL_RAG_USE_OPENAI") == "1"
    ):
        return OpenAIGenerator(settings)
    return ExtractiveGenerator()


__all__ = [
    "ABSTENTION_TEXT",
    "ExtractiveGenerator",
    "GenerationResult",
    "Generator",
    "HFLocalGenerator",
    "OpenAIGenerator",
    "build_prompt",
    "get_generator",
]
