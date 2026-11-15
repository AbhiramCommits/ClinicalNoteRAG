#!/usr/bin/env python
"""Offline end-to-end smoke pipeline used by `make smoke` and CI.

Generates 20 patients, de-identifies/chunks/embeds with the deterministic
hashing backend, ingests into FAISS, asks five real questions, and writes a
tiny evaluation artifact. No network, model download, Docker or Postgres.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("LLM_BACKEND", "extractive")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from clinicalnoterag.config import get_settings  # noqa: E402
from clinicalnoterag.data.synthea_gen import generate_corpus  # noqa: E402
from clinicalnoterag.deid.engine import DeidEngine  # noqa: E402
from clinicalnoterag.embeddings.backends import get_embedding_backend  # noqa: E402
from clinicalnoterag.eval.metrics import exact_match  # noqa: E402
from clinicalnoterag.pipeline.ingest import run_ingest  # noqa: E402
from clinicalnoterag.rag.answerer import ClinicalAnswerer  # noqa: E402
from clinicalnoterag.rag.retriever import HybridRetriever  # noqa: E402
from clinicalnoterag.seeds import set_global_seed  # noqa: E402
from clinicalnoterag.telemetry import TelemetryRecorder  # noqa: E402
from clinicalnoterag.vectorstore.faiss_store import FaissStore  # noqa: E402


def resources(bundle: dict, resource_type: str) -> list[dict]:
    return [
        entry["resource"]
        for entry in bundle["entry"]
        if entry["resource"]["resourceType"] == resource_type
    ]


def main() -> int:
    set_global_seed(42)
    corpus = Path("data/smoke_corpus")
    artifacts = Path("artifacts/smoke")
    artifacts.mkdir(parents=True, exist_ok=True)
    stats = generate_corpus(20, corpus, seed=42)
    settings = get_settings(
        embedding_backend="hashing",
        llm_backend="extractive",
        artifact_dir=artifacts,
        faiss_index_path=artifacts / "index" / "smoke.faiss",
        sqlite_path=artifacts / "index" / "smoke.sqlite3",
        telemetry_path=artifacts / "telemetry.jsonl",
        seed=42,
    )
    manifest = json.loads(Path(stats.manifest_path).read_text(encoding="utf-8"))
    bundles = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((corpus / "bundles").glob("*.json"))
    ]
    embedder = get_embedding_backend(settings)
    deid = DeidEngine(backend="regex", known_names=manifest["known_names"])
    store = FaissStore(
        dim=embedder.dim,
        index_path=settings.faiss_index_path,
        sqlite_path=settings.sqlite_path,
        fresh=True,
    )
    try:
        report = run_ingest(
            corpus,
            store,
            deid,
            embedder,
            settings,
            out_path=artifacts / "ingest_report.json",
            deid_map_dir=artifacts / "deid_maps",
        )
        retriever = HybridRetriever(store, embedder, settings)
        answerer = ClinicalAnswerer(
            retriever,
            settings=settings,
            telemetry=TelemetryRecorder(settings.telemetry_path, enabled=False),
            system_name="smoke",
        )

        patient = resources(bundles[0], "Patient")[0]
        patient_id = patient["id"]
        encounter = resources(bundles[0], "Encounter")[0]
        enc_id = encounter["id"]
        conditions = [
            r
            for r in resources(bundles[0], "Condition")
            if r["encounter"]["reference"].endswith(enc_id)
        ]
        procedures = [
            r
            for r in resources(bundles[0], "Procedure")
            if r["encounter"]["reference"].endswith(enc_id)
        ]
        denied = None
        for bundle in bundles:
            for response in resources(bundle, "ClaimResponse"):
                if any(
                    a["category"]["coding"][0]["code"] == "denial"
                    for a in response.get("adjudication", [])
                ):
                    claim_id = response["request"]["reference"].split("/")[-1]
                    denied = {
                        "patient_id": resources(bundle, "Patient")[0]["id"],
                        "claim_id": claim_id,
                    }
                    break
            if denied:
                break
        if not conditions or not procedures or not denied:
            raise RuntimeError("smoke corpus did not contain required evidence")

        gold_diagnosis = " ".join(
            f"{r['code']['coding'][0]['display']} was documented "
            f"(ICD-10 {r['code']['coding'][0]['code']})."
            for r in sorted(conditions, key=lambda r: r["code"]["coding"][0]["code"])
        )
        questions = [
            {
                "name": "diagnosis",
                "question": f"What diagnoses were documented for encounter {enc_id}?",
                "filters": {
                    "patient_id": patient_id,
                    "encounter_id": enc_id,
                    "resource_type": "Condition",
                    "doc_type": "fhir_fact",
                },
                "k": 5,
                "expect_abstain": False,
                "gold": gold_diagnosis,
            },
            {
                "name": "procedure",
                "question": f"What procedures were performed for encounter {enc_id}?",
                "filters": {
                    "patient_id": patient_id,
                    "encounter_id": enc_id,
                    "resource_type": "Procedure",
                    "doc_type": "fhir_fact",
                },
                "k": 5,
                "expect_abstain": False,
            },
            {
                "name": "payment",
                "question": (
                    f"Why was claim {denied['claim_id']} denied and what supports an appeal?"
                ),
                "filters": {
                    "patient_id": denied["patient_id"],
                    "claim_id": denied["claim_id"],
                    "doc_type": "fhir_fact",
                },
                "k": 5,
                "expect_abstain": False,
            },
            {
                "name": "operations",
                "question": f"How many encounters did patient {patient_id} have?",
                "filters": {
                    "patient_id": patient_id,
                    "resource_type": "Encounter",
                    "doc_type": "fhir_fact",
                },
                "k": 10,
                "expect_abstain": False,
            },
            {
                "name": "adversarial",
                "question": f"What diagnosis code Z99.999 was documented for patient {patient_id}?",
                "filters": {
                    "patient_id": patient_id,
                    "resource_type": "Condition",
                    "doc_type": "fhir_fact",
                },
                "k": 5,
                "expect_abstain": True,
            },
        ]
        results = []
        for item in questions:
            answer = answerer.answer(item["question"], filters=item["filters"], k=item["k"])
            passed = (
                answer.abstained is item["expect_abstain"]
                and (answer.abstained or bool(answer.citations))
                and not answer.unsupported_claim
            )
            record = {
                "name": item["name"],
                "question": item["question"],
                "abstained": answer.abstained,
                "abstention_reason": answer.abstention_reason,
                "text": answer.text,
                "n_citations": len(answer.citations),
                "unsupported_claim": answer.unsupported_claim,
                "latency_ms": answer.latency_ms,
                "passed": passed,
                "exact_match": (
                    exact_match(answer.text, item["gold"]) if item.get("gold") else None
                ),
            }
            results.append(record)
        tiny_eval = {
            "n_questions": len(results),
            "n_passed": sum(1 for r in results if r["passed"]),
            "pass_rate": round(sum(1 for r in results if r["passed"]) / len(results), 4),
            "citation_rate": round(
                sum(1 for r in results if r["abstained"] or r["n_citations"] > 0) / len(results), 4
            ),
            "abstention_on_adversarial": next(
                r["abstained"] for r in results if r["name"] == "adversarial"
            ),
            "diagnosis_exact_match": results[0]["exact_match"],
        }
        payload = {
            "ingest": {
                key: report[key]
                for key in (
                    "documents",
                    "chunks",
                    "store_backend",
                    "embedding_backend",
                    "deid_backend",
                )
            },
            "backend": "hashing-tfidf (offline)",
            "tiny_eval": tiny_eval,
            "questions": results,
        }
        (artifacts / "smoke_results.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload["tiny_eval"], indent=2))
        print(f"artifact: {artifacts / 'smoke_results.json'}")
        return 0 if tiny_eval["n_passed"] == tiny_eval["n_questions"] else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
