#!/usr/bin/env python
"""Fire real queries at the local FastAPI service and write service metrics.

The script can start uvicorn itself, waits for /healthz, sends treatment,
payment, operations and adversarial questions, and derives p50/p95 latency,
abstention rate and mean cost from the responses and the API telemetry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from clinicalnoterag.telemetry import load_telemetry, telemetry_summary

REPO = Path(__file__).resolve().parents[1]


def load_bundles(corpus: Path, limit: int | None = None) -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []
    for path in sorted((corpus / "bundles").glob("*.json")):
        bundles.append(json.loads(path.read_text(encoding="utf-8")))
        if limit and len(bundles) >= limit:
            break
    return bundles


def resource(entry: dict[str, Any]) -> dict[str, Any]:
    return entry["resource"]


def resources(bundle: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    return [resource(e) for e in bundle["entry"] if resource(e)["resourceType"] == resource_type]


def build_queries(corpus: Path) -> list[dict[str, Any]]:
    bundles = load_bundles(corpus)
    queries: list[dict[str, Any]] = []
    treatment_seen = 0
    for bundle in bundles:
        patient = resources(bundle, "Patient")[0]["id"]
        encounters = resources(bundle, "Encounter")
        conditions = resources(bundle, "Condition")
        procedures = resources(bundle, "Procedure")
        observations = resources(bundle, "Observation")
        medications = resources(bundle, "MedicationRequest")
        claims = resources(bundle, "Claim")
        responses = {
            r["request"]["reference"].split("/")[-1]: r for r in resources(bundle, "ClaimResponse")
        }
        for encounter in encounters:
            if treatment_seen >= 12:
                break
            enc_id = encounter["id"]
            if any(c["encounter"]["reference"].endswith(enc_id) for c in conditions):
                queries.append(
                    {
                        "query_type": "treatment",
                        "question": f"What diagnoses were documented for encounter {enc_id}?",
                        "filters": {
                            "patient_id": patient,
                            "encounter_id": enc_id,
                            "resource_type": "Condition",
                            "doc_type": "fhir_fact",
                        },
                        "k": 5,
                    }
                )
            if any(p["encounter"]["reference"].endswith(enc_id) for p in procedures):
                queries.append(
                    {
                        "query_type": "treatment",
                        "question": f"What procedures were performed for encounter {enc_id}?",
                        "filters": {
                            "patient_id": patient,
                            "encounter_id": enc_id,
                            "resource_type": "Procedure",
                            "doc_type": "fhir_fact",
                        },
                        "k": 10,
                    }
                )
            enc_obs = [o for o in observations if o["encounter"]["reference"].endswith(enc_id)]
            if enc_obs:
                display = enc_obs[0]["code"]["coding"][0]["display"]
                queries.append(
                    {
                        "query_type": "treatment",
                        "question": f"What was the most recent {display} recorded for patient {patient}?",
                        "filters": {
                            "patient_id": patient,
                            "resource_type": "Observation",
                            "doc_type": "fhir_fact",
                        },
                        "k": 10,
                    }
                )
            enc_meds = [m for m in medications if m["encounter"]["reference"].endswith(enc_id)]
            if enc_meds:
                queries.append(
                    {
                        "query_type": "treatment",
                        "question": f"What medications were ordered for encounter {enc_id}?",
                        "filters": {
                            "patient_id": patient,
                            "encounter_id": enc_id,
                            "resource_type": "MedicationRequest",
                            "doc_type": "fhir_fact",
                        },
                        "k": 10,
                    }
                )
            treatment_seen += 1
        # payment queries from denied claims
        for claim in claims:
            response = responses.get(claim["id"])
            if response is None:
                continue
            denied = any(
                a["category"]["coding"][0]["code"] == "denial"
                for a in response.get("adjudication", [])
            )
            if not denied:
                continue
            if sum(1 for q in queries if q["query_type"] == "payment") >= 16:
                break
            claim_id = claim["id"]
            queries.append(
                {
                    "query_type": "payment",
                    "question": f"Why was claim {claim_id} denied and what supports an appeal?",
                    "filters": {"patient_id": patient},
                    "k": 10,
                }
            )
            queries.append(
                {
                    "query_type": "payment",
                    "question": f"Which code drove the denial for claim {claim_id}?",
                    "filters": {"patient_id": patient},
                    "k": 10,
                }
            )
        if sum(1 for q in queries if q["query_type"] == "payment") >= 16:
            break

    operations_seen = 0
    for bundle in bundles:
        if operations_seen >= 10:
            break
        patient = resources(bundle, "Patient")[0]["id"]
        if not resources(bundle, "Encounter"):
            continue
        operations_seen += 1
        base = {"patient_id": patient, "doc_type": "fhir_fact"}
        queries.append(
            {
                "query_type": "operations",
                "question": f"How many encounters did patient {patient} have?",
                "filters": {**base, "resource_type": "Encounter"},
                "k": 10,
            }
        )
        queries.append(
            {
                "query_type": "operations",
                "question": f"What is the total billed amount for patient {patient}?",
                "filters": {**base, "resource_type": "Claim"},
                "k": 10,
            }
        )
        queries.append(
            {
                "query_type": "operations",
                "question": f"How many claims for patient {patient} were denied?",
                "filters": {**base, "resource_type": "Claim"},
                "k": 10,
            }
        )

    adversarial = bundles[:2]
    for bundle in adversarial:
        patient = resources(bundle, "Patient")[0]["id"]
        num = patient.split("-")[-1]
        queries.extend(
            [
                {
                    "query_type": "adversarial",
                    "question": f"What diagnosis code Z99.999 was documented for patient {patient}?",
                    "filters": {
                        "patient_id": patient,
                        "resource_type": "Condition",
                        "doc_type": "fhir_fact",
                    },
                    "k": 5,
                },
                {
                    "query_type": "adversarial",
                    "question": f"Why was claim CLM-{num}-99 denied?",
                    "filters": {"patient_id": patient},
                    "k": 10,
                },
                {
                    "query_type": "adversarial",
                    "question": f"What was the LDL cholesterol recorded for patient {patient}?",
                    "filters": {
                        "patient_id": patient,
                        "resource_type": "Observation",
                        "doc_type": "fhir_fact",
                    },
                    "k": 10,
                },
                {
                    "query_type": "adversarial",
                    "question": f"What procedure with CPT 99999 was performed for patient {patient}?",
                    "filters": {
                        "patient_id": patient,
                        "resource_type": "Procedure",
                        "doc_type": "fhir_fact",
                    },
                    "k": 10,
                },
            ]
        )
    return queries


def wait_for_health(base_url: str, timeout: float = 180.0) -> dict[str, Any]:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            response = httpx.get(f"{base_url}/healthz", timeout=3.0)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"API did not become healthy at {base_url} within {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="artifacts/smoke_responses.jsonl")
    parser.add_argument("--metrics-out", default="artifacts/service_metrics.json")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reset-telemetry", action="store_true")
    args = parser.parse_args()

    telemetry_path = REPO / "artifacts" / "query_telemetry.jsonl"
    if args.reset_telemetry:
        telemetry_path.unlink(missing_ok=True)

    process: subprocess.Popen | None = None
    base_url = args.base_url
    if args.start_server:
        base_url = f"http://127.0.0.1:{args.port}"
        env = os.environ.copy()
        env.setdefault("LLM_BACKEND", "extractive")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "clinicalnoterag.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--log-level",
                "warning",
            ],
            cwd=REPO,
            env=env,
        )
    try:
        health = wait_for_health(base_url)
        queries = build_queries(Path(args.corpus))
        print(f"health={json.dumps(health)}")
        print(f"firing {len(queries)} real queries at {base_url}")
        results: list[dict[str, Any]] = []
        with httpx.Client(timeout=120.0) as client:
            for index, query in enumerate(queries, start=1):
                started = time.perf_counter()
                try:
                    response = client.post(
                        f"{base_url}/ask",
                        json={
                            "question": query["question"],
                            "filters": query["filters"],
                            "k": query["k"],
                        },
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    body = (
                        response.json() if response.status_code == 200 else {"error": response.text}
                    )
                    results.append(
                        {
                            **query,
                            "status_code": response.status_code,
                            "client_latency_ms": round(elapsed_ms, 3),
                            "response": body,
                        }
                    )
                except httpx.HTTPError as exc:
                    results.append(
                        {
                            **query,
                            "status_code": None,
                            "client_latency_ms": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if index % 20 == 0:
                    print(f"  {index}/{len(queries)} queries complete", flush=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for result in results:
                fh.write(json.dumps(result) + "\n")

        ok = [r for r in results if r.get("status_code") == 200]
        latencies = [float(r["response"]["latency_ms"]) for r in ok]
        client_latencies = [float(r["client_latency_ms"]) for r in ok]
        by_type: dict[str, Any] = {}
        for result in results:
            bucket = by_type.setdefault(
                result["query_type"], {"n": 0, "abstained": 0, "latency_ms": []}
            )
            bucket["n"] += 1
            if result.get("status_code") == 200:
                bucket["abstained"] += int(bool(result["response"].get("abstained")))
                bucket["latency_ms"].append(float(result["response"].get("latency_ms", 0.0)))
        for bucket in by_type.values():
            values = bucket.pop("latency_ms")
            bucket["abstention_rate"] = (
                round(bucket["abstained"] / bucket["n"], 6) if bucket["n"] else 0.0
            )
            bucket["mean_latency_ms"] = round(float(np.mean(values)), 3) if values else 0.0

        telemetry_events = load_telemetry(telemetry_path, system="api")
        summary = telemetry_summary(telemetry_events)
        metrics = {
            "n_queries": len(results),
            "n_successful": len(ok),
            "n_failed": len(results) - len(ok),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3) if latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
            "client_p50_latency_ms": round(float(np.percentile(client_latencies, 50)), 3)
            if client_latencies
            else 0.0,
            "client_p95_latency_ms": round(float(np.percentile(client_latencies, 95)), 3)
            if client_latencies
            else 0.0,
            "abstention_rate": round(
                sum(1 for r in ok if r["response"].get("abstained")) / len(ok), 6
            )
            if ok
            else 0.0,
            "mean_cost_usd": round(
                float(np.mean([float(r["response"].get("est_cost_usd", 0.0)) for r in ok])), 8
            )
            if ok
            else 0.0,
            "unsupported_claim_rate": round(
                sum(1 for r in ok if r["response"].get("unsupported_claim")) / len(ok), 6
            )
            if ok
            else 0.0,
            "mean_citations": round(
                float(np.mean([len(r["response"].get("citations", [])) for r in ok])), 3
            )
            if ok
            else 0.0,
            "by_query_type": by_type,
            "telemetry_summary": summary,
            "telemetry_events_for_api": len(telemetry_events),
            "backends": {
                "generator": ok[0]["response"].get("backend_name") if ok else None,
                "store": health.get("store_backend"),
                "embedding": health.get("embedding_backend"),
                "deid": health.get("deid_backend"),
            },
            "health": health,
            "note": "Measured by scripts/smoke_queries.py against a live uvicorn API.",
        }
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return 0 if len(ok) == len(results) else 1
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
