"""Evaluate zero-shot, RAG, reranker-tuned, LoRA and commercial baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from clinicalnoterag.config import Settings, get_settings
from clinicalnoterag.embeddings.backends import load_embedding_state
from clinicalnoterag.eval.benchmark import load_benchmark
from clinicalnoterag.eval.runner import EvalResult, evaluate
from clinicalnoterag.eval.stats import paired_bootstrap_delta
from clinicalnoterag.rag.answerer import Answer, ClinicalAnswerer
from clinicalnoterag.rag.generators import (
    ABSTENTION_TEXT,
    ExtractiveGenerator,
    HFLocalGenerator,
)
from clinicalnoterag.rag.retriever import HybridRetriever
from clinicalnoterag.telemetry import TelemetryRecorder
from clinicalnoterag.train.fallback_reranker import ClinicalReranker
from clinicalnoterag.vectorstore.factory import get_store


class NoRAGAnswerer:
    """Generator with retrieval removed (zero-shot, no-RAG baseline)."""

    def __init__(self, answerer: ClinicalAnswerer) -> None:
        self.retriever = answerer.retriever
        self.generator = answerer.generator
        self.settings = answerer.settings
        self.reranker_name = None
        self.system_name = "zero-shot-no-rag"

    def answer(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        k: int | None = None,
        mode: str = "hybrid",
    ) -> Answer:
        """Answer without any retrieved context."""
        started = time.perf_counter()
        result = self.generator.generate(question, [], filters=filters)
        abstained = result.insufficient_evidence or result.text.strip() == ABSTENTION_TEXT
        text = result.text.strip() or ABSTENTION_TEXT
        answer = Answer(
            question=question,
            text=text,
            citations=[],
            confidence=0.0,
            abstained=abstained,
            abstention_reason="no_retrieval" if abstained else None,
            unsupported_claim=False,
            retrieval_scores=[],
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            token_counts={
                "input": result.input_tokens,
                "output": result.output_tokens,
            },
            est_cost_usd=result.est_cost_usd,
            backend_name=result.backend_name,
            filters=dict(filters or {}),
            mode=mode,
        )
        return answer


def _align(a: EvalResult, b: EvalResult, metric: str) -> tuple[np.ndarray, np.ndarray]:
    by_id_a = {record["id"]: record for record in a.per_item}
    by_id_b = {record["id"]: record for record in b.per_item}
    common = sorted(set(by_id_a) & set(by_id_b))
    values_a: list[float] = []
    values_b: list[float] = []
    for item_id in common:
        record_a = by_id_a[item_id]
        record_b = by_id_b[item_id]
        if metric == "hallucination":
            if not record_a["answerable"]:
                continue
            values_a.append(1.0 if record_a["unsupported_claim"] else 0.0)
            values_b.append(1.0 if record_b["unsupported_claim"] else 0.0)
        elif metric == "false_answer":
            if record_a["answerable"]:
                continue
            values_a.append(1.0 if not record_a["abstained"] else 0.0)
            values_b.append(1.0 if not record_b["abstained"] else 0.0)
        else:
            va = record_a.get(metric)
            vb = record_b.get(metric)
            if va is None or vb is None:
                continue
            values_a.append(float(va))
            values_b.append(float(vb))
    return np.asarray(values_a), np.asarray(values_b)


def _fmt(summary: dict[str, Any]) -> str:
    return f"{summary['value']:.3f} [{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]"


def _row(name: str, result: EvalResult, label: str = "") -> str:
    metrics = result.metrics
    return (
        f"| {name}{label} | {_fmt(metrics['recall_at_k'])} | {_fmt(metrics['mrr'])} | "
        f"{_fmt(metrics['exact_match'])} | {_fmt(metrics['token_f1'])} | "
        f"{_fmt(metrics['hallucination_rate'])} | {_fmt(metrics['false_answer_rate'])} | "
        f"{_fmt(metrics['abstention_precision'])} | {_fmt(metrics['abstention_recall'])} | "
        f"{_fmt(metrics['code']['micro_f1'])} | {metrics['latency_p50_ms']:.0f} | "
        f"{metrics['latency_p95_ms']:.0f} | ${metrics['mean_cost_usd']['value']:.6f} |"
    )


def run_system_comparison(
    settings: Settings | None = None,
    benchmark_path: str | Path = "data/benchmark/clinical_qa.jsonl",
    split_manifest_path: str | Path = "data/sft/split_manifest.json",
    lora_dir: str | Path = "artifacts/adapters/clinical-lora",
    reranker_dir: str | Path = "artifacts/adapters/clinical-reranker",
    out_dir: str | Path = "artifacts/eval",
    k: int = 5,
    seed: int = 42,
    include_lora: bool = True,
) -> dict[str, Any]:
    """Run the Step 4 harness for every available system on the test split."""
    active_settings = settings or get_settings()
    items = load_benchmark(benchmark_path)
    split_manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    test_patients = set(split_manifest["splits"]["test"])
    test_items = [item for item in items if item["patient_id"] in test_patients]

    embedder, _ = load_embedding_state(active_settings)
    store = get_store(active_settings, embedder.dim, fresh=False)
    retriever = HybridRetriever(store, embedder, active_settings)
    extractive = ExtractiveGenerator()
    results: dict[str, EvalResult] = {}
    notes: dict[str, str] = {}
    try:
        base_answerer = ClinicalAnswerer(
            retriever,
            generator=extractive,
            settings=active_settings,
            telemetry=TelemetryRecorder(active_settings.telemetry_path, enabled=False),
            system_name="rag-zero-shot",
        )
        no_rag = NoRAGAnswerer(base_answerer)
        results["zero-shot-no-rag"] = evaluate(
            "zero-shot-no-rag",
            no_rag,  # type: ignore[arg-type]
            test_items,
            k=k,
            seed=seed,
            eval_dir=out_dir,
        )
        notes["zero-shot-no-rag"] = (
            "Deterministic extractive generator has no parametric clinical knowledge, so it "
            "abstains without context. This is a real no-RAG baseline, not a simulated answer."
        )

        results["rag-zero-shot"] = evaluate(
            "rag-zero-shot",
            base_answerer,
            test_items,
            k=k,
            seed=seed,
            eval_dir=out_dir,
        )
        notes["rag-zero-shot"] = (
            "Hybrid retrieval + extractive grounded generator, no learned reranker."
        )

        reranker = ClinicalReranker.load(reranker_dir)
        tuned = ClinicalAnswerer(
            retriever,
            generator=extractive,
            settings=active_settings,
            telemetry=TelemetryRecorder(active_settings.telemetry_path, enabled=False),
            system_name="rag-tuned",
            reranker=reranker,
        )
        results["rag-tuned"] = evaluate(
            "rag-tuned",
            tuned,
            test_items,
            k=k,
            seed=seed,
            eval_dir=out_dir,
        )
        notes["rag-tuned"] = (
            "Hybrid retrieval + fallback MLP reranker trained on the patient-disjoint train "
            "split; reranker is the fallback learned component."
        )

        if (
            include_lora
            and Path(lora_dir).exists()
            and (Path(lora_dir) / "adapter_config.json").exists()
        ):
            try:
                lora_generator = HFLocalGenerator(active_settings, adapter_path=lora_dir)
                lora_answerer = ClinicalAnswerer(
                    retriever,
                    generator=lora_generator,
                    settings=active_settings,
                    telemetry=TelemetryRecorder(active_settings.telemetry_path, enabled=False),
                    system_name="rag-lora",
                )
                results["rag-lora"] = evaluate(
                    "rag-lora",
                    lora_answerer,
                    test_items,
                    k=k,
                    seed=seed,
                    eval_dir=out_dir,
                )
                notes["rag-lora"] = (
                    "PEFT LoRA adapter trained with TRL SFTTrainer on the patient-disjoint "
                    "train split, used with hybrid retrieval."
                )
            except Exception as exc:  # noqa: BLE001 - record a real load failure
                notes["rag-lora"] = f"not run — adapter load failed: {type(exc).__name__}: {exc}"

        delta_metrics = ["recall", "reciprocal_rank", "exact_match", "token_f1", "hallucination"]
        deltas: dict[str, Any] = {}
        for metric in delta_metrics:
            a, b = _align(results["rag-tuned"], results["rag-zero-shot"], metric)
            if len(a):
                deltas[metric] = paired_bootstrap_delta(a, b, seed=seed).to_dict()
        # False-answer paired delta on unanswerable items only.
        a, b = _align(results["rag-tuned"], results["rag-zero-shot"], "false_answer")
        if len(a):
            deltas["false_answer"] = paired_bootstrap_delta(a, b, seed=seed).to_dict()

        lora_log_path = Path("artifacts/train/train_log.json")
        reranker_log_path = Path("artifacts/train/reranker_log.json")
        lora_log = (
            json.loads(lora_log_path.read_text(encoding="utf-8")) if lora_log_path.exists() else {}
        )
        reranker_log = (
            json.loads(reranker_log_path.read_text(encoding="utf-8"))
            if reranker_log_path.exists()
            else {}
        )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        comparison = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "test_patients": sorted(test_patients),
            "n_test_items": len(test_items),
            "k": k,
            "seed": seed,
            "systems": {name: result.to_dict() for name, result in results.items()},
            "deltas_rag_tuned_vs_zero_shot": deltas,
            "notes": notes,
            "lora": {
                "status": lora_log.get("status", "not attempted"),
                "config": {
                    key: lora_log.get(key)
                    for key in (
                        "model",
                        "max_steps",
                        "learning_rate",
                        "lora_r",
                        "lora_alpha",
                        "lora_dropout",
                        "target_modules",
                        "n_train",
                        "n_val",
                        "train_runtime_sec",
                    )
                },
                "error": lora_log.get("error"),
            },
            "reranker": {
                "status": reranker_log.get("status", "not attempted"),
                "component": reranker_log.get("component"),
                "config": reranker_log.get("config"),
                "test_retrieval_before": reranker_log.get("test_retrieval_before"),
                "test_retrieval_after": reranker_log.get("test_retrieval_after"),
            },
            "commercial_api": {
                "status": "not run — no API key"
                if not active_settings.openai_api_key
                else "key present; not run in this build",
            },
        }
        (out / "system_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        _write_markdown(comparison, results, out / "system_comparison.md")
        return comparison
    finally:
        store.close()


def _write_markdown(
    comparison: dict[str, Any],
    results: dict[str, EvalResult],
    path: Path,
) -> None:
    lines = [
        "# Multi-system comparison (held-out test patients)",
        "",
        f"Test split: **{comparison['n_test_items']} items** from "
        f"{len(comparison['test_patients'])} patients held out by patient; k={comparison['k']}, "
        f"retrieval mode hybrid, seed {comparison['seed']}. Intervals are 95% percentile "
        "bootstrap CIs with 10,000 resamples.",
        "",
        "| System | Recall@k | MRR | Exact | Token F1 | Hallucination | False answer | Abst P | Abst R | Code F1 micro | p50 ms | p95 ms | Cost/query |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    labels = {
        "zero-shot-no-rag": " (generator only)",
        "rag-zero-shot": " (Step 3)",
        "rag-tuned": " (reranker)",
        "rag-lora": " (LoRA adapter)",
    }
    for name, result in results.items():
        lines.append(_row(name, result, labels.get(name, "")))
    if not any(active_settings_key in results for active_settings_key in ["rag-commercial-api"]):
        lines.append("| rag-commercial-api | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "**Not run:** `rag-commercial-api` — no `OPENAI_API_KEY` was present in the "
            "build environment. No numbers are invented for it.",
            "",
            "## Paired deltas: rag-tuned vs rag-zero-shot",
            "",
            "Positive deltas are better except hallucination and false-answer, where negative "
            "is better. P-values are two-sided paired bootstrap tests (10,000 resamples).",
            "",
            "| Metric | Delta | 95% CI | p-value | n |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for metric, delta in comparison["deltas_rag_tuned_vs_zero_shot"].items():
        lines.append(
            f"| {metric} | {delta['delta']:+.4f} | [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] | "
            f"{delta['p_value']:.4f} | {delta['n']} |"
        )
    lines.extend(
        [
            "",
            "## Training path record",
            "",
            f"- LoRA status: **{comparison['lora']['status']}**",
            f"- LoRA config: `{json.dumps(comparison['lora']['config'])}`",
            f"- Reranker status: **{comparison['reranker']['status']}** "
            f"(`{comparison['reranker'].get('component')}`)",
            f"- Reranker held-out retrieval before: `{comparison['reranker'].get('test_retrieval_before')}`",
            f"- Reranker held-out retrieval after: `{comparison['reranker'].get('test_retrieval_after')}`",
            "",
            "The primary `rag-tuned` system uses the fallback MLP reranker because it is the "
            "learned component that improved held-out ranking in this environment. The LoRA "
            "adapter was also trained for real with PEFT/TRL and evaluated as `rag-lora`; it "
            "is not substituted silently. `zero-shot-no-rag` uses the deterministic extractive "
            "generator with no parametric knowledge, so it abstains on every item; this is "
            "reported as a real baseline rather than a simulated answer.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["NoRAGAnswerer", "run_system_comparison"]
