"""Evaluation runner, confidence intervals, leaderboard and ablation sweep."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from clinicalnoterag.eval.metrics import (
    code_counts,
    code_set,
    exact_match,
    numeric_match,
    predicted_codes,
    prf,
    reciprocal_rank,
    retrieval_recall,
    token_f1,
)
from clinicalnoterag.eval.stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_indices,
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    paired_bootstrap_delta,
)
from clinicalnoterag.rag.answerer import ClinicalAnswerer


def _summary(values: Iterable[float], n_resamples: int, seed: int) -> dict[str, Any]:
    return bootstrap_mean_ci(list(values), n_resamples=n_resamples, seed=seed)


def _ci_from_indices(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    samples = values[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def _abstention_summary(
    answerable: np.ndarray, abstained: np.ndarray, n_resamples: int, seed: int
) -> dict[str, Any]:
    truth = ~answerable
    predicted = abstained
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    idx = bootstrap_indices(len(truth), n_resamples, seed)
    t = (predicted & truth)[idx].sum(axis=1)
    p = (predicted & ~truth)[idx].sum(axis=1)
    n = ((~predicted) & truth)[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_samples = np.where(t + p > 0, t / (t + p), 0.0)
        r_samples = np.where(t + n > 0, t / (t + n), 0.0)
    p_low, p_high = np.percentile(p_samples, [2.5, 97.5])
    r_low, r_high = np.percentile(r_samples, [2.5, 97.5])
    return {
        "precision": {
            "value": round(precision, 6),
            "ci_low": round(float(p_low), 6),
            "ci_high": round(float(p_high), 6),
            "n": int(truth.size),
        },
        "recall": {
            "value": round(recall, 6),
            "ci_low": round(float(r_low), 6),
            "ci_high": round(float(r_high), 6),
            "n": int(truth.size),
        },
    }


def _code_summary(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    per_item_f1: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if tp.size == 0:
        zero = {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
        return {
            "micro_precision": zero,
            "micro_recall": zero.copy(),
            "micro_f1": zero.copy(),
            "macro_precision": zero.copy(),
            "macro_recall": zero.copy(),
            "macro_f1": zero.copy(),
        }
    micro_p = bootstrap_ratio_ci(tp, tp + fp, n_resamples, seed)
    micro_r = bootstrap_ratio_ci(tp, tp + fn, n_resamples, seed)
    idx = bootstrap_indices(tp.size, n_resamples, seed)
    t = tp[idx].sum(axis=1)
    p = fp[idx].sum(axis=1)
    n = fn[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_samples = np.where(2 * t + p + n > 0, (2 * t) / (2 * t + p + n), 0.0)
    point_tp = tp.sum()
    point_f1 = (
        (2 * point_tp) / (2 * point_tp + fp.sum() + fn.sum())
        if (2 * point_tp + fp.sum() + fn.sum())
        else 0.0
    )
    f1_low, f1_high = np.percentile(f1_samples, [2.5, 97.5])
    micro_f1 = {
        "value": round(float(point_f1), 6),
        "ci_low": round(float(f1_low), 6),
        "ci_high": round(float(f1_high), 6),
        "n": int(tp.size),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        macro_p = np.where(t + p > 0, t / (t + p), 0.0)
        macro_r = np.where(t + n > 0, t / (t + n), 0.0)
    macro_f1_values = per_item_f1
    macro_f1 = bootstrap_mean_ci(macro_f1_values, n_resamples, seed)
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": _ci_dict(macro_p.mean(), macro_p, int(tp.size)),
        "macro_recall": _ci_dict(macro_r.mean(), macro_r, int(tp.size)),
        "macro_f1": macro_f1,
    }


def _ci_dict(point: float, values: np.ndarray, n: int) -> dict[str, Any]:
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "value": round(float(point), 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "n": n,
    }


@dataclass
class EvalResult:
    """Metrics and per-item records for one system evaluation."""

    system_name: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    per_family: dict[str, Any]
    per_item: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evaluation artifact."""
        return asdict(self)

    def write(self, eval_dir: str | Path) -> Path:
        """Write ``artifacts/eval/<system_name>.json``."""
        out = Path(eval_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.system_name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def evaluate(
    system_name: str,
    answerer: ClinicalAnswerer,
    benchmark: list[dict[str, Any]],
    k: int | None = 5,
    mode: str = "hybrid",
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
    eval_dir: str | Path | None = "artifacts/eval",
    write: bool = True,
) -> EvalResult:
    """Run a system over the benchmark and compute all metrics with 95% CIs.

    Retrieval, QA, hallucination, abstention and code metrics are computed from
    real per-item outputs. Confidence intervals use 10,000 bootstrap resamples
    with seed 42.
    """
    per_item: list[dict[str, Any]] = []
    for item in benchmark:
        item_k = k if k is not None else int(item.get("k", 5))
        answer = answerer.answer(
            item["question"], filters=item.get("filters", {}), k=item_k, mode=mode
        )
        retrieved = [score["chunk_id"] for score in answer.retrieval_scores]
        recall = retrieval_recall(retrieved, item.get("gold_supporting_chunk_ids", []))
        rr = reciprocal_rank(retrieved, item.get("gold_supporting_chunk_ids", []))
        gold_codes = code_set(item.get("gold_codes", []))
        pred_codes = predicted_codes(answer.text) if not answer.abstained else set()
        tp, fp, fn = code_counts(pred_codes, gold_codes)
        record = {
            "id": item["id"],
            "family": item["family"],
            "task_type": item.get("task_type"),
            "answerable": bool(item["answerable"]),
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "answer": answer.text,
            "abstained": answer.abstained,
            "abstention_reason": answer.abstention_reason,
            "unsupported_claim": answer.unsupported_claim,
            "citations": [citation.chunk_id for citation in answer.citations],
            "retrieved_chunk_ids": retrieved,
            "gold_supporting_chunk_ids": item.get("gold_supporting_chunk_ids", []),
            "recall": recall,
            "reciprocal_rank": rr,
            "exact_match": exact_match(answer.text, item["gold_answer"]),
            "token_f1": token_f1(answer.text, item["gold_answer"]),
            "numeric_match": (
                numeric_match(answer.text, float(item["gold_numeric"]))
                if item.get("gold_numeric") is not None
                else None
            ),
            "code_accuracy": (
                (1.0 if gold_codes and gold_codes <= pred_codes else 0.0) if gold_codes else None
            ),
            "predicted_codes": sorted(
                [{"system": s, "code": c} for s, c in pred_codes],
                key=lambda x: (x["system"], x["code"]),
            ),
            "gold_codes": item.get("gold_codes", []),
            "code_tp": tp,
            "code_fp": fp,
            "code_fn": fn,
            "latency_ms": answer.latency_ms,
            "est_cost_usd": answer.est_cost_usd,
            "confidence": answer.confidence,
            "backend_name": answer.backend_name,
        }
        per_item.append(record)

    answerable = np.asarray([r["answerable"] for r in per_item], dtype=bool)
    abstained = np.asarray([r["abstained"] for r in per_item], dtype=bool)
    answerable_records = [r for r in per_item if r["answerable"]]
    unanswerable_records = [r for r in per_item if not r["answerable"]]

    recall_values = [r["recall"] for r in answerable_records if r["recall"] is not None]
    mrr_values = [
        r["reciprocal_rank"] for r in answerable_records if r["reciprocal_rank"] is not None
    ]
    exact_values = [r["exact_match"] for r in answerable_records]
    f1_values = [r["token_f1"] for r in answerable_records]
    numeric_values = [
        r["numeric_match"] for r in answerable_records if r["numeric_match"] is not None
    ]
    code_acc_values = [
        r["code_accuracy"] for r in answerable_records if r["code_accuracy"] is not None
    ]
    hallucination_values = [1.0 if r["unsupported_claim"] else 0.0 for r in answerable_records]
    false_answer_values = [1.0 if not r["abstained"] else 0.0 for r in unanswerable_records]
    citation_values = [
        r.get("confidence", 0.0) for r in per_item
    ]  # placeholder replaced below by verification coverage if present
    # The answerer exposes confidence per item; citation support is measured by
    # the verifier and stored in the answer object, so recompute from the
    # unsupported flag as a conservative lower bound.
    citation_values = [
        0.0 if r["unsupported_claim"] else 1.0 for r in per_item if not r["abstained"]
    ]

    code_records = [r for r in answerable_records if r["gold_codes"]]
    tp = np.asarray([r["code_tp"] for r in code_records], dtype=float)
    fp = np.asarray([r["code_fp"] for r in code_records], dtype=float)
    fn = np.asarray([r["code_fn"] for r in code_records], dtype=float)
    per_item_f1 = np.asarray(
        [prf(r["code_tp"], r["code_fp"], r["code_fn"])[2] for r in code_records], dtype=float
    )
    code_metrics = _code_summary(tp, fp, fn, per_item_f1, n_resamples, seed)

    metrics: dict[str, Any] = {
        "n_items": len(per_item),
        "n_answerable": len(answerable_records),
        "n_unanswerable": len(unanswerable_records),
        "n_answered": int((~abstained).sum()),
        "abstention_rate": _summary(
            [1.0 if r["abstained"] else 0.0 for r in per_item], n_resamples, seed
        ),
        "answerable_abstention_rate": _summary(
            [1.0 if r["abstained"] else 0.0 for r in answerable_records], n_resamples, seed
        ),
        "recall_at_k": _summary(recall_values, n_resamples, seed),
        "mrr": _summary(mrr_values, n_resamples, seed),
        "exact_match": _summary(exact_values, n_resamples, seed),
        "token_f1": _summary(f1_values, n_resamples, seed),
        "numeric_accuracy": _summary(numeric_values, n_resamples, seed),
        "code_aware_accuracy": _summary(code_acc_values, n_resamples, seed),
        "hallucination_rate": _summary(hallucination_values, n_resamples, seed),
        "false_answer_rate": _summary(false_answer_values, n_resamples, seed),
        "citation_support_rate": _summary(citation_values, n_resamples, seed)
        if citation_values
        else _summary([], n_resamples, seed),
        "latency_p50_ms": round(float(np.percentile([r["latency_ms"] for r in per_item], 50)), 3),
        "latency_p95_ms": round(float(np.percentile([r["latency_ms"] for r in per_item], 95)), 3),
        "mean_latency_ms": _summary([r["latency_ms"] for r in per_item], n_resamples, seed),
        "mean_cost_usd": _summary([r["est_cost_usd"] for r in per_item], n_resamples, seed),
        "code": code_metrics,
    }
    abstention = _abstention_summary(answerable, abstained, n_resamples, seed)
    metrics["abstention_precision"] = abstention["precision"]
    metrics["abstention_recall"] = abstention["recall"]

    per_family: dict[str, Any] = {}
    for family in ("treatment", "payment", "operations", "adversarial"):
        subset = [r for r in per_item if r["family"] == family]
        answered = [r for r in subset if not r["abstained"]]
        per_family[family] = {
            "n": len(subset),
            "abstention_rate": round(sum(1 for r in subset if r["abstained"]) / len(subset), 6)
            if subset
            else 0.0,
            "exact_match": round(float(np.mean([r["exact_match"] for r in subset])), 6)
            if subset
            else 0.0,
            "token_f1": round(float(np.mean([r["token_f1"] for r in subset])), 6)
            if subset
            else 0.0,
            "recall_at_k": round(
                float(np.mean([r["recall"] for r in subset if r["recall"] is not None])), 6
            )
            if any(r["recall"] is not None for r in subset)
            else 0.0,
            "hallucination_rate": round(
                sum(1 for r in subset if r["unsupported_claim"]) / len(subset), 6
            )
            if subset
            else 0.0,
            "mean_citations": round(float(np.mean([len(r["citations"]) for r in answered])), 3)
            if answered
            else 0.0,
        }

    result = EvalResult(
        system_name=system_name,
        config={
            "k": k,
            "mode": mode,
            "seed": seed,
            "n_resamples": n_resamples,
            "embedding_backend": answerer.retriever.embedder.name,
            "generator_backend": answerer.generator.name,
            "store_backend": answerer.retriever.store.name,
            "system_name": system_name,
        },
        metrics=metrics,
        per_family=per_family,
        per_item=per_item,
    )
    if write and eval_dir is not None:
        result.write(eval_dir)
    return result


def write_leaderboard(
    results: list[EvalResult],
    eval_dir: str | Path = "artifacts/eval",
) -> tuple[Path, Path]:
    """Write ``leaderboard.md`` and ``leaderboard.csv`` with metric ± 95% CI."""
    out = Path(eval_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = [
        ("recall_at_k", "Recall@k"),
        ("mrr", "MRR"),
        ("exact_match", "Exact"),
        ("token_f1", "Token F1"),
        ("numeric_accuracy", "Numeric acc"),
        ("code_aware_accuracy", "Code acc"),
        ("hallucination_rate", "Hallucination"),
        ("false_answer_rate", "False answer"),
        ("abstention_precision", "Abst P"),
        ("abstention_recall", "Abst R"),
    ]

    def fmt(summary: dict[str, Any]) -> str:
        return f"{summary['value']:.3f} [{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]"

    lines = [
        "# Clinical RAG evaluation leaderboard",
        "",
        "Synthetic-data evaluation; all numbers produced by `clinicalnoterag eval`.",
        "Intervals are 95% percentile bootstrap CIs with 10,000 resamples (seed 42).",
        "",
        "| System | k | Mode | Recall@k | MRR | Exact | Token F1 | Hallucination | False answer | Abst P | Abst R | Code F1 micro | Code F1 macro | p50 ms | p95 ms | Cost/query |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        m = result.metrics
        lines.append(
            "| {system} | {k} | {mode} | {recall} | {mrr} | {exact} | {f1} | {hall} | {false} | {ap} | {ar} | {micro} | {macro} | {p50:.1f} | {p95:.1f} | ${cost:.6f} |".format(
                system=result.system_name,
                k=result.config.get("k"),
                mode=result.config.get("mode"),
                recall=fmt(m["recall_at_k"]),
                mrr=fmt(m["mrr"]),
                exact=fmt(m["exact_match"]),
                f1=fmt(m["token_f1"]),
                hall=fmt(m["hallucination_rate"]),
                false=fmt(m["false_answer_rate"]),
                ap=fmt(m["abstention_precision"]),
                ar=fmt(m["abstention_recall"]),
                micro=fmt(m["code"]["micro_f1"]),
                macro=fmt(m["code"]["macro_f1"]),
                p50=m["latency_p50_ms"],
                p95=m["latency_p95_ms"],
                cost=m["mean_cost_usd"]["value"],
            )
        )
    lines.extend(
        [
            "",
            "Metric definitions: Recall@k and MRR over gold supporting chunks; Exact and Token F1",
            "against gold answers; Hallucination = fraction of answerable items with at least one",
            "unsupported sentence; False answer = answered unanswerable items; Abst P/R treat",
            "abstention as the positive class; code F1 uses ICD-10/CPT/SNOMED/RxNorm/CARC codes.",
            "The benchmark is closed-world: questions are generated from the same synthetic FHIR",
            "bundles that populate the index, so these results are an upper bound on real-world",
            "performance.",
        ]
    )
    md_path = out / "leaderboard.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out / "leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["system", "k", "mode", "embedding_backend", "generator_backend", "store_backend"]
        for _key, label in columns:
            header.extend([label, f"{label} ci_low", f"{label} ci_high"])
        header.extend(["code_f1_micro", "code_f1_micro_ci_low", "code_f1_micro_ci_high"])
        header.extend(["code_f1_macro", "code_f1_macro_ci_low", "code_f1_macro_ci_high"])
        header.extend(["latency_p50_ms", "latency_p95_ms", "mean_cost_usd"])
        writer.writerow(header)
        for result in results:
            m = result.metrics
            row = [
                result.system_name,
                result.config.get("k"),
                result.config.get("mode"),
                result.config.get("embedding_backend"),
                result.config.get("generator_backend"),
                result.config.get("store_backend"),
            ]
            for key, _ in columns:
                summary = m[key]
                row.extend([summary["value"], summary["ci_low"], summary["ci_high"]])
            row.extend(
                [
                    m["code"]["micro_f1"]["value"],
                    m["code"]["micro_f1"]["ci_low"],
                    m["code"]["micro_f1"]["ci_high"],
                    m["code"]["macro_f1"]["value"],
                    m["code"]["macro_f1"]["ci_low"],
                    m["code"]["macro_f1"]["ci_high"],
                    m["latency_p50_ms"],
                    m["latency_p95_ms"],
                    m["mean_cost_usd"]["value"],
                ]
            )
            writer.writerow(row)
    return md_path, csv_path


def run_ablations(
    answerer: ClinicalAnswerer,
    benchmark: list[dict[str, Any]],
    modes: tuple[str, ...] = ("dense", "bm25", "hybrid"),
    ks: tuple[int, ...] = (1, 3, 5, 10),
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_RESAMPLES,
    out_path: str | Path = "artifacts/eval/ablations.csv",
) -> list[dict[str, Any]]:
    """Sweep retrieval mode and k, writing real retrieval/QA metrics to CSV."""
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for k in ks:
            result = evaluate(
                f"ablation-{mode}-k{k}",
                answerer,
                benchmark,
                k=k,
                mode=mode,
                seed=seed,
                n_resamples=n_resamples,
                write=False,
            )
            metrics = result.metrics
            rows.append(
                {
                    "mode": mode,
                    "k": k,
                    "n_items": metrics["n_items"],
                    "recall_at_k": metrics["recall_at_k"]["value"],
                    "recall_ci_low": metrics["recall_at_k"]["ci_low"],
                    "recall_ci_high": metrics["recall_at_k"]["ci_high"],
                    "mrr": metrics["mrr"]["value"],
                    "exact_match": metrics["exact_match"]["value"],
                    "token_f1": metrics["token_f1"]["value"],
                    "hallucination_rate": metrics["hallucination_rate"]["value"],
                    "abstention_rate": metrics["abstention_rate"]["value"],
                    "p50_latency_ms": metrics["latency_p50_ms"],
                    "p95_latency_ms": metrics["latency_p95_ms"],
                }
            )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


__all__ = [
    "EvalResult",
    "evaluate",
    "paired_bootstrap_delta",
    "run_ablations",
    "write_leaderboard",
]
