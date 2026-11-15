"""Benchmark construction, evaluation runner and learned reranker tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clinicalnoterag.data.synthea_gen import generate_corpus
from clinicalnoterag.eval.benchmark import build_benchmark, load_benchmark
from clinicalnoterag.eval.runner import evaluate, write_leaderboard
from clinicalnoterag.rag.generators import ABSTENTION_TEXT
from clinicalnoterag.train.fallback_reranker import ClinicalReranker, extract_features


def test_build_benchmark_has_all_families(tmp_path: Path):
    corpus = tmp_path / "corpus"
    generate_corpus(25, corpus, seed=42)
    items = build_benchmark(corpus, tmp_path / "benchmark" / "clinical_qa.jsonl")
    assert len(items) >= 200
    families = {item["family"] for item in items}
    assert families == {"treatment", "payment", "operations", "adversarial"}
    assert any(item["answerable"] for item in items)
    assert any(not item["answerable"] for item in items)
    assert all("question" in item and "gold_answer" in item for item in items)
    assert load_benchmark(tmp_path / "benchmark" / "clinical_qa.jsonl")[0]["id"] == items[0]["id"]


def _sample_items(tiny_system) -> list[dict]:
    items = [
        {
            "id": "T-test-1",
            "family": "treatment",
            "task_type": "diagnoses",
            "question": "What diagnoses were documented for encounter ENC-0001-01?",
            "gold_answer": "Cerebral infarction, unspecified was documented (ICD-10 I63.9).",
            "gold_codes": [{"system": "ICD-10", "code": "I63.9"}],
            "gold_supporting_chunk_ids": ["FACT::Condition::CON-0001-01-1"],
            "answerable": True,
            "gold_numeric": None,
            "filters": {
                "patient_id": "PAT-0001",
                "encounter_id": "ENC-0001-01",
                "resource_type": "Condition",
                "doc_type": "fhir_fact",
            },
            "k": 5,
            "patient_id": "PAT-0001",
        },
        {
            "id": "O-test-1",
            "family": "operations",
            "task_type": "encounter_count",
            "question": "How many encounters did patient PAT-0001 have?",
            "gold_answer": "There are 2 encounters for patient PAT-0001.",
            "gold_codes": [],
            "gold_supporting_chunk_ids": [
                "FACT::Encounter::ENC-0001-01",
                "FACT::Encounter::ENC-0001-02",
            ],
            "answerable": True,
            "gold_numeric": 2.0,
            "filters": {
                "patient_id": "PAT-0001",
                "resource_type": "Encounter",
                "doc_type": "fhir_fact",
            },
            "k": 5,
            "patient_id": "PAT-0001",
        },
        {
            "id": "A-test-1",
            "family": "adversarial",
            "task_type": "absent_diagnosis",
            "question": "What diagnosis code Z99.999 was documented for patient PAT-0001?",
            "gold_answer": ABSTENTION_TEXT,
            "gold_codes": [],
            "gold_supporting_chunk_ids": [],
            "answerable": False,
            "gold_numeric": None,
            "filters": {
                "patient_id": "PAT-0001",
                "resource_type": "Condition",
                "doc_type": "fhir_fact",
            },
            "k": 5,
            "patient_id": "PAT-0001",
        },
    ]
    return items


def test_evaluate_runner_writes_artifact_and_leaderboard(tiny_system, tmp_path: Path):
    items = _sample_items(tiny_system)
    result = evaluate(
        "test-system",
        tiny_system.answerer,
        items,
        k=5,
        n_resamples=200,
        eval_dir=tmp_path / "eval",
    )
    assert result.metrics["n_items"] == 3
    assert result.metrics["n_answerable"] == 2
    assert result.metrics["n_unanswerable"] == 1
    assert result.metrics["recall_at_k"]["n"] == 2
    assert result.metrics["false_answer_rate"]["value"] == 0.0
    assert (tmp_path / "eval" / "test-system.json").exists()
    md_path, csv_path = write_leaderboard([result], tmp_path / "eval")
    assert md_path.exists() and csv_path.exists()
    assert "test-system" in md_path.read_text(encoding="utf-8")


def test_reranker_features_and_save_load(tiny_system, tmp_path: Path):
    hits = tiny_system.retriever.retrieve(
        "What diagnoses were documented for encounter ENC-0001-01?",
        k=5,
        filters={
            "patient_id": "PAT-0001",
            "encounter_id": "ENC-0001-01",
            "resource_type": "Condition",
            "doc_type": "fhir_fact",
        },
    )
    features = extract_features("What diagnoses", hits)
    assert features.shape[0] == len(hits)
    assert features.shape[1] > 10

    rng = np.random.default_rng(42)
    n_features = features.shape[1]
    train_x = rng.normal(size=(30, n_features)).astype(np.float32)
    train_y = (rng.random(30) > 0.5).astype(np.float32)
    val_x = rng.normal(size=(10, n_features)).astype(np.float32)
    val_y = (rng.random(10) > 0.5).astype(np.float32)
    model = ClinicalReranker(hidden_dim=8, seed=42)
    model.fit(train_x, train_y, val_x, val_y, hidden_dim=8, epochs=8, seed=42)
    model.save(tmp_path / "reranker")
    loaded = ClinicalReranker.load(tmp_path / "reranker")
    subset = hits[:2]
    scores = loaded.score("What diagnoses", subset)
    assert scores.shape == (len(subset),)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    ranked = loaded.rerank("What diagnoses", hits, top_k=min(2, len(hits)))
    assert len(ranked) == min(2, len(hits))
