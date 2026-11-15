"""Fallback learned component: a cross-encoder-style reranker.

If the local instruct model cannot be trained in the build environment, this
reranker is trained instead on the same patient-disjoint splits. It scores
(question, candidate chunk) pairs and is used in the retrieval loop for the
``rag-tuned`` system. It is always labelled as the fallback learned component,
never silently substituted for a LoRA adapter.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clinicalnoterag.eval.benchmark import load_benchmark
from clinicalnoterag.eval.metrics import reciprocal_rank, retrieval_recall
from clinicalnoterag.rag.codes import extract_codes
from clinicalnoterag.rag.retriever import HybridRetriever, RetrievedChunk
from clinicalnoterag.rag.verify import content_tokens
from clinicalnoterag.seeds import set_global_seed

RESOURCE_TYPES = [
    "Condition",
    "Procedure",
    "Claim",
    "ClaimResponse",
    "Observation",
    "MedicationRequest",
    "Encounter",
    "Note",
]
INTENTS = [
    ("diagnos", "diagnosis"),
    ("procedure", "procedure"),
    ("medication", "medication"),
    ("observation", "observation"),
    ("systolic", "observation"),
    ("blood pressure", "observation"),
    ("how many", "count"),
    ("total", "total"),
    ("denied", "denied"),
    ("appeal", "appeal"),
]


def _id_flags(question: str, chunk: RetrievedChunk) -> list[float]:
    flags = []
    for pattern, key in (
        (r"\bPAT-\d{4}\b", "patient_id"),
        (r"\bENC-\d{4}-\d{2}\b", "encounter_id"),
        (r"\bCLM-\d{4}-\d{2}\b", "claim_id"),
    ):
        match = re.search(pattern, question)
        flags.append(
            1.0
            if match
            and (
                match.group(0) == str(chunk.metadata.get(key) or "") or match.group(0) in chunk.text
            )
            else 0.0
        )
    return flags


def extract_features(question: str, chunks: list[RetrievedChunk]) -> np.ndarray:
    """Build deterministic numeric features for each candidate chunk."""
    q_tokens = content_tokens(question)
    q_codes = {(code.system, code.code) for code in extract_codes(question)}
    rows: list[list[float]] = []
    for rank, chunk in enumerate(chunks, start=1):
        c_tokens = content_tokens(chunk.text)
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
        chunk_codes = {(code.system, code.code) for code in extract_codes(chunk.text)}
        if chunk.metadata.get("code"):
            chunk_codes.add(
                (str(chunk.metadata.get("code_system")), str(chunk.metadata.get("code")))
            )
        code_match = len(q_codes & chunk_codes) / max(1, len(q_codes)) if q_codes else 0.0
        resource = str(chunk.metadata.get("resource_type") or "Note")
        resource_onehot = [1.0 if resource == value else 0.0 for value in RESOURCE_TYPES]
        intent_onehot = [1.0 if needle in question.lower() else 0.0 for needle, _ in INTENTS]
        rows.append(
            [
                float(chunk.score),
                float(chunk.dense_score),
                float(chunk.bm25_score),
                1.0 / rank,
                float(rank) / 20.0,
                overlap,
                code_match,
                *_id_flags(question, chunk),
                1.0 if chunk.metadata.get("doc_type") == "fhir_fact" else 0.0,
                0.0 if chunk.metadata.get("doc_type") == "fhir_fact" else 1.0,
                *resource_onehot,
                *intent_onehot,
                math.log1p(len(c_tokens)),
                math.log1p(len(q_tokens)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


@dataclass
class RerankerConfig:
    """Saved configuration for the fallback reranker."""

    hidden_dim: int
    epochs: int
    learning_rate: float
    seed: int
    n_features: int
    train_pairs: int
    val_pairs: int
    best_val_loss: float


class ClinicalReranker:
    """A small PyTorch MLP cross-encoder reranker over retrieval candidates."""

    name = "fallback-mlp-reranker"

    def __init__(self, hidden_dim: int = 32, seed: int = 42) -> None:
        import torch
        from torch import nn

        self.config: RerankerConfig | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        torch.manual_seed(seed)
        self._model = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self._torch = torch

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return features
        return ((features - self.feature_mean) / self.feature_std).astype(np.float32)

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        hidden_dim: int = 32,
        epochs: int = 120,
        learning_rate: float = 1e-3,
        seed: int = 42,
        patience: int = 20,
    ) -> dict[str, Any]:
        """Train the MLP with BCE loss and early stopping on validation loss."""
        import torch
        from torch import nn

        torch.set_num_threads(1)
        set_global_seed(seed)
        self.feature_mean = train_features.mean(axis=0)
        self.feature_std = train_features.std(axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        train_x = torch.from_numpy(self._standardize(train_features))
        train_y = torch.from_numpy(train_labels.astype(np.float32)).view(-1, 1)
        val_x = torch.from_numpy(self._standardize(val_features))
        val_y = torch.from_numpy(val_labels.astype(np.float32)).view(-1, 1)
        self._model = nn.Sequential(
            nn.Linear(train_x.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        positives = float(train_labels.sum())
        negatives = float(len(train_labels) - positives)
        pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        history: list[dict[str, float]] = []
        best_state = None
        best_val = float("inf")
        stale = 0
        for epoch in range(1, epochs + 1):
            self._model.train()
            optimizer.zero_grad()
            logits = self._model(train_x)
            loss = loss_fn(logits, train_y)
            loss.backward()
            optimizer.step()
            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(val_x)
                val_loss = float(nn.functional.binary_cross_entropy_with_logits(val_logits, val_y))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(float(loss.item()), 6),
                    "val_loss": round(val_loss, 6),
                }
            )
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {
                    key: value.detach().clone() for key, value in self._model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self.config = RerankerConfig(
            hidden_dim=hidden_dim,
            epochs=len(history),
            learning_rate=learning_rate,
            seed=seed,
            n_features=int(train_x.shape[1]),
            train_pairs=int(train_x.shape[0]),
            val_pairs=int(val_x.shape[0]),
            best_val_loss=round(best_val, 6),
        )
        return {"history": history, "best_val_loss": best_val, "epochs": len(history)}

    def score(self, question: str, chunks: list[RetrievedChunk]) -> np.ndarray:
        """Return a relevance probability for each candidate chunk."""
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        self._torch.set_num_threads(1)
        features = self._standardize(extract_features(question, chunks))
        self._model.eval()
        with self._torch.no_grad():
            logits = self._model(self._torch.from_numpy(features))
            probabilities = self._torch.sigmoid(logits).view(-1).numpy()
        return np.asarray(probabilities, dtype=np.float32)

    def rerank(
        self, question: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Reorder candidates by learned relevance and return the top ``top_k``."""
        scores = self.score(question, chunks)
        order = np.argsort(-scores)[:top_k]
        reranked: list[RetrievedChunk] = []
        for rank, index in enumerate(order, start=1):
            original = chunks[int(index)]
            reranked.append(
                RetrievedChunk(
                    record_id=original.record_id,
                    chunk_id=original.chunk_id,
                    text=original.text,
                    metadata=original.metadata,
                    score=float(scores[int(index)]),
                    dense_score=original.dense_score,
                    bm25_score=original.bm25_score,
                    rank=rank,
                )
            )
        return reranked

    def save(self, directory: str | Path) -> Path:
        """Save model weights, feature normalization and config."""
        import torch

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "model.pt")
        np.savez(
            path / "feature_stats.npz",
            mean=self.feature_mean,
            std=self.feature_std,
        )
        (path / "config.json").write_text(
            json.dumps(self.config.__dict__ if self.config else {}, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> ClinicalReranker:
        """Load a trained reranker from disk."""
        import torch

        path = Path(directory)
        config_data = json.loads((path / "config.json").read_text(encoding="utf-8"))
        instance = cls(hidden_dim=int(config_data["hidden_dim"]), seed=int(config_data["seed"]))
        instance.feature_mean = np.load(path / "feature_stats.npz")["mean"]
        instance.feature_std = np.load(path / "feature_stats.npz")["std"]
        from torch import nn

        instance._model = nn.Sequential(
            nn.Linear(int(config_data["n_features"]), int(config_data["hidden_dim"])),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(int(config_data["hidden_dim"]), 1),
        )
        instance._model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        instance._model.eval()
        instance.config = RerankerConfig(**config_data)
        return instance


def _candidate_pairs(
    retriever: HybridRetriever,
    items: list[dict[str, Any]],
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray, int]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with_positive = 0
    for item in items:
        chunks = retriever.retrieve(item["question"], k=k, filters=item.get("filters", {}))
        if not chunks:
            continue
        gold = set(item.get("gold_supporting_chunk_ids", []))
        y = np.asarray([1.0 if chunk.chunk_id in gold else 0.0 for chunk in chunks])
        if y.sum() == 0:
            continue
        features.append(extract_features(item["question"], chunks))
        labels.append(y)
        with_positive += 1
    if not features:
        raise RuntimeError("no candidate pairs with positive labels were built")
    return np.vstack(features), np.concatenate(labels), with_positive


def train_reranker(
    retriever: HybridRetriever,
    benchmark_path: str | Path = "data/benchmark/clinical_qa.jsonl",
    split_manifest_path: str | Path = "data/sft/split_manifest.json",
    out_dir: str | Path = "artifacts/adapters/clinical-reranker",
    log_path: str | Path = "artifacts/train/reranker_log.json",
    seed: int = 42,
    epochs: int = 120,
) -> dict[str, Any]:
    """Train the fallback reranker on train/val patients and log a real curve."""
    set_global_seed(seed)
    items = load_benchmark(benchmark_path)
    manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    train_patients = set(manifest["splits"]["train"])
    val_patients = set(manifest["splits"]["val"])
    test_patients = set(manifest["splits"]["test"])
    train_items = [item for item in items if item["patient_id"] in train_patients]
    val_items = [item for item in items if item["patient_id"] in val_patients]
    test_items = [item for item in items if item["patient_id"] in test_patients]

    train_x, train_y, n_train_items = _candidate_pairs(retriever, train_items)
    val_x, val_y, n_val_items = _candidate_pairs(retriever, val_items)
    model = ClinicalReranker(hidden_dim=32, seed=seed)
    fit_result = model.fit(
        train_x,
        train_y,
        val_x,
        val_y,
        hidden_dim=32,
        epochs=epochs,
        learning_rate=1e-3,
        seed=seed,
    )
    model.save(out_dir)

    # Honest before/after retrieval comparison on the held-out test patients.
    before_recall: list[float] = []
    after_recall: list[float] = []
    before_rr: list[float] = []
    after_rr: list[float] = []
    for item in test_items:
        candidates = retriever.retrieve(item["question"], k=20, filters=item.get("filters", {}))
        gold = item.get("gold_supporting_chunk_ids", [])
        if not gold:
            continue
        before = candidates[:5]
        after = model.rerank(item["question"], candidates, top_k=5)
        before_recall.append(retrieval_recall([c.chunk_id for c in before], gold) or 0.0)
        after_recall.append(retrieval_recall([c.chunk_id for c in after], gold) or 0.0)
        before_rr.append(reciprocal_rank([c.chunk_id for c in before], gold) or 0.0)
        after_rr.append(reciprocal_rank([c.chunk_id for c in after], gold) or 0.0)

    result = {
        "component": "fallback-mlp-reranker",
        "status": "trained",
        "path": str(out_dir),
        "config": model.config.__dict__ if model.config else {},
        "train_items": len(train_items),
        "val_items": len(val_items),
        "test_items": len(test_items),
        "train_pairs": int(train_x.shape[0]),
        "val_pairs": int(val_x.shape[0]),
        "train_items_with_positives": n_train_items,
        "val_items_with_positives": n_val_items,
        "history": fit_result["history"],
        "best_val_loss": round(float(fit_result["best_val_loss"]), 6),
        "test_retrieval_before": {
            "recall_at_5": round(float(np.mean(before_recall)), 6) if before_recall else 0.0,
            "mrr": round(float(np.mean(before_rr)), 6) if before_rr else 0.0,
            "n": len(before_recall),
        },
        "test_retrieval_after": {
            "recall_at_5": round(float(np.mean(after_recall)), 6) if after_recall else 0.0,
            "mrr": round(float(np.mean(after_rr)), 6) if after_rr else 0.0,
            "n": len(after_recall),
        },
        "label": "Fallback learned component (not a LoRA adapter).",
    }
    log_target = Path(log_path)
    log_target.parent.mkdir(parents=True, exist_ok=True)
    log_target.write_text(json.dumps(result, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        history = fit_result["history"]
        plt.figure(figsize=(6, 4))
        plt.plot(
            [row["epoch"] for row in history], [row["train_loss"] for row in history], label="train"
        )
        plt.plot(
            [row["epoch"] for row in history], [row["val_loss"] for row in history], label="val"
        )
        plt.xlabel("Epoch")
        plt.ylabel("BCE loss")
        plt.title("Fallback reranker training curve")
        plt.legend()
        plt.tight_layout()
        png = log_target.with_suffix(".png")
        plt.savefig(png, dpi=150)
        plt.close()
        result["curve_png"] = str(png)
        log_target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - plotting is best-effort
        result["plot_error"] = f"{type(exc).__name__}: {exc}"
        log_target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = [
    "ClinicalReranker",
    "RerankerConfig",
    "extract_features",
    "train_reranker",
]
