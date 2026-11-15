"""Build supervised fine-tuning pairs with a strict patient-disjoint split."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clinicalnoterag.eval.benchmark import load_benchmark
from clinicalnoterag.rag.codes import extract_codes
from clinicalnoterag.rag.generators import ABSTENTION_TEXT, build_prompt
from clinicalnoterag.rag.retriever import HybridRetriever
from clinicalnoterag.rag.verify import content_tokens, split_sentences

DEFAULT_SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass
class SFTStats:
    """Counts and split provenance for the SFT dataset."""

    n_train: int
    n_val: int
    n_test: int
    n_train_patients: int
    n_val_patients: int
    n_test_patients: int
    train_patients: list[str]
    val_patients: list[str]
    test_patients: list[str]
    seed: int
    out_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return asdict(self)


def split_patients(patient_ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Split unique patients 70/15/15 with a fixed shuffle seed."""
    patients = sorted(set(patient_ids))
    random.Random(seed).shuffle(patients)
    n = len(patients)
    n_train = max(1, int(round(n * DEFAULT_SPLIT["train"])))
    n_val = max(1, int(round(n * DEFAULT_SPLIT["val"])))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = sorted(patients[:n_train])
    val = sorted(patients[n_train : n_train + n_val])
    test = sorted(patients[n_train + n_val :])
    if not test:
        test = [val.pop()]
    if not val:
        val = [train.pop()]
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    return {"train": train, "val": val, "test": test}


def _citation_target(item: dict[str, Any], contexts: list[Any]) -> str:
    """Attach inline citations to a gold answer using the retrieved context."""
    if not item["answerable"]:
        return ABSTENTION_TEXT
    gold_ids = set(item.get("gold_supporting_chunk_ids", []))
    gold_indices = [
        index for index, chunk in enumerate(contexts, start=1) if chunk.chunk_id in gold_ids
    ]
    if not gold_indices:
        gold_indices = [1] if contexts else []
    sentences = split_sentences(item["gold_answer"]) or [item["gold_answer"]]
    cited: list[str] = []
    for sentence in sentences:
        sentence_codes = {(code.system, code.code) for code in extract_codes(sentence)}
        best_index = gold_indices[0] if gold_indices else None
        best_overlap = -1.0
        for index in gold_indices:
            chunk = contexts[index - 1]
            chunk_codes = {(code.system, code.code) for code in extract_codes(chunk.text)}
            code_bonus = 0.5 if sentence_codes & chunk_codes else 0.0
            overlap = len(content_tokens(sentence) & content_tokens(chunk.text)) / max(
                1, len(content_tokens(sentence))
            )
            if overlap + code_bonus > best_overlap:
                best_overlap = overlap + code_bonus
                best_index = index
        marker = f" [{best_index}]" if best_index is not None else ""
        cited.append(sentence.rstrip(".") + marker + ".")
    return " ".join(cited)


def build_sft_dataset(
    corpus_dir: str | Path,
    benchmark_path: str | Path,
    retriever: HybridRetriever,
    out_dir: str | Path = "data/sft",
    seed: int = 42,
    k: int = 5,
) -> SFTStats:
    """Build train/val/test JSONL SFT pairs, split strictly by patient.

    Each example is ``prompt`` (numbered retrieved context + question) and
    ``completion`` (a cited gold answer, or the explicit abstention target for
    unanswerable items). The split is by patient so no patient appears in more
    than one split.
    """
    items = load_benchmark(benchmark_path)
    splits = split_patients([item["patient_id"] for item in items], seed=seed)
    split_of = {patient: name for name, patients in splits.items() for patient in patients}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    handles = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name in splits}
    try:
        for item in items:
            patient = item["patient_id"]
            split = split_of[patient]
            contexts = retriever.retrieve(
                item["question"],
                k=k,
                filters=item.get("filters", {}),
            )
            prompt = build_prompt(item["question"], contexts)
            completion = _citation_target(item, contexts)
            record = {
                "id": item["id"],
                "patient_id": patient,
                "split": split,
                "family": item["family"],
                "answerable": item["answerable"],
                "question": item["question"],
                "prompt": prompt,
                "completion": completion,
                "text": f"{prompt}\n{completion}",
                "context_chunk_ids": [chunk.chunk_id for chunk in contexts],
                "gold_supporting_chunk_ids": item.get("gold_supporting_chunk_ids", []),
                "gold_codes": item.get("gold_codes", []),
            }
            handles[split].write(json.dumps(record) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "seed": seed,
        "k": k,
        "splits": splits,
        "counts": counts,
        "n_patients": {name: len(patients) for name, patients in splits.items()},
        "disjoint": not (
            set(splits["train"]) & set(splits["val"])
            or set(splits["train"]) & set(splits["test"])
            or set(splits["val"]) & set(splits["test"])
        ),
        "policy": "Strict patient-level split; no patient appears in more than one split.",
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return SFTStats(
        n_train=counts["train"],
        n_val=counts["val"],
        n_test=counts["test"],
        n_train_patients=len(splits["train"]),
        n_val_patients=len(splits["val"]),
        n_test_patients=len(splits["test"]),
        train_patients=splits["train"],
        val_patients=splits["val"],
        test_patients=splits["test"],
        seed=seed,
        out_dir=str(out),
    )


def load_sft_split(split: str, out_dir: str | Path = "data/sft") -> list[dict[str, Any]]:
    """Load one SFT split (``train``, ``val`` or ``test``)."""
    path = Path(out_dir) / f"{split}.jsonl"
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


__all__ = ["SFTStats", "build_sft_dataset", "load_sft_split", "split_patients"]
