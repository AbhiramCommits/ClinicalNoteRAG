"""ClinicalNoteRAG command-line interface."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import typer

from clinicalnoterag import __version__
from clinicalnoterag.config import get_settings
from clinicalnoterag.data.synthea_gen import generate_corpus
from clinicalnoterag.deid.audit import scan_corpus
from clinicalnoterag.deid.engine import DeidEngine
from clinicalnoterag.embeddings.backends import get_embedding_backend, load_embedding_state
from clinicalnoterag.eval.benchmark import build_benchmark, load_benchmark
from clinicalnoterag.eval.runner import evaluate, run_ablations, write_leaderboard
from clinicalnoterag.pipeline.ingest import run_ingest
from clinicalnoterag.rag.answerer import ClinicalAnswerer
from clinicalnoterag.rag.generators import get_generator
from clinicalnoterag.rag.retriever import HybridRetriever
from clinicalnoterag.seeds import set_global_seed
from clinicalnoterag.telemetry import TelemetryRecorder
from clinicalnoterag.vectorstore.factory import get_store

app = typer.Typer(
    name="clinicalnoterag",
    help="Citation-grounded RAG over synthetic clinical and payment records.",
    no_args_is_help=True,
    add_completion=False,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _iter_note_records(corpus: Path):
    note_dir = corpus / "notes"
    files = sorted(note_dir.glob("*.jsonl"))
    if not files:
        raise typer.BadParameter(f"No note JSONL files found under {note_dir}")
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


@app.command("version")
def version_command() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("gen-corpus")
def gen_corpus_command(
    patients: int = typer.Option(300, min=1, help="Number of synthetic patients."),
    out: Path = typer.Option(Path("data/corpus"), help="Corpus output directory."),
    seed: int = typer.Option(42, help="Base seed (patient i uses seed+i-1)."),
    notes_per_encounter: int = typer.Option(2, min=1, max=2),
) -> None:
    """Generate synthetic FHIR bundles and clinical/payment notes."""
    set_global_seed(seed)
    start = time.perf_counter()
    stats = generate_corpus(patients, out, seed=seed, notes_per_encounter=notes_per_encounter)
    elapsed = time.perf_counter() - start
    typer.echo(
        json.dumps(
            {
                "patients": stats.n_patients,
                "encounters": stats.n_encounters,
                "documents": stats.n_documents,
                "clinical_notes": stats.n_clinical_notes,
                "payment_notes": stats.n_payment_notes,
                "claims": stats.n_claims,
                "denied_claims": stats.n_denied_claims,
                "out_dir": stats.out_dir,
                "manifest": stats.manifest_path,
                "elapsed_sec": round(elapsed, 3),
            },
            indent=2,
        )
    )


@app.command("gen-sample")
def gen_sample_command(
    out: Path = typer.Option(Path("data/sample"), help="Sample output directory."),
    patients: int = typer.Option(3, min=1, max=10),
    seed: int = typer.Option(42),
) -> None:
    """Generate the tiny committed sample corpus used by tests."""
    set_global_seed(seed)
    stats = generate_corpus(patients, out, seed=seed)
    typer.echo(
        json.dumps(
            {
                "patients": stats.n_patients,
                "documents": stats.n_documents,
                "out_dir": stats.out_dir,
                "manifest": stats.manifest_path,
            },
            indent=2,
        )
    )


@app.command("ingest")
def ingest_command(
    corpus: Path = typer.Option(Path("data/corpus"), help="Corpus directory."),
    store_name: str = typer.Option(
        "faiss", "--store", help="Vector store backend: faiss | pgvector."
    ),
    out: Path = typer.Option(Path("artifacts/ingest_report.json"), help="Report path."),
    fresh: bool = typer.Option(True, help="Rebuild the store from scratch."),
) -> None:
    """De-identify, chunk, embed, and upsert the synthetic corpus."""
    settings = get_settings()
    set_global_seed(settings.seed)
    manifest_path = corpus / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedder = get_embedding_backend(settings)
    deid = DeidEngine(backend="auto", known_names=list(manifest.get("known_names", [])))
    store = get_store(settings, embedder.dim, store_name=store_name, fresh=fresh)
    try:
        report = run_ingest(
            corpus,
            store,
            deid,
            embedder,
            settings,
            out_path=out,
        )
    finally:
        store.close()
    typer.echo(
        json.dumps(
            {
                "documents": report["documents"],
                "chunks": report["chunks"],
                "store": report["store_backend"],
                "embedding_backend": report["embedding_backend"],
                "deid_backend": report["deid_backend"],
                "chunks_per_sec": report["chunks_per_sec"],
                "total_wall_sec": report["total_wall_sec"],
                "artifact": str(out),
            },
            indent=2,
        )
    )


@app.command("eval")
def eval_command(
    system: str = typer.Option("rag-baseline", help="System name for the artifact."),
    k: int = typer.Option(5, min=1, max=50),
    mode: str = typer.Option("hybrid", help="hybrid | dense | bm25"),
    benchmark: Path = typer.Option(
        Path("data/benchmark/clinical_qa.jsonl"), help="Benchmark JSONL path."
    ),
    corpus: Path = typer.Option(Path("data/corpus"), help="Corpus directory."),
    ablations: bool = typer.Option(True, help="Run the k x retrieval-mode ablation sweep."),
    resamples: int = typer.Option(10000, min=100, help="Bootstrap resamples."),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate the RAG system on the labeled clinical benchmark.

    Writes ``artifacts/eval/<system>.json``, ``leaderboard.md/csv`` and, by
    default, ``artifacts/eval/ablations.csv``.
    """
    settings = get_settings()
    set_global_seed(seed)
    if not benchmark.exists():
        typer.echo(f"benchmark not found; building it at {benchmark}")
        build_benchmark(corpus, benchmark, seed=seed)
    items = load_benchmark(benchmark)
    embedder, _ = load_embedding_state(settings)
    store = get_store(settings, embedder.dim, fresh=False)
    try:
        retriever = HybridRetriever(store, embedder, settings)
        generator = get_generator(settings)
        answerer = ClinicalAnswerer(
            retriever,
            generator=generator,
            settings=settings,
            telemetry=TelemetryRecorder(settings.telemetry_path, enabled=False),
            system_name=system,
        )
        result = evaluate(
            system,
            answerer,
            items,
            k=k,
            mode=mode,
            seed=seed,
            n_resamples=resamples,
            eval_dir=settings.artifact_dir / "eval",
        )
        write_leaderboard([result], settings.artifact_dir / "eval")
        ablation_rows = []
        if ablations:
            ablation_rows = run_ablations(
                answerer,
                items,
                seed=seed,
                n_resamples=resamples,
                out_path=settings.artifact_dir / "eval" / "ablations.csv",
            )
    finally:
        store.close()
    typer.echo(
        json.dumps(
            {
                "system": system,
                "items": result.metrics["n_items"],
                "recall_at_k": result.metrics["recall_at_k"],
                "mrr": result.metrics["mrr"],
                "exact_match": result.metrics["exact_match"],
                "token_f1": result.metrics["token_f1"],
                "hallucination_rate": result.metrics["hallucination_rate"],
                "false_answer_rate": result.metrics["false_answer_rate"],
                "abstention_precision": result.metrics["abstention_precision"],
                "abstention_recall": result.metrics["abstention_recall"],
                "code_f1_micro": result.metrics["code"]["micro_f1"],
                "backends": {
                    "embedding": result.config["embedding_backend"],
                    "generator": result.config["generator_backend"],
                    "store": result.config["store_backend"],
                },
                "ablations": len(ablation_rows),
            },
            indent=2,
        )
    )


@app.command("train")
def train_command(
    component: str = typer.Option("all", help="all | lora | reranker"),
    max_steps: int = typer.Option(12, min=1, max=200),
    seed: int = typer.Option(42),
) -> None:
    """Build SFT pairs, train the LoRA adapter and/or the fallback reranker."""
    from clinicalnoterag.train.dataset import build_sft_dataset
    from clinicalnoterag.train.fallback_reranker import train_reranker
    from clinicalnoterag.train.finetune_lora import train_lora

    settings = get_settings()
    set_global_seed(seed)
    embedder, _ = load_embedding_state(settings)
    store = get_store(settings, embedder.dim, fresh=False)
    summary: dict[str, Any] = {}
    try:
        retriever = HybridRetriever(store, embedder, settings)
        sft_path = Path("data/sft/train.jsonl")
        if not sft_path.exists():
            stats = build_sft_dataset(
                "data/corpus", "data/benchmark/clinical_qa.jsonl", retriever, "data/sft", seed=seed
            )
            summary["sft"] = {
                "train": stats.n_train,
                "val": stats.n_val,
                "test": stats.n_test,
                "train_patients": stats.n_train_patients,
                "val_patients": stats.n_val_patients,
                "test_patients": stats.n_test_patients,
            }
        if component in {"all", "reranker"}:
            summary["reranker"] = train_reranker(retriever, seed=seed)
        if component in {"all", "lora"}:
            summary["lora"] = train_lora(settings, max_steps=max_steps)
    finally:
        store.close()
    typer.echo(
        json.dumps(
            {
                "component": component,
                "sft": summary.get("sft"),
                "lora_status": summary.get("lora", {}).get("status")
                if "lora" in summary
                else "not_run",
                "reranker_status": summary.get("reranker", {}).get("status")
                if "reranker" in summary
                else "not_run",
                "lora_config": {
                    key: summary.get("lora", {}).get(key)
                    for key in ("max_steps", "lora_r", "lora_alpha", "lora_target_modules")
                }
                if "lora" in summary
                else None,
            },
            indent=2,
        )
    )


@app.command("compare-systems")
def compare_systems_command(
    k: int = typer.Option(5, min=1, max=50),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate no-RAG, RAG zero-shot, tuned reranker and LoRA systems."""
    from clinicalnoterag.train.evaluate_systems import run_system_comparison

    settings = get_settings()
    set_global_seed(seed)
    result = run_system_comparison(settings=settings, k=k, seed=seed)
    typer.echo(
        json.dumps(
            {
                "test_items": result["n_test_items"],
                "systems": list(result["systems"].keys()),
                "deltas": result["deltas_rag_tuned_vs_zero_shot"],
                "artifact": "artifacts/eval/system_comparison.md",
            },
            indent=2,
        )
    )


@app.command("deid-audit")
def deid_audit_command(
    corpus: Path = typer.Option(Path("data/corpus"), help="Corpus directory."),
    out: Path = typer.Option(Path("artifacts/deid_audit.json"), help="Audit artifact path."),
    backend: str = typer.Option("auto", help="auto | presidio | regex"),
    limit: int | None = typer.Option(None, min=1, help="Optional document limit for quick checks."),
) -> None:
    """De-identify every note and measure the real PHI leak rate."""
    settings = get_settings()
    set_global_seed(settings.seed)
    manifest_path = corpus / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine = DeidEngine(backend=backend, known_names=list(manifest.get("known_names", [])))

    started = time.perf_counter()
    records = []
    for i, record in enumerate(_iter_note_records(corpus), start=1):
        if limit is not None and i > limit:
            break
        records.append(record)
        if i % 250 == 0:
            typer.echo(f"scrubbed {i} documents...")

    report = scan_corpus(records, engine=engine)
    elapsed = time.perf_counter() - started
    payload = report.to_dict()
    payload.update(
        {
            "corpus": str(corpus),
            "n_documents_in_corpus": manifest.get("n_documents"),
            "documents_audited": report.total_documents,
            "elapsed_sec": round(elapsed, 3),
            "seed": settings.seed,
            "engine_selection_reason": engine.selection_reason,
            "phi_leak_rate": report.phi_leak_rate,
            "per_entity_recall": report.per_entity_recall,
            "backend_name": report.backend_name,
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "backend": report.backend_name,
                "phi_leak_rate": report.phi_leak_rate,
                "leaked_documents": report.leaked_documents,
                "documents": report.total_documents,
                "per_entity_recall": report.per_entity_recall,
                "artifact": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
