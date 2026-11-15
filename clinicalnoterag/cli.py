"""ClinicalNoteRAG command-line interface."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer

from clinicalnoterag import __version__
from clinicalnoterag.config import get_settings
from clinicalnoterag.data.synthea_gen import generate_corpus
from clinicalnoterag.deid.audit import scan_corpus
from clinicalnoterag.deid.engine import DeidEngine
from clinicalnoterag.embeddings.backends import get_embedding_backend
from clinicalnoterag.pipeline.ingest import run_ingest
from clinicalnoterag.seeds import set_global_seed
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
