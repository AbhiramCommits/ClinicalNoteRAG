"""FAISS and pgvector contract tests, including mandatory patient isolation."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from clinicalnoterag.config import get_settings
from clinicalnoterag.vectorstore.base import normalize_filters, validate_vector
from clinicalnoterag.vectorstore.faiss_store import FaissStore
from clinicalnoterag.vectorstore.pgvector_store import PgVectorStore
from tests.conftest import make_records


def _pg_reason() -> str | None:
    """Return None when pgvector is reachable, else a recorded skip reason."""
    import psycopg

    try:
        conn = psycopg.connect(get_settings().postgres_dsn, connect_timeout=2)
        conn.close()
        return None
    except Exception as exc:  # noqa: BLE001 - this is the skip path
        return f"Docker/pgvector unavailable: {type(exc).__name__}: {exc}"


@pytest.fixture(params=["faiss", "pgvector"])
def store(request, tmp_path: Path):
    """Contract fixture exercised against both production stores."""
    if request.param == "faiss":
        instance = FaissStore(
            dim=8,
            index_path=tmp_path / "index.faiss",
            sqlite_path=tmp_path / "index.sqlite3",
            fresh=True,
        )
        yield instance
        instance.close()
        return
    reason = _pg_reason()
    if reason:
        pytest.skip(reason)
    table = f"contract_{uuid.uuid4().hex[:10]}"
    instance = PgVectorStore(get_settings().postgres_dsn, dim=8, table=table, fresh=True)
    try:
        yield instance
    finally:
        instance.close()
        import psycopg

        with (
            psycopg.connect(get_settings().postgres_dsn, autocommit=True) as conn,
            conn.cursor() as cur,
        ):
            cur.execute(f"DROP TABLE IF EXISTS {table}")


def _q(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.linalg.norm(array)


def test_upsert_is_idempotent(store):
    records = make_records()
    assert store.upsert(records) == 3
    assert store.upsert(records) == 0
    assert store.count() == 3


def test_search_orders_by_similarity(store):
    store.upsert(make_records())
    hits = store.search(_q([1, 0, 0, 0, 0, 0, 0, 0]), k=2)
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > hits[1].score


def test_patient_filter(store):
    store.upsert(make_records())
    hits = store.search(_q([0, 1, 0, 0, 0, 0, 0, 0]), k=10, filters={"patient_id": "PAT-0001"})
    assert hits
    assert {hit.patient_id for hit in hits} == {"PAT-0001"}


def test_no_cross_patient_leakage(store):
    """Mandatory isolation test: a patient filter must never return another patient."""
    store.upsert(make_records())
    hits = store.search(_q([0, 1, 0, 0, 0, 0, 0, 0]), k=10, filters={"patient_id": "PAT-0001"})
    assert hits
    assert all(hit.patient_id == "PAT-0001" for hit in hits)
    assert all(hit.chunk_id != "c2" for hit in hits)


def test_code_filter(store):
    store.upsert(make_records())
    hits = store.search(
        _q([0, 1, 0, 0, 0, 0, 0, 0]),
        k=10,
        filters={"code_system": "CPT", "code": "99213"},
    )
    assert [hit.chunk_id for hit in hits] == ["c2"]


def test_service_date_range_filter(store):
    store.upsert(make_records())
    hits = store.search(
        _q([1, 0, 0, 0, 0, 0, 0, 0]),
        k=10,
        filters={"service_date_gte": "2024-02-01"},
    )
    assert {hit.chunk_id for hit in hits} == {"c2", "c3"}


def test_denied_filter(store):
    store.upsert(make_records())
    hits = store.search(_q([0, 0, 1, 0, 0, 0, 0, 0]), k=10, filters={"denied": True})
    assert [hit.chunk_id for hit in hits] == ["c2"]


def test_all_records_roundtrip(store):
    store.upsert(make_records())
    records = store.all_records(include_vectors=True)
    assert len(records) == 3
    assert all(record.vector is not None and record.vector.shape == (8,) for record in records)


def test_unknown_filter_is_rejected(store):
    store.upsert(make_records())
    with pytest.raises(ValueError):
        store.search(_q([1, 0, 0, 0, 0, 0, 0, 0]), k=1, filters={"doctor_name": "x"})
    with pytest.raises(ValueError):
        normalize_filters({"not_a_filter": 1})


def test_query_dimension_mismatch_is_rejected(store):
    with pytest.raises(ValueError):
        store.search(np.ones(4, dtype=np.float32), k=1)
    with pytest.raises(ValueError):
        validate_vector(np.ones(4, dtype=np.float32), 8)


def test_faiss_persistence(tmp_path: Path):
    path = tmp_path / "persist.faiss"
    sqlite_path = tmp_path / "persist.sqlite3"
    first = FaissStore(dim=8, index_path=path, sqlite_path=sqlite_path, fresh=True)
    assert first.upsert(make_records()) == 3
    first.close()
    second = FaissStore(dim=8, index_path=path, sqlite_path=sqlite_path, fresh=False)
    try:
        assert second.count() == 3
        hits = second.search(_q([1, 0, 0, 0, 0, 0, 0, 0]), k=1)
        assert hits[0].chunk_id == "c1"
    finally:
        second.close()
