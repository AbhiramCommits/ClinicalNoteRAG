"""Airflow DAG for the daily clinical RAG embedding refresh.

The tasks call the real pipeline functions in
:mod:`clinicalnoterag.pipeline.refresh`. This file imports cleanly whether or
not Apache Airflow is installed; see ``dags/_compat.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from clinicalnoterag.pipeline import refresh
from dags._compat import DAG, PythonOperator

default_args = {
    "owner": "clinicalnoterag",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=refresh.DAG_ID,
    description="Daily refresh of the synthetic clinical corpus, PHI audit, embeddings and vector index",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["clinical", "rag", "embeddings", "deid"],
    default_args=default_args,
) as dag:
    generate_or_sync_corpus = PythonOperator(
        task_id="generate_or_sync_corpus",
        python_callable=refresh.generate_or_sync_corpus,
    )
    deid_scan = PythonOperator(
        task_id="deid_scan",
        python_callable=refresh.deid_scan,
    )
    chunk_and_embed = PythonOperator(
        task_id="chunk_and_embed",
        python_callable=refresh.chunk_and_embed,
    )
    upsert = PythonOperator(
        task_id="upsert",
        python_callable=refresh.upsert,
    )
    smoke_query = PythonOperator(
        task_id="smoke_query",
        python_callable=refresh.smoke_query,
    )
    publish_metrics = PythonOperator(
        task_id="publish_metrics",
        python_callable=refresh.publish_metrics,
    )

    (
        generate_or_sync_corpus
        >> deid_scan
        >> chunk_and_embed
        >> upsert
        >> smoke_query
        >> publish_metrics
    )
