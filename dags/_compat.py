"""Airflow compatibility shim.

The DAG file must import cleanly in environments that do not install Apache
Airflow (for example, the offline smoke job). When Airflow is installed this
module re-exports the real classes; otherwise it provides a tiny dependency-free
stand-in with the same context-manager and task-chaining surface used here.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised only when Airflow is installed
    from airflow import DAG  # type: ignore[no-redef]
    from airflow.operators.python import PythonOperator  # type: ignore[no-redef]

    HAS_AIRFLOW = True
except ImportError:  # pragma: no cover - the path this repo runs by default
    HAS_AIRFLOW = False

    class DAG:  # type: ignore[no-redef]
        """Minimal stand-in for ``airflow.DAG`` used by the local import test."""

        def __init__(
            self,
            dag_id: str,
            schedule: str | None = None,
            start_date: Any = None,
            catchup: bool = False,
            tags: list[str] | None = None,
            description: str | None = None,
            default_args: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self.dag_id = dag_id
            self.schedule = schedule
            self.start_date = start_date
            self.catchup = catchup
            self.tags = tags or []
            self.description = description
            self.default_args = default_args or {}
            self.extra = kwargs
            self.tasks: list[PythonOperator] = []

        def __enter__(self) -> DAG:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    class PythonOperator:  # type: ignore[no-redef]
        """Minimal stand-in for ``airflow.operators.python.PythonOperator``."""

        def __init__(
            self,
            task_id: str,
            python_callable: Any,
            dag: DAG | None = None,
            op_kwargs: dict[str, Any] | None = None,
            retries: int = 0,
            **kwargs: Any,
        ) -> None:
            self.task_id = task_id
            self.python_callable = python_callable
            self.dag = dag
            self.op_kwargs = op_kwargs or {}
            self.retries = retries
            self.kwargs = kwargs
            self.downstream: list[PythonOperator] = []
            if dag is not None:
                dag.tasks.append(self)

        def __rshift__(self, other: PythonOperator) -> PythonOperator:
            self.downstream.append(other)
            return other

        def __rrshift__(self, other: PythonOperator) -> PythonOperator:
            other.downstream.append(self)
            return self


__all__ = ["DAG", "HAS_AIRFLOW", "PythonOperator"]
