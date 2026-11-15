#!/usr/bin/env python
"""Run the same refresh tasks as the Airflow DAG, locally, without a scheduler."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from clinicalnoterag.config import get_settings
from clinicalnoterag.pipeline import refresh
from clinicalnoterag.seeds import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--patients", type=int, default=300)
    args = parser.parse_args()

    settings = get_settings()
    set_global_seed(settings.seed)
    started = time.perf_counter()
    completed: list[str] = []
    try:
        steps = [
            (
                "generate_or_sync_corpus",
                lambda: refresh.generate_or_sync_corpus(args.corpus, args.patients),
            ),
            ("deid_scan", lambda: refresh.deid_scan(args.corpus)),
            ("chunk_and_embed", lambda: refresh.chunk_and_embed(args.corpus)),
            ("upsert", refresh.upsert),
            ("smoke_query", refresh.smoke_query),
        ]
        for name, fn in steps:
            print(f"[run_refresh] {name} ...", flush=True)
            result = fn()
            completed.append(name)
            print(f"[run_refresh] {name}: {json.dumps(result)}", flush=True)
        metrics = refresh.publish_metrics()
        metrics["completed_tasks"] = completed
        metrics["total_wall_sec"] = round(time.perf_counter() - started, 4)
        metrics["status"] = "success"
        Path(settings.artifact_dir / "refresh_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": "success",
                    "tasks": completed,
                    "artifact": "artifacts/refresh_metrics.json",
                }
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - record the real failure and exit non-zero
        metrics = {
            "dag_id": refresh.DAG_ID,
            "status": "failed",
            "failed_at": datetime.now(UTC).isoformat(),
            "completed_tasks": completed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "total_wall_sec": round(time.perf_counter() - started, 4),
        }
        Path(settings.artifact_dir / "refresh_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
