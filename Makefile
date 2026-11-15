SHELL := /bin/bash
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
PATIENTS ?= 300
PORT ?= 8000
CORPUS ?= data/corpus

.PHONY: setup gen sample deid-audit ingest refresh serve eval train compare smoke test lint fmt reproduce docker-up clean

setup:
	python3.11 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip setuptools wheel
	.venv/bin/python -m pip install -e ".[dev,ml]"

gen:
	$(PYTHON) -m clinicalnoterag gen-corpus --patients $(PATIENTS) --out $(CORPUS)

sample:
	$(PYTHON) -m clinicalnoterag gen-sample --patients 3 --out data/sample

deid-audit:
	$(PYTHON) -m clinicalnoterag deid-audit --corpus $(CORPUS) --out artifacts/deid_audit.json

ingest:
	$(PYTHON) -m clinicalnoterag ingest --corpus $(CORPUS) --store faiss --out artifacts/ingest_report.json

refresh:
	$(PYTHON) scripts/run_refresh.py --corpus $(CORPUS)

serve:
	$(PYTHON) -m uvicorn clinicalnoterag.api.app:app --host 0.0.0.0 --port $(PORT)

eval:
	$(PYTHON) -m clinicalnoterag eval --system rag-baseline --k 5

train:
	$(PYTHON) -m clinicalnoterag train --max-steps 12
	$(PYTHON) -m clinicalnoterag compare-systems --k 5

compare:
	$(PYTHON) -m clinicalnoterag compare-systems --k 5

smoke:
	KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 EMBEDDING_BACKEND=hashing LLM_BACKEND=extractive HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) scripts/smoke_pipeline.py

test:
	mkdir -p artifacts
	$(PYTHON) -m pytest -q --cov=clinicalnoterag --cov-report=term --cov-report=json:artifacts/coverage.json

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

fmt:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

reproduce:
	@if [ ! -x .venv/bin/python ]; then \
		python3.11 -m venv .venv; \
		.venv/bin/python -m pip install --upgrade pip setuptools wheel; \
		.venv/bin/python -m pip install -e ".[dev,ml]"; \
	fi
	$(MAKE) gen deid-audit ingest refresh eval train compare smoke test PYTHON=.venv/bin/python

docker-up:
	docker compose up -d db

clean:
	rm -rf artifacts data/corpus data/smoke_corpus data/benchmark data/sft .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
