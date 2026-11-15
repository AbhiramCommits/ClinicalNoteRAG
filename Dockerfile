# Multi-stage image for the ClinicalNoteRAG FastAPI service.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel

# Install a CPU-only torch wheel before the project to avoid pulling CUDA libs.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /build
COPY pyproject.toml README.md ./
COPY clinicalnoterag ./clinicalnoterag
RUN pip install .

FROM python:3.11-slim AS runtime

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app clinicalnoterag /app/clinicalnoterag
COPY --chown=app:app pyproject.toml README.md /app/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "clinicalnoterag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
