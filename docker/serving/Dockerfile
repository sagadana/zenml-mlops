# Shared FastAPI serving image for all workflows.
# Build from repo root:
#   docker build -f docker/serving/Dockerfile --build-arg WORKFLOW=<workflow_name> .
#
# WORKFLOW controls:
#   - which workflow's models/ and serving/app.py are copied in
#   - the uvicorn module path at runtime

ARG WORKFLOW
ARG MODEL_URI
ARG MODEL_NAME
ARG SERVICE="model-serving"

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install serving dependencies only (no ZenML, no Dask, no Numba)
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    pydantic==2.10.4 \
    cloudpickle==3.1.0 \
    "numpy>=1.26,<2.0" \
    pandas==2.2.3

# Re-declare ARG after FROM so it is available in COPY/ENV layers
ARG WORKFLOW
ARG MODEL_URI
ARG MODEL_NAME
ARG SERVICE

# Copy workflow-specific source 
COPY workflows/${WORKFLOW}/configs/ workflows/${WORKFLOW}/configs/
COPY workflows/${WORKFLOW}/models/ workflows/${WORKFLOW}/models/
COPY workflows/${WORKFLOW}/serving/app.py workflows/${WORKFLOW}/serving/app.py

# Local fallback model artefacts (used when MODEL_URI is not an S3 URI)
COPY model/ model/

# Resolve model artefact at build time.
# - s3://...  -> download from S3 into /app/model/<MODEL_NAME>.pkl
# - otherwise -> copy local file path into /app/model/<MODEL_NAME>.pkl
RUN test -n "${MODEL_URI}" && test -n "${MODEL_NAME}" && mkdir -p /app/model && \
    if echo "${MODEL_URI}" | grep -qE '^s3://'; then \
    aws s3 cp "${MODEL_URI}" "/app/model/${MODEL_NAME}.pkl"; \
    elif [ -f "${MODEL_URI}" ]; then \
    cp "${MODEL_URI}" "/app/model/${MODEL_NAME}.pkl"; \
    elif [ -f "/app/model/$(basename "${MODEL_URI}")" ]; then \
    cp "/app/model/$(basename "${MODEL_URI}")" "/app/model/${MODEL_NAME}.pkl"; \
    else \
    cp "/app/${MODEL_URI}" "/app/model/${MODEL_NAME}.pkl" || \
    (echo "MODEL_URI not found in build context: ${MODEL_URI}" && exit 1); \
    fi

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV WORKFLOW=${WORKFLOW}
ENV MODEL_URI=${MODEL_URI}
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_PATH=/app/model/${MODEL_NAME}.pkl
ENV MODEL_DATA_CAPTURE_PATH=/app/logs/inference.jsonl
ENV MODEL_DATA_CAPTURE_ENABLED=true
ENV SERVICE=${SERVICE}

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["sh", "-c", "uvicorn workflows.${WORKFLOW}.serving.app:app --host 0.0.0.0 --port 8080"]
