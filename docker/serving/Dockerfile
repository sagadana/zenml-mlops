# Shared FastAPI serving image for all workflows.
# Build from repo root:
#   docker build -f docker/serving/Dockerfile --build-arg WORKFLOW=<workflow_name> .
#
# WORKFLOW controls:
#   - which workflow's models/ and serving/app.py are copied in
#   - the uvicorn module path at runtime

ARG WORKFLOW=matrix_factorization

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
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

# Copy workflow-specific source (model class definitions + serving app)
COPY workflows/${WORKFLOW}/models/ workflows/${WORKFLOW}/models/
COPY workflows/${WORKFLOW}/serving/app.py workflows/${WORKFLOW}/serving/app.py

# Model artefact — baked in at build time by the build_serving_image step
COPY model/ model/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Persist WORKFLOW at runtime so CMD can reference it
ENV WORKFLOW=${WORKFLOW}
ENV MODEL_PATH=/app/model/${WORKFLOW}_model.pkl
ENV LOG_PATH=/app/logs/inference.jsonl
ENV LOG_ENABLED=true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Shell form so ${WORKFLOW} is expanded from ENV at container start
CMD uvicorn workflows.${WORKFLOW}.serving.app:app --host 0.0.0.0 --port 8080
