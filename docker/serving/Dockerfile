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
ARG SERVICE="model-serving-${MODEL_NAME}"

FROM python:3.12.11-slim-trixie

USER root

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a dedicated non-root runtime user.
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

# Install serving dependencies only (no ZenML, no Numba)
RUN pip install --no-cache-dir \
    fastapi==0.139.0 \
    "uvicorn[standard]==0.51.0" \
    pydantic==2.13.4 \
    cloudpickle==3.1.2 \
    numpy==2.5.1 \
    pandas==3.0.3 \
    boto3==1.43.48

# Re-declare ARG after FROM so it is available in COPY/ENV layers
ARG WORKFLOW
ARG MODEL_URI
ARG MODEL_NAME
ARG SERVICE

# Copy workflow-specific source 
COPY workflows/${WORKFLOW}/configs/*.py workflows/${WORKFLOW}/configs/
COPY workflows/${WORKFLOW}/models/ workflows/${WORKFLOW}/models/
COPY workflows/${WORKFLOW}/serving/app.py workflows/${WORKFLOW}/serving/app.py

RUN mkdir -p /app/model /app/logs
RUN chown -R app:app /app /home/app

USER app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV WORKFLOW=${WORKFLOW}
ENV MODEL_URI=${MODEL_URI}
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_PATH=/app/model/${MODEL_NAME}.pkl
ENV MODEL_INFERENCE_LOG_PATH=/app/logs/inference.jsonl
ENV MODEL_INFERENCE_LOG_ENABLED=true
ENV SERVICE=${SERVICE}

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["sh", "-c", "uvicorn workflows.${WORKFLOW}.serving.app:app --host 0.0.0.0 --port 8080"]
