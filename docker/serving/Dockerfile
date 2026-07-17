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

FROM python:3.13-slim

USER root

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a dedicated non-root runtime user for the pipeline
ENV HOME=/home/app
RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir ${HOME} app && \
    mkdir -p ${HOME}/.config && \
    # Create directories for model and logs
    mkdir -p /app/model /app/logs && \
    # Set ownership of /app and ${HOME} to the non-root user
    chown -R app:app /app ${HOME}

# Set the user to the non-root user
USER app

# Set the PATH environment variable to include the user's local bin directory
ENV PATH="${HOME}/.local/bin:${PATH}"

# Install serving dependencies only (no ZenML, no Numba)
RUN pip install --no-cache-dir \
    fastapi==0.139.0 \
    "uvicorn[standard]==0.51.0" \
    pydantic==2.13.4 \
    cloudpickle==3.1.2 \
    numpy==2.5.1 \
    pandas==3.0.3 \
    boto3==1.43.48 \
    psutil==7.1.0

# Re-declare ARG after FROM so it is available in COPY/ENV layers
ARG WORKFLOW
ARG MODEL_URI
ARG MODEL_NAME
ARG SERVICE

# Copy workflow-specific source 
COPY workflows/${WORKFLOW}/configs/*.py workflows/${WORKFLOW}/configs/
COPY workflows/${WORKFLOW}/models/ workflows/${WORKFLOW}/models/
COPY workflows/${WORKFLOW}/serving/app.py workflows/${WORKFLOW}/serving/app.py

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
