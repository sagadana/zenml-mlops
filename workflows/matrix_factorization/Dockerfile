# Base image for all ZenML pipeline steps.
# Each step runs inside a container built from this image when deploying to AWS.

FROM python:3.12-slim

# System deps for numba/llvm and data processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec first for layer caching
COPY pyproject.toml .
COPY README.md .

# Install dependencies (no dev extras)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e "."

# Install ZenML integrations
RUN zenml integration install aws s3 mlflow -y --no-input 2>/dev/null || true

# Copy source code
COPY . .

# Pre-compile Numba cache at image build time to avoid cold-start JIT latency
RUN python -c "from workflows.matrix_factorization.utils.als_numba import warmup_jit; warmup_jit()" 2>/dev/null || true

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run.py"]
