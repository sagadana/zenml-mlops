# Base image for all ZenML pipeline steps.
# Each step runs inside a container built from this image when deploying to AWS.
# Build from repo root: docker build -f docker/pipeline/Dockerfile .

FROM python:3.12-slim

# System deps for numba/llvm and data processing
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install official Docker CE CLI
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec first for layer caching
COPY pyproject.toml .
COPY README.md .

# Install dependencies (no dev extras)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e "."

# Install ZenML integrations
RUN zenml integration install aws s3 evidently -y --no-input 2>/dev/null || true

# Copy source code
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

ENTRYPOINT ["python", "run.py"]
