# Base image for all ZenML pipelines.

FROM python:3.13-slim

USER root

# System deps for numba/llvm and data processing
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec first for layer caching
COPY pyproject.toml .

# Create a virtualenv because ensurepip is disabled for the Ubuntu system Python.
RUN python3 -m venv /app/venv && \
    . /app/venv/bin/activate 

ENV PATH="/app/venv/bin:${PATH}"


# Install dependencies (no dev extras)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e "."

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Set UV_SYSTEM_PYTHON to 0 to ensure that uvicorn uses the virtualenv Python interpreter instead of the system Python interpreter. 
# This is important because the system Python interpreter does not have the required dependencies installed.
ENV UV_SYSTEM_PYTHON=0 