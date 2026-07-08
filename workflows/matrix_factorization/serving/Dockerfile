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
    uvicorn[standard]==0.34.0 \
    pydantic==2.10.4 \
    cloudpickle==3.1.0 \
    numpy>=1.26,\<2.0 \
    pandas==2.2.3

# Copy source code (model file is baked in at build time via build_serving_image step)
COPY models/ models/
COPY serving/app.py serving/app.py
COPY model/ model/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/als_recommender.pkl
ENV LOG_PATH=/app/logs/inference.jsonl
ENV LOG_ENABLED=true

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
