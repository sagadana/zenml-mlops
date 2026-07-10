"""
serving/app.py

FastAPI recommendation service.

Endpoints:
  GET  /health        — liveness probe
  POST /recommend     — top-K recommendations for a user

The ALSRecommender model is loaded once at startup from the path specified
by the MODEL_PATH environment variable (defaults to model/als_recommender.pkl).

Inference logs (JSON lines) are written to MODEL_DATA_CAPTURE_PATH for drift monitoring.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import TextIO
from urllib.parse import urlparse

import cloudpickle
import psutil
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from workflows.matrix_factorization.configs import (
    CFG_INFERENCE_LOGS_EXT,
    CFG_MODEL_NAME,
    CFG_RECS_LOG_FIELD_NAMES,
)
from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", CFG_MODEL_NAME)
MODEL_PATH = os.environ.get("MODEL_PATH", f"model/${MODEL_NAME}.pkl")
MODEL_DATA_CAPTURE_PATH = os.environ.get(
    "MODEL_DATA_CAPTURE_PATH", f"/app/logs/inference.${CFG_INFERENCE_LOGS_EXT}"
)
MODEL_DATA_CAPTURE_ENABLED = os.environ.get("MODEL_DATA_CAPTURE_ENABLED", "true").lower() == "true"
MODEL_DATA_CAPTURE_BATCH_SIZE = 10

SERVICE = os.environ.get("SERVICE", f"{MODEL_NAME}_serving")
VERSION = os.environ.get("VERSION", "1.0.0")

# ── Global model handle ───────────────────────────────────────────────────────
_model: ALSRecommender | None = None
_inference_log_handle: TextIO | None = None
_inference_log_lock = Lock()
_inference_log_buffer: list[str] = []
_inference_log_batch_seq = 0
_inference_log_is_s3 = MODEL_DATA_CAPTURE_PATH.startswith("s3://")
_inference_s3_client = None


# ── Utilities for inference logging ───────────────────────────────────────────────


def _open_inference_log_handle() -> None:
    """Open and keep a line-buffered inference log handle ready for streaming writes."""
    global _inference_log_handle

    if not MODEL_DATA_CAPTURE_ENABLED or _inference_log_is_s3 or _inference_log_handle is not None:
        return

    log_file = Path(MODEL_DATA_CAPTURE_PATH)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _inference_log_handle = log_file.open("a", encoding="utf-8", buffering=1)


def _close_inference_log_handle() -> None:
    """Close the shared inference log file handle."""
    global _inference_log_handle

    if _inference_log_handle is None:
        return

    _inference_log_handle.close()
    _inference_log_handle = None


def _get_s3_client():
    global _inference_s3_client
    if _inference_s3_client is None:
        import boto3

        _inference_s3_client = boto3.client("s3")
    return _inference_s3_client


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _build_s3_batch_key(base_key: str, batch_seq: int) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")

    prefix = base_key
    if not prefix:
        prefix = "logs"
    elif base_key.endswith("/"):
        prefix = f"{base_key.rstrip('/')}"

    return f"{prefix}/inference.batch-{timestamp}-{batch_seq:06d}.${CFG_INFERENCE_LOGS_EXT}"


def _flush_inference_logs(force: bool = False) -> None:
    """Flush grouped inference logs to local file or S3."""
    global _inference_log_batch_seq

    if not MODEL_DATA_CAPTURE_ENABLED:
        return

    with _inference_log_lock:
        if not _inference_log_buffer:
            return
        if not force and len(_inference_log_buffer) < MODEL_DATA_CAPTURE_BATCH_SIZE:
            return
        lines_to_flush = _inference_log_buffer[:]
        _inference_log_buffer.clear()
        batch_seq = _inference_log_batch_seq
        _inference_log_batch_seq += 1

    payload = "\n".join(lines_to_flush) + "\n"

    # Flush to S3 if configured
    if _inference_log_is_s3:
        try:
            bucket, base_key = _parse_s3_path(MODEL_DATA_CAPTURE_PATH)
            if not bucket:
                raise ValueError(
                    f"Invalid S3 capture path '{MODEL_DATA_CAPTURE_PATH}'. Expected s3://bucket/key."
                )

            key = _build_s3_batch_key(base_key, batch_seq)
            _get_s3_client().put_object(
                Bucket=bucket,
                Key=key,
                Body=payload.encode("utf-8"),
                ContentType="application/x-ndjson",
            )
            return
        except (OSError, ValueError, ImportError, BotoCoreError, ClientError) as exc:
            with _inference_log_lock:
                _inference_log_buffer[:0] = lines_to_flush
            logger.warning("Failed to flush inference logs: %s", exc)
            return

    # Flush to local file if configured
    try:
        _open_inference_log_handle()
        if _inference_log_handle is None:
            raise ValueError("Local inference log file handle is not available.")
        _inference_log_handle.write(payload)
        _inference_log_handle.flush()
    except (OSError, ValueError, ImportError) as exc:
        with _inference_log_lock:
            _inference_log_buffer[:0] = lines_to_flush
        logger.warning("Failed to flush inference logs: %s", exc)


# ── FastAPI app ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    # Load the model once at startup and keep it in memory for inference.
    global _model
    logger.info("Loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        _model = cloudpickle.load(f)

    # Open the inference log handle for streaming writes.
    _open_inference_log_handle()

    logger.info("Model loaded: %s", _model)

    yield

    # Clean up on shutdown
    _flush_inference_logs(force=True)
    _close_inference_log_handle()
    _model = None

    logger.info("Model unloaded")


app = FastAPI(
    title=SERVICE,
    version=VERSION,
    lifespan=lifespan,
)

# ── Request / Response schemas ────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Raw user ID (as in the training dataset)", ge=0)
    top_k: int = Field(10, description="Number of recommendations to return", ge=1, le=200)


class RecommendationItem(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    app_version: str = VERSION

    model_version: str
    n_users: int
    n_items: int
    rank: int

    cpu_percent: float
    memory_percent: float
    disk_percent: float


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness probe."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="ok",
        model_version=_model.model_version,
        n_users=_model.n_users,
        n_items=_model.n_items,
        rank=_model.rank,
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        disk_percent=psutil.disk_usage("/").percent,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    request: RecommendRequest, background_tasks: BackgroundTasks
) -> RecommendResponse:
    """
    Generate top-K movie recommendations for a user.

    Returns recommendations sorted by predicted score descending.
    Unknown user IDs return HTTP 404.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.perf_counter()

    try:
        recs = _model.predict(request.user_id, top_k=request.top_k)
    except KeyError as err:
        raise HTTPException(
            status_code=404,
            detail=f"User ID {request.user_id} not found in training data.",
        ) from err

    latency_ms = (time.perf_counter() - t_start) * 1000

    if MODEL_DATA_CAPTURE_ENABLED:
        background_tasks.add_task(
            _log_inference, request.user_id, request.top_k, latency_ms, len(recs)
        )

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=[RecommendationItem(**r) for r in recs],
        model_version=_model.model_version,
        latency_ms=round(latency_ms, 2),
    )


# ── Inference logging ─────────────────────────────────────────────────────────


# TODO: Consider logging to event stream, notification or database instead of local file for better scalability and reliability.
# E.g Kafka, Kinesis, SNS, DynamoDB, or a managed logging service.
def _log_inference(user_id: int, top_k: int, latency_ms: float, count: int) -> None:
    """Buffer one JSON line and stream grouped logs for drift monitoring."""
    if not MODEL_DATA_CAPTURE_ENABLED:
        return

    try:
        log_entry = json.dumps(
            {
                CFG_RECS_LOG_FIELD_NAMES.TIMESTAMP.value: datetime.now(UTC).isoformat(),
                CFG_RECS_LOG_FIELD_NAMES.USER_ID.value: user_id,
                CFG_RECS_LOG_FIELD_NAMES.TOP_K.value: top_k,
                CFG_RECS_LOG_FIELD_NAMES.LATENCY_MS.value: round(latency_ms, 2),
                CFG_RECS_LOG_FIELD_NAMES.COUNT.value: count,
            }
        )
        with _inference_log_lock:
            _inference_log_buffer.append(log_entry)
            should_flush = len(_inference_log_buffer) >= MODEL_DATA_CAPTURE_BATCH_SIZE

        if should_flush:
            _flush_inference_logs()

    except (OSError, ValueError) as exc:
        logger.warning("Failed to buffer inference log: %s", exc)
