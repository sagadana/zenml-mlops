"""
serving/app.py

FastAPI prediction service.

Endpoints:
    GET  /health     — liveness probe
    POST /predict    — top-K predictions for a user

The model is loaded once at startup from the path specified
by the MODEL_PATH environment variable.

Inference logs (JSON lines) are written to MODEL_INFERENCE_LOG_PATH for drift monitoring.
"""

from __future__ import annotations

import logging
import os
import shutil
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
)
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    PredictionItem,
    PredictionLog,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", CFG_MODEL_NAME)
MODEL_PATH = os.environ.get("MODEL_PATH", f"model/{MODEL_NAME}.pkl")
MODEL_URI = os.environ.get("MODEL_URI", MODEL_PATH)

MODEL_INFERENCE_LOG_PATH = os.environ.get(
    "MODEL_INFERENCE_LOG_PATH", f"/app/logs/inference.{CFG_INFERENCE_LOGS_EXT}"
)
MODEL_INFERENCE_LOG_ENABLED = os.environ.get("MODEL_INFERENCE_LOG_ENABLED", "true").lower() in [
    "true",
    "1",
    "yes",
]
MODEL_INFERENCE_LOG_BATCH_SIZE = int(os.environ.get("MODEL_INFERENCE_LOG_BATCH_SIZE", 10))

SEAWEEDFS_S3_INTERNAL_ENDPOINT = os.environ.get("SEAWEEDFS_S3_INTERNAL_ENDPOINT")
SEAWEEDFS_ACCESS_KEY_ID = os.environ.get("SEAWEEDFS_ACCESS_KEY_ID")
SEAWEEDFS_SECRET_ACCESS_KEY = os.environ.get("SEAWEEDFS_SECRET_ACCESS_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION")

SERVICE = os.environ.get("SERVICE", f"{MODEL_NAME}_serving")
VERSION = os.environ.get("VERSION", "1.0.0")

# ── Global model handle ───────────────────────────────────────────────────────
_model: BaseRecommender | None = None
_inference_log_handle: TextIO | None = None
_inference_log_lock = Lock()
_inference_log_buffer: list[str] = []
_inference_log_batch_seq = 0
_inference_log_is_s3 = MODEL_INFERENCE_LOG_PATH.startswith("s3://")
_inference_s3_client = None
_model_s3_client = None


# ── Utilities for inference logging ───────────────────────────────────────────────


def _open_inference_log_handle() -> None:
    """Open and keep a line-buffered inference log handle ready for streaming writes."""
    global _inference_log_handle

    if not MODEL_INFERENCE_LOG_ENABLED or _inference_log_is_s3 or _inference_log_handle is not None:
        return

    log_file = Path(MODEL_INFERENCE_LOG_PATH)
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

        client_kwargs: dict[str, str] = {}
        if SEAWEEDFS_S3_INTERNAL_ENDPOINT:
            client_kwargs["endpoint_url"] = SEAWEEDFS_S3_INTERNAL_ENDPOINT

        access_key_id = SEAWEEDFS_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID
        secret_access_key = SEAWEEDFS_SECRET_ACCESS_KEY or AWS_SECRET_ACCESS_KEY
        if access_key_id and secret_access_key:
            client_kwargs["aws_access_key_id"] = access_key_id
            client_kwargs["aws_secret_access_key"] = secret_access_key

        if AWS_SESSION_TOKEN:
            client_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
        if AWS_DEFAULT_REGION:
            client_kwargs["region_name"] = AWS_DEFAULT_REGION

        _inference_s3_client = boto3.client("s3", **client_kwargs)
    return _inference_s3_client


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _get_model_s3_client():
    """Build an S3 client for model download (AWS default or SeaweedFS endpoint)."""
    global _model_s3_client

    if _model_s3_client is not None:
        return _model_s3_client

    import boto3

    client_kwargs: dict[str, str] = {}
    if SEAWEEDFS_S3_INTERNAL_ENDPOINT:
        client_kwargs["endpoint_url"] = SEAWEEDFS_S3_INTERNAL_ENDPOINT

    access_key_id = SEAWEEDFS_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID
    secret_access_key = SEAWEEDFS_SECRET_ACCESS_KEY or AWS_SECRET_ACCESS_KEY
    if access_key_id and secret_access_key:
        client_kwargs["aws_access_key_id"] = access_key_id
        client_kwargs["aws_secret_access_key"] = secret_access_key

    if AWS_SESSION_TOKEN:
        client_kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    if AWS_DEFAULT_REGION:
        client_kwargs["region_name"] = AWS_DEFAULT_REGION

    _model_s3_client = boto3.client("s3", **client_kwargs)

    return _model_s3_client


def _prepare_model_file() -> str:
    """Ensure model is available at MODEL_PATH before loading the server."""
    model_path = Path(MODEL_PATH)
    model_uri = MODEL_URI

    if model_uri.startswith("s3://"):
        bucket, key = _parse_s3_path(model_uri)
        if not bucket or not key:
            raise ValueError(f"Invalid MODEL_URI '{model_uri}'. Expected s3://bucket/key.")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading model from %s to %s", model_uri, model_path)
        _get_model_s3_client().download_file(bucket, key, str(model_path))
        return str(model_path)

    src = Path(model_uri)
    if src.exists():
        if src.resolve() != model_path.resolve():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Copying model from %s to %s", src, model_path)
            shutil.copy2(src, model_path)
        return str(model_path)

    if model_path.exists():
        return str(model_path)

    raise FileNotFoundError(f"Model not found. MODEL_URI='{model_uri}', MODEL_PATH='{model_path}'.")


def _build_s3_batch_key(base_key: str, batch_seq: int) -> str:
    """Build a unique S3 key for the inference log batch."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")

    prefix = base_key
    if not prefix:
        prefix = "logs"
    elif base_key.endswith("/"):
        prefix = f"{base_key.rstrip('/')}"

    return f"{prefix}/inference.batch-{timestamp}-{batch_seq:06d}.{CFG_INFERENCE_LOGS_EXT}"


def _flush_inference_logs(force: bool = False) -> None:
    """Flush grouped inference logs to local file or S3."""
    global _inference_log_batch_seq

    if not MODEL_INFERENCE_LOG_ENABLED:
        return

    with _inference_log_lock:
        if not _inference_log_buffer:
            return
        if not force and len(_inference_log_buffer) < MODEL_INFERENCE_LOG_BATCH_SIZE:
            return
        lines_to_flush = _inference_log_buffer[:]
        _inference_log_buffer.clear()
        batch_seq = _inference_log_batch_seq
        _inference_log_batch_seq += 1

    payload = "\n".join(lines_to_flush) + "\n"

    # Flush to S3 if configured
    if _inference_log_is_s3:
        try:
            bucket, base_key = _parse_s3_path(MODEL_INFERENCE_LOG_PATH)
            if not bucket:
                raise ValueError(
                    f"Invalid S3 capture path '{MODEL_INFERENCE_LOG_PATH}'. Expected s3://bucket/key."
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
    model_file = _prepare_model_file()
    logger.info("Loading model from %s", model_file)
    with open(model_file, "rb") as f:
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


class PredictRequest(BaseModel):
    user_id: int = Field(..., description="Raw user ID (as in the training dataset)", ge=0)
    top_k: int = Field(10, description="Number of predictions to return", ge=1, le=200)


class PredictResponse(BaseModel):
    user_id: int
    predictions: list[PredictionItem]
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
        model_version=_model.version,
        n_users=_model.n_users,
        n_items=_model.n_items,
        rank=_model.params.rank,
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        disk_percent=psutil.disk_usage("/").percent,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    """
    Generate top-K movie predictions for a user.

    Returns predictions sorted by predicted score descending.
    Unknown user IDs return HTTP 404.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.perf_counter()

    try:
        prediction_items = _model.predict(request.user_id, top_k=request.top_k)
    except KeyError as err:
        raise HTTPException(
            status_code=404,
            detail=f"User ID {request.user_id} not found in training data.",
        ) from err

    latency_ms = (time.perf_counter() - t_start) * 1000

    # NOTE: This may not be needed as we don't serve predictions if user_id is unknown, hence no opportunity for drift.
    # But we keep it for now in case we want to log all requests in the future.
    if MODEL_INFERENCE_LOG_ENABLED:
        background_tasks.add_task(
            _log_inference,
            request.user_id,
            request.top_k,
            latency_ms,
            len(prediction_items),
            prediction_items,
            model_name=MODEL_NAME,
            model_version=_model.version,
        )

    return PredictResponse(
        user_id=request.user_id,
        predictions=prediction_items,
        model_version=_model.version,
        latency_ms=round(latency_ms, 2),
    )


# ── Inference logging ─────────────────────────────────────────────────────────


# TODO: Consider logging to event stream, notification or database instead of local file for better scalability and reliability.
# E.g Kafka, Kinesis, SNS, DynamoDB, or a managed logging service.
def _log_inference(
    user_id: int,
    top_k: int,
    latency_ms: float,
    count: int,
    predictions: list[PredictionItem],
    model_name: str = MODEL_NAME,
    model_version: str | None = None,
) -> None:
    """Buffer one JSON line and stream grouped logs for drift monitoring."""
    if not MODEL_INFERENCE_LOG_ENABLED:
        return

    try:
        log_entry = PredictionLog(
            timestamp=datetime.now(UTC).isoformat(),
            user_id=user_id,
            top_k=top_k,
            latency_ms=round(latency_ms, 2),
            count=count,
            predictions=predictions,
            model_name=model_name,
            model_version=model_version or "unknown",
        ).model_dump_json()

        with _inference_log_lock:
            _inference_log_buffer.append(log_entry)
            should_flush = len(_inference_log_buffer) >= MODEL_INFERENCE_LOG_BATCH_SIZE

        if should_flush:
            _flush_inference_logs()

    except (OSError, ValueError) as exc:
        logger.exception("Failed to buffer inference log", exc_info=exc)
