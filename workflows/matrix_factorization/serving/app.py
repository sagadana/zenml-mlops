"""
serving/app.py

FastAPI recommendation service.

Endpoints:
  GET  /health        — liveness probe
  POST /recommend     — top-K recommendations for a user

The ALSRecommender model is loaded once at startup from the path specified
by the MODEL_PATH environment variable (defaults to model/als_recommender.pkl).

Inference logs (JSON lines) are written to LOG_PATH for drift monitoring.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import cloudpickle
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model/als_recommender.pkl")
LOG_PATH = os.environ.get("LOG_PATH", "logs/inference.jsonl")
LOG_ENABLED = os.environ.get("LOG_ENABLED", "true").lower() == "true"

# ── Global model handle ───────────────────────────────────────────────────────
_model: ALSRecommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    global _model
    logger.info("Loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        _model = cloudpickle.load(f)
    logger.info("Model loaded: %s", _model)
    yield
    _model = None
    logger.info("Model unloaded")


app = FastAPI(
    title="AIPS Recommendations — ALS Movie Recommender",
    version="1.0.0",
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
async def recommend(request: RecommendRequest) -> RecommendResponse:
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

    if LOG_ENABLED:
        _log_inference(request.user_id, request.top_k, latency_ms, len(recs))

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=[RecommendationItem(**r) for r in recs],
        model_version=_model.model_version,
        latency_ms=round(latency_ms, 2),
    )


# ── Inference logging ─────────────────────────────────────────────────────────


def _log_inference(user_id: int, top_k: int, latency_ms: float, items_returned: int) -> None:
    """Append one JSON line to the inference log for drift monitoring."""
    try:
        log_entry = json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "user_id": user_id,
                "top_k": top_k,
                "latency_ms": round(latency_ms, 2),
                "items_returned": items_returned,
            }
        )
        log_file = Path(LOG_PATH)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(log_entry + "\n")
    except Exception as exc:
        logger.warning("Failed to write inference log: %s", exc)
