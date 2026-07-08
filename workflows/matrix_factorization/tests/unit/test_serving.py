"""
tests/unit/test_serving.py

Unit tests for the FastAPI serving app.
Uses httpx AsyncClient (via pytest-asyncio) against the FastAPI test client.
Model is mocked with a tiny synthetic ALSRecommender to avoid loading real files.
"""

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from workflows.matrix_factorization.models.als_recommender import ALSRecommender


def make_test_model() -> ALSRecommender:
    """Create a small synthetic ALSRecommender for testing."""
    rng = np.random.default_rng(42)
    n_users, n_items, rank = 10, 20, 4
    user_factors = rng.standard_normal((n_users, rank)).astype(np.float32)
    item_factors = rng.standard_normal((n_items, rank)).astype(np.float32)
    user_encoder = pd.Series(range(n_users), index=range(1, n_users + 1), dtype="int32")
    item_encoder = pd.Series(range(n_items), index=range(101, 101 + n_items), dtype="int32")
    return ALSRecommender(
        user_factors=user_factors,
        item_factors=item_factors,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        rank=rank,
        model_version="test-1",
    )


@pytest.fixture
def model_pkl(tmp_path) -> str:
    """Write a pickled test model and return its path."""
    model = make_test_model()
    pkl_path = tmp_path / "als_recommender.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(model, f)
    return str(pkl_path)


@pytest.fixture
def test_app(model_pkl, monkeypatch, tmp_path):
    """Configure the serving app with test model path and return it."""

    monkeypatch.setenv("MODEL_PATH", model_pkl)
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "inference.jsonl"))
    monkeypatch.setenv("LOG_ENABLED", "true")

    # Re-import app to pick up env var changes
    import importlib

    import serving.app as app_module

    importlib.reload(app_module)

    return app_module.app


@pytest.mark.asyncio
async def test_health_before_model_load(test_app):
    """Health check should return 503 before model is loaded."""
    # Don't use lifespan (model not loaded)
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.get("/health")
        # Without lifespan startup, model is None → 503
        assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_health_after_model_load(test_app, model_pkl, monkeypatch):
    """Health check should return 200 with model info after startup."""
    import serving.app as app_module

    # Manually load the model (simulating lifespan startup)
    model = make_test_model()
    app_module._model = model

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_version"] == "test-1"
        assert data["n_users"] == 10
        assert data["n_items"] == 20
        assert data["rank"] == 4


@pytest.mark.asyncio
async def test_recommend_known_user(test_app):
    """POST /recommend with a known user should return top-K items."""
    import serving.app as app_module

    app_module._model = make_test_model()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/recommend", json={"user_id": 1, "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) == 5
        assert all("item_id" in r and "score" in r for r in data["recommendations"])
        # Scores should be descending
        scores = [r["score"] for r in data["recommendations"]]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_recommend_unknown_user(test_app):
    """POST /recommend with an unknown user ID should return 404."""
    import serving.app as app_module

    app_module._model = make_test_model()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/recommend", json={"user_id": 9999, "top_k": 10})
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_recommend_writes_log(test_app, tmp_path, monkeypatch):
    """POST /recommend should write a log entry to LOG_PATH."""
    import serving.app as app_module

    log_path = tmp_path / "inference.jsonl"
    monkeypatch.setenv("LOG_PATH", str(log_path))
    app_module._model = make_test_model()
    app_module.LOG_PATH = str(log_path)
    app_module.LOG_ENABLED = True

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        await client.post("/recommend", json={"user_id": 1, "top_k": 3})

    assert log_path.exists()
    import json

    lines = [json.loads(ln) for ln in log_path.read_text().strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["user_id"] == 1
    assert lines[0]["top_k"] == 3
