"""
steps/model_evaluation/register.py

ZenML step: register_model

Wraps trained ALS factors and encoders into an ALSRecommender, registers it
with the ZenML Model Control Plane, and promotes to 'staging' if it passes
the quality gate (RMSE < threshold).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import Model, get_step_context, log_metadata, step
from zenml.client import Client

from workflows.matrix_factorization.configs import (
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_DESCRIPTION,
    CFG_MODEL_NAME,
)
from workflows.matrix_factorization.materializers.als_recommender_materializer import (
    ALSRecommenderMaterializer,
)
from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logger = logging.getLogger(__name__)


@step(
    enable_cache=False,
    model=Model(
        name=CFG_MODEL_NAME, description=CFG_MODEL_DESCRIPTION, save_models_to_registry=True
    ),
    output_materializers={CFG_MODEL_ARTIFACT_NAME: ALSRecommenderMaterializer},
)
def register_model(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    eval_metrics: dict,
    best_hyperparams: dict,
    rmse_threshold: float = 1.0,
    model_stage: str = "staging",
) -> Annotated[ALSRecommender, CFG_MODEL_ARTIFACT_NAME]:
    """
    Register the trained ALS model with ZenML Model Control Plane.

    Args:
        user_factors: Trained user factor matrix.
        item_factors: Trained item factor matrix.
        user_encoder: pd.Series mapping raw userId → dense index.
        item_encoder: pd.Series mapping raw movieId → dense index.
        eval_metrics: Evaluation metrics dict from compute_metrics step.
        best_hyperparams: Hyperparameters used for training.
        rmse_threshold: Maximum RMSE to promote model to 'staging'.
        model_stage: ZenML model stage to register the trained model ("staging" or "production").
    Returns:
        tuple of (passed: bool, model_artifact: ALSRecommender)
    """
    rank = int(best_hyperparams.get("rank", user_factors.shape[1]))
    regularization = float(best_hyperparams.get("regularization", 0.01))
    alpha = float(best_hyperparams.get("alpha", 1.0))
    n_iter = int(best_hyperparams.get("n_iter", 15))
    passed = False

    # Determine model version from ZenML context
    try:
        ctx = get_step_context()
        model_version = str(ctx.model.version)
    except Exception:
        model_version = uuid.uuid5(
            uuid.NAMESPACE_DNS, f"{CFG_MODEL_NAME}-{datetime.now(UTC).isoformat()}"
        ).hex[:8]

    model = ALSRecommender(
        user_factors=user_factors.astype(np.float32),
        item_factors=item_factors.astype(np.float32),
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        rank=rank,
        regularization=regularization,
        alpha=alpha,
        n_iter=n_iter,
        model_version=model_version,
    )

    rmse = float(eval_metrics.get("rmse", float("inf")))
    if rmse < rmse_threshold:
        passed = True
        logger.info(
            "Quality gate for model (%s) PASSED (RMSE=%.4f < threshold=%.4f). Promoting to 'staging'.",
            model_version,
            rmse,
            rmse_threshold,
        )

        # Register model with ZenML Model Control Plane and promote to 'staging'
        try:
            ctx = get_step_context()
            ctx.model.set_stage(model_stage, force=True)
        except Exception as exc:
            logger.warning("Could not promote model to staging: %s", exc)
    else:
        logger.warning(
            "Quality gate for model (%s) FAILED (RMSE=%.4f >= threshold=%.4f). Model NOT promoted.",
            model_version,
            rmse,
            rmse_threshold,
        )

    # Log metadata to ZenML model version
    try:
        ctx = get_step_context()
        run_id = ctx.pipeline_run.id

        log_metadata(
            metadata={
                "n_users": model.n_users,
                "n_items": model.n_items,
                "rmse_threshold": rmse_threshold,
                "rmse": rmse,
                "quality_gate_passed": passed,
                **eval_metrics,
                **best_hyperparams,
            },
            run_id_name_or_prefix=str(run_id),
            step_name=ctx.step_name,
        )

    except Exception as exc:
        logger.warning("Metadata logging skipped: %s", exc)

    logger.info("Model (%s) registered = %s: ", model, passed)
    return model
