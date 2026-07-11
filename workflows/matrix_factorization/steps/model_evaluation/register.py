"""
steps/model_evaluation/register.py

ZenML step: register_model

Wraps trained ALS factors and encoders into an ALSRecommender, registers it
with the ZenML Model Control Plane, and promotes to 'staging' if it passes
the quality gate (RMSE < threshold).
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import Model, get_step_context, log_metadata, step
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)

from helpers.checkpointing import clean_run_checkpoints
from workflows.matrix_factorization.configs import CFG_MODEL_ARTIFACT_NAME, CFG_MODEL_NAME
from workflows.matrix_factorization.materializers.als_recommender_materializer import (
    ALSRecommenderMaterializer,
)
from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logger = logging.getLogger(__name__)


@step(
    enable_cache=False,
    output_materializers={CFG_MODEL_ARTIFACT_NAME: ALSRecommenderMaterializer},
    model=Model(name=CFG_MODEL_NAME),
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
    checkpoint_path: str = "./checkpoints",
) -> tuple[bool, Annotated[ALSRecommender, CFG_MODEL_ARTIFACT_NAME]]:
    """
    Register the trained ALS model with ZenML Model Control Plane.

    Also logs and registers a model version in MLflow Model Registry when
    the MLflow experiment tracker/model registry integration is available.

    Args:
        user_factors: Trained user factor matrix.
        item_factors: Trained item factor matrix.
        user_encoder: pd.Series mapping raw userId → dense index.
        item_encoder: pd.Series mapping raw movieId → dense index.
        eval_metrics: Evaluation metrics dict from compute_metrics step.
        best_hyperparams: Hyperparameters used for training.
        rmse_threshold: Maximum RMSE to promote model to 'staging'.
        model_stage: ZenML model stage to register the trained model ("staging" or "production").
        checkpoint_path: Checkpoint base path to clean up after successful registration.

    Returns:
        tuple of (model_registered: bool, model_artifact: ALSRecommender)
    """
    rank = int(best_hyperparams.get("rank", user_factors.shape[1]))
    regularization = float(best_hyperparams.get("regularization", 0.01))
    alpha = float(best_hyperparams.get("alpha", 1.0))
    n_iter = int(best_hyperparams.get("n_iter", 15))
    registered = False

    # Determine model version from ZenML context
    try:
        ctx = get_step_context()
        model_version = str(ctx.model.version)
    except Exception:
        model_version = "unknown"

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

    # Log metadata to ZenML model version
    log_metadata(
        metadata={
            "n_users": model.n_users,
            "n_items": model.n_items,
            **eval_metrics,
            **best_hyperparams,
        },
        infer_model=True,
    )
    log_metadata(
        metadata={
            **eval_metrics,
            **best_hyperparams,
        },
        infer_artifact=True,
    )

    rmse = float(eval_metrics.get("rmse", float("inf")))
    if rmse < rmse_threshold:
        logger.info(
            "Quality gate PASSED (RMSE=%.4f < threshold=%.4f). Promoting to 'staging'.",
            rmse,
            rmse_threshold,
        )
        try:
            # Register model with ZenML Model Control Plane and promote to 'staging'
            ctx = get_step_context()
            ctx.model.set_stage(model_stage, force=True)

            # Register model with MLflow Model Registry if available
            try:
                mlflow_register_model_step(
                    model=model,
                    name=CFG_MODEL_NAME,
                    metadata={
                        "n_users": model.n_users,
                        "n_items": model.n_items,
                        **eval_metrics,
                        **best_hyperparams,
                    },
                )
            except Exception as exc:
                logger.warning("MLflow model registry registration skipped: %s", exc)

        except Exception as exc:
            logger.warning("Could not promote model to staging: %s", exc)
    else:
        logger.warning(
            "Quality gate FAILED (RMSE=%.4f >= threshold=%.4f). Model NOT promoted.",
            rmse,
            rmse_threshold,
        )

    # Clean up checkpoints now that training is complete and model is registered
    try:
        ctx = get_step_context()
        run_id = ctx.pipeline_run.id
        run_checkpoint_path = f"{checkpoint_path}/{run_id}"
        clean_run_checkpoints(run_checkpoint_path)
    except Exception as exc:
        logger.warning("Checkpoint cleanup skipped: %s", exc)

    logger.info("Model (%s) registered = %s: ", model, registered)
    return (
        registered,
        model,
    )
