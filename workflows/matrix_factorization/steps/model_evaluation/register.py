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
from zenml import Model, __version__, get_step_context, log_metadata, step
from zenml.cli import ModelRegistryModelMetadata
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
experiment_tracker = Client().active_stack.experiment_tracker


@step(
    enable_cache=False,
    model=Model(
        name=CFG_MODEL_NAME, description=CFG_MODEL_DESCRIPTION, save_models_to_registry=True
    ),
    output_materializers={CFG_MODEL_ARTIFACT_NAME: ALSRecommenderMaterializer},
    experiment_tracker=experiment_tracker.name if experiment_tracker else None,
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


@step(enable_cache=False)
def register_model_external(
    model_artifact: ALSRecommender,
    eval_metrics: dict,
    best_hyperparams: dict,
) -> Annotated[dict | None, "model_info"]:
    """
    Register the trained ALS model with the external MLflow Model Registry.

    Separated from register_model to keep ZenML Model Control Plane registration
    and external registry registration as independent pipeline steps.

    Args:
        model_artifact: The registered ALSRecommender artifact from register_model.
        eval_metrics: Evaluation metrics dict from compute_metrics step.
        best_hyperparams: Hyperparameters used for training.
    """
    client = Client()
    ctx = get_step_context()
    artifact = ctx.model.get_artifact(CFG_MODEL_ARTIFACT_NAME)

    registry = client.active_stack.model_registry
    if registry is None:
        logger.warning(
            "No model registry found in the active stack — skipping external registration."
        )
        return None

    try:
        registry.get_model(name=CFG_MODEL_NAME)
    except Exception:
        registry.register_model(
            name=CFG_MODEL_NAME,
            description=CFG_MODEL_DESCRIPTION,
        )

    model_version = registry.register_model_version(
        name=CFG_MODEL_NAME,
        description=f"Model version registered from ZenML pipeline {ctx.pipeline.name} run {ctx.pipeline_run.name}",
        version=str(ctx.model.version),
        model_source_uri=artifact.uri if artifact else None,
        metadata=ModelRegistryModelMetadata(
            zenml_pipeline_name=ctx.pipeline.name,
            zenml_pipeline_uuid=str(ctx.pipeline.id),
            zenml_pipeline_run_uuid=str(ctx.pipeline_run.id),
            zenml_step_name=ctx.step_name,
            zenml_run_name=ctx.pipeline_run.name,
            zenml_project=str(ctx.pipeline.project.name),
            zenml_version=__version__,
        ),
    )

    logger.info("Model (%s) registered in external MLflow registry.", CFG_MODEL_NAME)
    return {
        "n_users": model_artifact.n_users,
        "n_items": model_artifact.n_items,
        "alpha": model_artifact.alpha,
        "rank": model_artifact.rank,
        "best_hyperparams": best_hyperparams,
        "eval_metrics": eval_metrics,
        "registered_version": model_version.model_dump(mode="json"),
    }
