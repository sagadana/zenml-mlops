"""
steps/model_evaluation/register.py

ZenML step: register_model

Wraps trained ALS factors and encoders into an ALSRecommender and promotes the
candidate in the ZenML Model Control Plane when its external quality check passes.

Note: RMSE is logged as informational metadata but is NOT part of the quality
gate — the implicit ALS model optimises for preference ranking, not rating
prediction, so RMSE against scaled ratings is not a reliable promotion signal.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import Model, get_step_context, log_metadata, step
from zenml.enums import ModelStages

from workflows.matrix_factorization.configs import (
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_DESCRIPTION,
    CFG_MODEL_NAME,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.materializers.als_recommender_materializer import (
    ALSRecommenderMaterializer,
)
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    Hyperparameters,
    ModelMetrics,
    load_recommender_class,
)

logger = logging.getLogger(__name__)

MODEL = Model(
    name=CFG_MODEL_NAME,
    description=CFG_MODEL_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "movie_recommender"],
    save_models_to_registry=True,
)


@step(
    enable_cache=False,
    model=MODEL,  # Configure model produced by this step
    output_materializers={CFG_MODEL_ARTIFACT_NAME: ALSRecommenderMaterializer},
)
def register_model(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    best_hyperparams: Hyperparameters,
    eval_metrics: dict,
    quality_check_passed: bool,
    model_stage: ModelStages = ModelStages.STAGING,
    recommender_class_name: str = "workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender",
) -> Annotated[BaseRecommender, CFG_MODEL_ARTIFACT_NAME]:
    """
    Register the trained recommender model with ZenML Model Control Plane.

    Args:
        user_factors: Trained user factor matrix.
        item_factors: Trained item factor matrix.
        user_encoder: pd.Series mapping raw userId → dense index.
        item_encoder: pd.Series mapping raw movieId → dense index.
        best_hyperparams: Hyperparameters used for training.
        eval_metrics: Metrics computed on the held-out evaluation dataset.
        quality_check_passed: Whether the model passed absolute and regression checks.
        model_stage: ZenML model stage to register the trained model ("staging" or "production").
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass to instantiate.
    Returns:
        Registered BaseRecommender subclass artifact.
    """
    factors = best_hyperparams.factors
    regularization = best_hyperparams.regularization
    alpha = best_hyperparams.alpha
    n_iter = best_hyperparams.n_iter

    # Determine model version from ZenML context
    version_suffix = "1-alpha"  # Default suffix before promotion;
    try:
        ctx = get_step_context()
        model_name = ctx.model.name
        model_version = f"{ctx.model.version}.{version_suffix}"
    except Exception:
        model_name = CFG_MODEL_NAME
        model_version = f"{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.{version_suffix}"

    # Resolve recommender class
    recommender_cls: type[BaseRecommender] = load_recommender_class(recommender_class_name)

    metrics = ModelMetrics(
        k=int(eval_metrics["top_k"]),
        rmse=float(eval_metrics["rmse"]),
        precision_at_k=float(eval_metrics["precision_at_k"]),
        recall_at_k=float(eval_metrics["recall_at_k"]),
        ndcg_at_k=float(eval_metrics["ndcg_at_k"]),
    )

    promoted = False
    if quality_check_passed:
        try:
            ctx = get_step_context()
            z_model = ctx.model
            z_model.set_stage(model_stage, force=True)
            model_version = str(z_model.version)
            promoted = True
        except Exception as exc:
            logger.warning("Could not promote model to '%s': %s", model_stage, exc)
    else:
        logger.warning(
            "Skipping stage registration for model %s at '%s' because the quality check failed",
            model_version,
            model_stage,
        )

    # Wrap trained factors and encoders into a BaseRecommender subclass instance
    model = recommender_cls(
        name=model_name,
        version=model_version,
        promoted=promoted,
        user_factors=user_factors.astype(np.float32),
        item_factors=item_factors.astype(np.float32),
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        params=Hyperparameters(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            n_iter=n_iter,
        ),
        metrics=metrics,
    )

    # Log metadata to ZenML model version
    try:
        ctx = get_step_context()
        run_id = ctx.pipeline_run.id
        metadata = {
            "n_users": model.n_users,
            "n_items": model.n_items,
            "model_stage": str(model_stage),
            "model_version": model_version,
            "model_class": recommender_class_name,
            "quality_check_passed": quality_check_passed,
            "metrics": metrics.model_dump(),
            "hyperparameters": best_hyperparams.model_dump(),
        }
        log_metadata(metadata=metadata, infer_model=True)
        log_metadata(
            metadata=metadata,
            run_id_name_or_prefix=str(run_id),
            step_name=ctx.step_name,
        )

    except Exception as exc:
        logger.warning("Metadata logging skipped: %s", exc)

    logger.info(
        "Model: %s\nPromoted to '%s': %s\n",
        model,
        model_stage,
        "YES" if promoted else "NO",
    )

    return model
