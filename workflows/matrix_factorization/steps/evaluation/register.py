"""
steps/model_evaluation/register.py

ZenML step: register_model

Wraps trained ALS factors and encoders into an ALSRecommender, registers it
with the ZenML Model Control Plane, and promotes if it passes
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
    model=MODEL,  # Configure model produced by this step for the pipeline context
    output_materializers={CFG_MODEL_ARTIFACT_NAME: ALSRecommenderMaterializer},
)
def register_model(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    eval_metrics: dict,
    best_hyperparams: Hyperparameters,
    rmse_threshold: float = 1.0,
    precision_at_k_threshold: float = 0.5,
    recall_at_k_threshold: float = 0.5,
    ndcg_at_k_threshold: float = 0.5,
    model_stage: str = "staging",
    recommender_class_name: str = "workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender",
) -> Annotated[BaseRecommender, CFG_MODEL_ARTIFACT_NAME]:
    """
    Register the trained recommender model with ZenML Model Control Plane.

    Args:
        user_factors: Trained user factor matrix.
        item_factors: Trained item factor matrix.
        user_encoder: pd.Series mapping raw userId → dense index.
        item_encoder: pd.Series mapping raw movieId → dense index.
        eval_metrics: Evaluation metrics dict from compute_metrics step.
        best_hyperparams: Hyperparameters used for training.
        rmse_threshold: Maximum RMSE to promote model.
        precision_at_k_threshold: Minimum Precision@K to promote model.
        recall_at_k_threshold: Minimum Recall@K to promote model.
        ndcg_at_k_threshold: Minimum NDCG@K to promote model.
        model_stage: ZenML model stage to register the trained model ("staging" or "production").
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass to instantiate.
    Returns:
        Registered BaseRecommender subclass artifact.
    """
    rank = best_hyperparams.rank
    regularization = best_hyperparams.regularization
    alpha = best_hyperparams.alpha
    n_iter = best_hyperparams.n_iter

    # Determine model version from ZenML context
    try:
        ctx = get_step_context()
        model_version = str(ctx.model.version)
    except Exception:
        model_version = uuid.uuid5(
            uuid.NAMESPACE_DNS, f"{CFG_MODEL_NAME}-{datetime.now(UTC).isoformat()}"
        ).hex[:8]

    # Resolve recommender class
    recommender_cls: type[BaseRecommender] = load_recommender_class(recommender_class_name)

    rmse = float(eval_metrics.get("rmse", float("inf")))
    precision_at_k = float(eval_metrics.get("precision_at_k", float("inf")))
    recall_at_k = float(eval_metrics.get("recall_at_k", float("inf")))
    ndcg_at_k = float(eval_metrics.get("ndcg_at_k", float("inf")))
    passed = (
        rmse <= rmse_threshold
        and recall_at_k >= recall_at_k_threshold
        and precision_at_k >= precision_at_k_threshold
        and ndcg_at_k >= ndcg_at_k_threshold
    )

    if passed:
        logger.info(
            "Quality gate for model (%s) PASSED (RMSE=%.4f <= threshold=%.4f, Precision@K=%.4f >= threshold=%.4f, Recall@K=%.4f >= threshold=%.4f, NDCG@K=%.4f >= threshold=%.4f). Promoting to '%s'.",
            model_version,
            rmse,
            rmse_threshold,
            precision_at_k,
            precision_at_k_threshold,
            recall_at_k,
            recall_at_k_threshold,
            ndcg_at_k,
            ndcg_at_k_threshold,
            model_stage,
        )

        # Promote model to the specified stage in ZenML Model Registry
        try:
            ctx = get_step_context()
            ctx.model.set_stage(model_stage, force=True)
        except Exception as exc:
            logger.warning("Could not promote model to staging: %s", exc)
    else:
        logger.warning(
            "Quality gate for model (%s) FAILED (RMSE=%.4f <= threshold=%.4f, Precision@K=%.4f >= threshold=%.4f, Recall@K=%.4f >= threshold=%.4f, NDCG@K=%.4f >= threshold=%.4f). Not Promoting to '%s'.",
            model_version,
            rmse,
            rmse_threshold,
            precision_at_k,
            precision_at_k_threshold,
            recall_at_k,
            recall_at_k_threshold,
            ndcg_at_k,
            ndcg_at_k_threshold,
            model_stage,
        )

    # Wrap trained factors and encoders into a BaseRecommender subclass instance
    model = recommender_cls(
        user_factors=user_factors.astype(np.float32),
        item_factors=item_factors.astype(np.float32),
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        params=Hyperparameters(
            rank=rank,
            regularization=regularization,
            alpha=alpha,
            n_iter=n_iter,
        ),
        version=model_version,
        promoted=passed,
    )

    # Log metadata to ZenML model version
    try:
        ctx = get_step_context()
        run_id = ctx.pipeline_run.id

        log_metadata(
            metadata={
                "n_users": model.n_users,
                "n_items": model.n_items,
                "model_stage": model_stage,
                "model_version": model_version,
                "model_class": recommender_class_name,
                "rmse_threshold": rmse_threshold,
                "precision_at_k_threshold": precision_at_k_threshold,
                "recall_at_k_threshold": recall_at_k_threshold,
                "ndcg_at_k_threshold": ndcg_at_k_threshold,
                "quality_gate_passed": passed,
                "metrics": eval_metrics,
                "hyperparameters": best_hyperparams.model_dump(),
            },
            run_id_name_or_prefix=str(run_id),
            step_name=ctx.step_name,
        )

    except Exception as exc:
        logger.warning("Metadata logging skipped: %s", exc)

    logger.info("Model (%s) registered = %s: ", model, passed)

    return model
