"""
steps/model_evaluation/register.py

ZenML step: register_model

Wraps trained ALS factors and encoders into an ALSRecommender, registers it
with the ZenML Model Control Plane, and promotes if it passes
the quality gate (ranking metrics: Precision@K, Recall@K, NDCG@K).

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
from zenml.client import Client

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
    EpochStates,
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


def _fetch_previous_model_metrics(
    model_name: str,
    artifact_name: str,
    stage: str,
) -> ModelMetrics | None:
    """
    Load ranking metrics from the model currently at *stage* in the ZenML
    Model Control Plane.

    Returns None when no model is deployed at that stage yet (first run) or
    when the artifact cannot be loaded, so the caller can skip the comparison
    and promote unconditionally.
    """
    client = Client()
    try:
        prev_version = client.get_model_version(
            model_name_or_id=model_name,
            model_version_name_or_number_or_id=stage,
        )
    except Exception:
        # No model version exists at this stage — first deployment.
        return None

    try:
        artifact_response = prev_version.get_artifact(name=artifact_name)
        if artifact_response is None:
            return None
        prev_recommender: BaseRecommender = artifact_response.load()
        return prev_recommender.metrics
    except Exception as exc:
        logger.warning(
            "Could not load artifact '%s' from previous '%s' model: %s. "
            "Skipping metric comparison.",
            artifact_name,
            stage,
            exc,
        )
        return None


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
    training_states: EpochStates,
    best_hyperparams: Hyperparameters,
    precision_at_k_threshold: float = 0.1,
    recall_at_k_threshold: float = 0.1,
    ndcg_at_k_threshold: float = 0.1,
    model_stage: str = "staging",
    force_promote: bool = False,
    recommender_class_name: str = "workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender",
) -> Annotated[BaseRecommender, CFG_MODEL_ARTIFACT_NAME]:
    """
    Register the trained recommender model with ZenML Model Control Plane.

    Args:
        user_factors: Trained user factor matrix.
        item_factors: Trained item factor matrix.
        user_encoder: pd.Series mapping raw userId → dense index.
        item_encoder: pd.Series mapping raw movieId → dense index.
        training_states: Epoch states from train_als.
        best_hyperparams: Hyperparameters used for training.
        precision_at_k_threshold: Minimum Precision@K to promote model.
        recall_at_k_threshold: Minimum Recall@K to promote model.
        ndcg_at_k_threshold: Minimum NDCG@K to promote model.
        model_stage: ZenML model stage to register the trained model ("staging" or "production").
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass to instantiate.
        force_promote: If True, promote the model to the specified stage regardless of quality gate results.
    Returns:
        Registered BaseRecommender subclass artifact.
    """
    rank = best_hyperparams.rank
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

    last_state = training_states[-1]
    rmse = float(last_state.rmse)  # informational only
    precision_at_k = float(last_state.precision_at_k)
    recall_at_k = float(last_state.recall_at_k)
    ndcg_at_k = float(last_state.ndcg_at_k)

    # ── Step 1: Absolute quality gate (minimum thresholds) ───────────────────
    # The model must clear these floors regardless of the previous deployment.
    threshold_failures: list[str] = []
    if precision_at_k < precision_at_k_threshold:
        threshold_failures.append(
            f"Precision@K {precision_at_k:.4f} < threshold {precision_at_k_threshold:.4f}"
        )
    if recall_at_k < recall_at_k_threshold:
        threshold_failures.append(
            f"Recall@K {recall_at_k:.4f} < threshold {recall_at_k_threshold:.4f}"
        )
    if ndcg_at_k < ndcg_at_k_threshold:
        threshold_failures.append(f"NDCG@K {ndcg_at_k:.4f} < threshold {ndcg_at_k_threshold:.4f}")

    gate_passed = not threshold_failures

    if not gate_passed:
        logger.warning(
            "Quality gate FAILED for model (%s). Not promoting to '%s'. "
            "Reasons: %s. [RMSE=%.4f, informational]",
            model_version,
            model_stage,
            "; ".join(threshold_failures),
            rmse,
        )
        passed = False
    else:
        logger.info(
            "Quality gate PASSED for model (%s) "
            "(Precision@K=%.4f >= %.4f, Recall@K=%.4f >= %.4f, NDCG@K=%.4f >= %.4f) "
            "[RMSE=%.4f, informational].",
            model_version,
            precision_at_k,
            precision_at_k_threshold,
            recall_at_k,
            recall_at_k_threshold,
            ndcg_at_k,
            ndcg_at_k_threshold,
            rmse,
        )

        # ── Step 2: Regression check against the currently deployed model ────
        # Fetch metrics from whatever model is already at the target stage.
        # This prevents promoting a model that passes absolute thresholds but
        # is still worse than what is already deployed.
        prev_metrics = _fetch_previous_model_metrics(
            CFG_MODEL_NAME, CFG_MODEL_ARTIFACT_NAME, model_stage
        )

        if prev_metrics is None:
            # No previous model / model metrics at this stage.
            logger.info("No previous model / model metrics found at stage '%s'. ", model_stage)
            passed = True
        else:
            # Require the new model to be at least as good on every ranking
            # metric.  A regression on any metric blocks promotion, even when
            # absolute thresholds are met.
            regressions: list[str] = []
            if precision_at_k < prev_metrics.precision_at_k:
                regressions.append(
                    f"Precision@K {precision_at_k:.4f} < prev {prev_metrics.precision_at_k:.4f}"
                )
            if recall_at_k < prev_metrics.recall_at_k:
                regressions.append(
                    f"Recall@K {recall_at_k:.4f} < prev {prev_metrics.recall_at_k:.4f}"
                )
            if ndcg_at_k < prev_metrics.ndcg_at_k:
                regressions.append(f"NDCG@K {ndcg_at_k:.4f} < prev {prev_metrics.ndcg_at_k:.4f}")

            if regressions:
                logger.warning(
                    "Model (%s) passed the quality gate but NOT promoted to '%s': "
                    "regresses on the currently deployed model. Regressions: %s. "
                    "[RMSE=%.4f, informational]",
                    model_version,
                    model_stage,
                    "; ".join(regressions),
                    rmse,
                )
                passed = False
            else:
                logger.info(
                    "Model (%s) improves on or matches the current '%s' model. "
                    "Promoting. "
                    "(Precision@K %.4f→%.4f, Recall@K %.4f→%.4f, NDCG@K %.4f→%.4f) "
                    "[RMSE=%.4f, informational].",
                    model_version,
                    model_stage,
                    prev_metrics.precision_at_k,
                    precision_at_k,
                    prev_metrics.recall_at_k,
                    recall_at_k,
                    prev_metrics.ndcg_at_k,
                    ndcg_at_k,
                    rmse,
                )
                passed = True

    # ── Step 3: Promote (or not) in the ZenML Model Control Plane ────────────
    promote = passed or force_promote
    if promote:
        try:
            ctx = get_step_context()
            z_model = ctx.model
            z_model.set_stage(model_stage, force=True)

            # NOTE: Update model_version to the latest version after promotion
            model_version = str(z_model.version)

            if force_promote and not passed:
                logger.warning(
                    "Model (%s) is being force-promoted to '%s' despite failing the quality gate or regressing on the previous model.",
                    model_version,
                    model_stage,
                )
        except Exception as exc:
            logger.warning("Could not promote model to '%s': %s", model_stage, exc)

    # Wrap trained factors and encoders into a BaseRecommender subclass instance
    metrics = ModelMetrics(
        rmse=rmse,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
    )
    model = recommender_cls(
        name=model_name,
        version=model_version,
        promoted=promote,
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
        metrics=metrics,
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
                "precision_at_k_threshold": precision_at_k_threshold,
                "recall_at_k_threshold": recall_at_k_threshold,
                "ndcg_at_k_threshold": ndcg_at_k_threshold,
                "quality_gate_passed": passed,
                "force_promote": force_promote,
                "metrics": metrics.model_dump(),
                "hyperparameters": best_hyperparams.model_dump(),
            },
            run_id_name_or_prefix=str(run_id),
            step_name=ctx.step_name,
        )

    except Exception as exc:
        logger.warning("Metadata logging skipped: %s", exc)

    logger.info("Model: %s\nPromoted to '%s': %s\n", model, model_stage, "YES" if promote else "NO")

    return model
