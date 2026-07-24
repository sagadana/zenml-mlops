"""
steps/training/train_als.py

ZenML step: train_als

Trains the full ALS model (all epochs) in a single step with automatic
checkpoint resume. Replaces the previous chain of per-epoch train_als_epoch
+ save_training_checkpoint steps.

Supports both ALSRecommender (numba) and ALSImplicitRecommender (implicit lib)
via the recommender_class_name parameter. To swap models, change that param —
no other code changes required.

Checkpointing protocol:
  On each epoch, checkpoint_callback saves factors to checkpoint_path/<run_id>/training/.
  On a fresh run the step trains from epoch 0.
  On resume (step re-run after failure), load_latest_checkpoint() finds the latest
  .done marker and resumes from that epoch, warm-starting the model.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import get_step_context, log_metadata, step
from zenml.client import Client

from helpers.checkpointing import (
    clean_run_checkpoints,
    get_zenml_step_checkpoint_path,
    load_latest_checkpoint,
    save_checkpoint,
)
from helpers.s3_client import resolve_zenml_s3_credentials
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    EpochState,
    EpochStates,
    Hyperparameters,
    load_recommender_class,
)

logger = logging.getLogger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=True)
def train_als(
    features: pd.DataFrame,
    best_hyperparams: Hyperparameters,
    checkpoint_path: str,
    n_workers: int = 4,
    eval_at_k: int = 10,
    eval_every_n_epochs: int = 1,
    checkpoint_every_n_epochs: int = 1,
    recommender_class_name: str = "workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender",
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> tuple[
    Annotated[np.ndarray, "user_factors"],
    Annotated[np.ndarray, "item_factors"],
    Annotated[EpochStates, "training_states"],
]:
    """
    Train the recommender model for all epochs using the full (unsplit) dataset.

    Automatically resumes from the latest checkpoint if one exists in
    checkpoint_path for the current pipeline run.

    Args:
        features: Full feature DataFrame (no train/val/test split).
        best_hyperparams: Dict with rank, regularization, alpha, n_iter.
        checkpoint_path: Base path for epoch-level checkpoints (local or s3://).
        n_workers: Parallel workers for training (interpretation is model-specific).
        eval_at_k: K for ranking metrics (Precision@K, Recall@K, NDCG@K).
        eval_every_n_epochs: Compute and log val RMSE every N epochs.
        checkpoint_every_n_epochs: Save checkpoint every N epochs.
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass.
            Any subclass can be used without modifying this step.
        seaweedfs_s3_internal_endpoint: SeaweedFS S3 endpoint for local stack checkpoints.
        zenml_local_s3_secret_name: ZenML secret with SeaweedFS credentials.

    Returns:
        (user_factors, item_factors, training_states)
    """
    recommender_cls: type[BaseRecommender] = load_recommender_class(recommender_class_name)

    # ── Resolve checkpoint credentials ────────────────────────────────────────
    access_key_id, secret_access_key = resolve_zenml_s3_credentials(zenml_local_s3_secret_name)
    training_cp_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="training")

    # ── Load latest checkpoint (or start fresh) ───────────────────────────────
    start_epoch, user_factors_cp, item_factors_cp = load_latest_checkpoint(
        training_cp_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )
    initial_factors = (
        (user_factors_cp, item_factors_cp)
        if user_factors_cp is not None and item_factors_cp is not None
        else None
    )

    rank = best_hyperparams.rank
    regularization = best_hyperparams.regularization
    alpha = best_hyperparams.alpha
    n_iter = best_hyperparams.n_iter

    logger.info(
        "Training %s: rank=%d, regularization=%.4f, alpha=%.4f, n_iter=%d, "
        "start_epoch=%d, n_workers=%d",
        recommender_class_name,
        rank,
        regularization,
        alpha,
        n_iter,
        start_epoch,
        n_workers,
    )

    # ── Define callbacks ────────────────────────────────────────────────────────────────────────────
    def checkpoint_callback(result: EpochState, ufs: np.ndarray, ifs: np.ndarray) -> None:
        save_checkpoint(
            epoch=result.epoch + 1,
            primary=ufs,
            secondary=ifs,
            base_path=training_cp_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=access_key_id,
            seaweedfs_secret_access_key=secret_access_key,
        )

    def epoch_end_callback(state: EpochState) -> None:
        logger.info(
            "Epoch %d/%d: Loss = %.6f, RMSE = %.6f, P@%d = %.6f, R@%d = %.6f, NDCG@%d = %.6f (Elapsed = %.2fs)",
            state.epoch,
            n_iter,
            state.loss,
            state.rmse,
            state.k,
            state.precision_at_k,
            state.k,
            state.recall_at_k,
            state.k,
            state.ndcg_at_k,
            state.elapsed_time,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    user_factors, item_factors, states = recommender_cls.train(
        rank=rank,
        regularization=regularization,
        alpha=alpha,
        n_iter=n_iter,
        train_data=features,
        val_data=None,  # No validation data is used in this step; evaluation is done on the full dataset.
        n_workers=n_workers,
        initial_factors=initial_factors,
        start_epoch=start_epoch,
        seed=42,
        k=eval_at_k,
        eval_every_n_epochs=eval_every_n_epochs,
        epoch_end_callback=epoch_end_callback,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        checkpoint_callback=checkpoint_callback,
    )

    # ── Log metadata ──────────────────────────────────────────────────────────
    try:
        ctx = get_step_context()
        run_id = ctx.pipeline_run.id
        log_metadata(
            metadata={
                "start_epoch": start_epoch,
                "n_iter": n_iter,
                "final_loss": states[-1].loss if states else 0,
                "final_rmse": states[-1].rmse if states else 0,
                "final_precision_at_k": states[-1].precision_at_k if states else 0,
                "final_recall_at_k": states[-1].recall_at_k if states else 0,
                "final_ndcg_at_k": states[-1].ndcg_at_k if states else 0,
            },
            run_id_name_or_prefix=str(run_id),
            step_name=ctx.step_name,
        )
    except Exception as exc:
        logger.warning("Metadata logging skipped: %s", exc)

    # ── Cleanup: Delete training checkpoints after successful completion ───────
    clean_run_checkpoints(
        base_path=training_cp_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )

    return user_factors, item_factors, states
