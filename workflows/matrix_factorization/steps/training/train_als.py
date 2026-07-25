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
from zenml.enums import ModelStages

from helpers.checkpointing import (
    clean_run_checkpoints,
    get_zenml_step_checkpoint_path,
    load_latest_checkpoint,
    save_checkpoint,
)
from helpers.s3_client import resolve_zenml_s3_credentials
from workflows.matrix_factorization.configs import (
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_NAME,
)
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    EpochState,
    EpochStates,
    Hyperparameters,
    load_recommender_class,
)

logger = logging.getLogger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker


def _load_warm_start_factors(
    model_name: str,
    artifact_name: str,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series] | None:
    """
    Load user_factors, item_factors, user_encoder, and item_encoder from the
    model currently at *stage* in the ZenML Model Control Plane.

    Returns None if no model is deployed at that stage or on any load error,
    so the caller falls back to a fresh random initialisation.
    """
    client = Client()
    try:
        prev_version = client.get_model_version(
            model_name_or_id=model_name,
            model_version_name_or_number_or_id=stage,
        )
    except Exception:
        logger.warning(
            "Warm start: no model found at stage '%s' — starting from random init.",
            stage,
        )
        return None

    try:
        artifact_response = prev_version.get_artifact(name=artifact_name)
        if artifact_response is None:
            logger.warning(
                "Warm start: artifact '%s' not found at stage '%s' — starting from random init.",
                artifact_name,
                stage,
            )
            return None
        recommender: BaseRecommender = artifact_response.load()
        logger.info(
            "Warm start: loaded model '%s' at stage '%s' "
            "(user_factors=%s, item_factors=%s, n_users=%d, n_items=%d).",
            model_name,
            stage,
            recommender.user_factors.shape,
            recommender.item_factors.shape,
            len(recommender.user_encoder),
            len(recommender.item_encoder),
        )
        return (
            recommender.user_factors,
            recommender.item_factors,
            recommender.user_encoder,
            recommender.item_encoder,
        )
    except Exception as exc:
        logger.warning(
            "Warm start: could not load factors from '%s' at stage '%s': %s "
            "— starting from random init.",
            artifact_name,
            stage,
            exc,
        )
        return None


def _remap_warm_start_factors(
    old_user_factors: np.ndarray,
    old_item_factors: np.ndarray,
    old_user_encoder: pd.Series,
    old_item_encoder: pd.Series,
    new_user_encoder: pd.Series,
    new_item_encoder: pd.Series,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remap previous model factors to the new encoder's index space using raw entity IDs.

    For each raw ID that exists in both the old and new encoders, the corresponding
    factor row is placed at the correct new dense index. This handles all dataset
    change scenarios:
    - IDs that shifted position (new IDs inserted between existing ones)
    - New entities added anywhere in the ID space
    - Entities removed from the dataset

    New entities (not seen during previous training) are initialised with small
    Gaussian noise matching the ALS random init scale.

    Args:
        old_user_factors: Factor matrix from the previous model, shape (n_users_old, factors).
        old_item_factors: Factor matrix from the previous model, shape (n_items_old, factors).
        old_user_encoder: pd.Series mapping raw userId → old dense index.
        old_item_encoder: pd.Series mapping raw movieId → old dense index.
        new_user_encoder: pd.Series mapping raw userId → new dense index.
        new_item_encoder: pd.Series mapping raw movieId → new dense index.
        seed: RNG seed for new-entity initialisations.

    Returns:
        (user_factors, item_factors) aligned to the new encoder's index space.
    """
    factors = old_user_factors.shape[1]
    n_users = len(new_user_encoder)
    n_items = len(new_item_encoder)
    init_scale = 0.01
    rng = np.random.default_rng(seed)

    # Initialise all rows with random noise; known entities will be overwritten below.
    user_factors = (rng.standard_normal((n_users, factors)) * init_scale).astype(np.float32)
    item_factors = (rng.standard_normal((n_items, factors)) * init_scale).astype(np.float32)

    # Remap rows for entities present in both old and new encoders.
    common_user_ids = old_user_encoder.index.intersection(new_user_encoder.index)
    common_item_ids = old_item_encoder.index.intersection(new_item_encoder.index)

    if len(common_user_ids) > 0:
        old_u_idx = old_user_encoder[common_user_ids].to_numpy()
        new_u_idx = new_user_encoder[common_user_ids].to_numpy()
        user_factors[new_u_idx] = old_user_factors[old_u_idx].astype(np.float32)

    if len(common_item_ids) > 0:
        old_i_idx = old_item_encoder[common_item_ids].to_numpy()
        new_i_idx = new_item_encoder[common_item_ids].to_numpy()
        item_factors[new_i_idx] = old_item_factors[old_i_idx].astype(np.float32)

    logger.info(
        "Warm start: remapped %d/%d users (%d new), %d/%d items (%d new).",
        len(common_user_ids),
        n_users,
        n_users - len(common_user_ids),
        len(common_item_ids),
        n_items,
        n_items - len(common_item_ids),
    )
    return user_factors, item_factors


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
    enable_warm_start: bool = False,
    warm_start_model_stage: ModelStages | None = None,
    user_encoder: pd.Series | None = None,
    item_encoder: pd.Series | None = None,
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
        best_hyperparams: Dict with factors, regularization, alpha, n_iter.
        checkpoint_path: Base path for epoch-level checkpoints (local or s3://).
        n_workers: Parallel workers for training (interpretation is model-specific).
        eval_at_k: K for ranking metrics (Precision@K, Recall@K, NDCG@K).
        eval_every_n_epochs: Compute and log val RMSE every N epochs.
        checkpoint_every_n_epochs: Save checkpoint every N epochs.
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass.
            Any subclass can be used without modifying this step.
        enable_warm_start: If True, initialise training from the latent factors of the
            model currently registered at *warm_start_model_stage* when no checkpoint
            exists for the current run. Checkpoints always take priority over warm start.
        warm_start_model_stage: ZenML model stage to load initial factors from (e.g.
            "production" or "staging"). Only used when enable_warm_start=True.
        user_encoder: pd.Series mapping raw userId → dense index for the current dataset.
            Required for warm start to correctly remap factors when the dataset changes.
        item_encoder: pd.Series mapping raw movieId → dense index for the current dataset.
            Required for warm start to correctly remap factors when the dataset changes.
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

    # Priority: checkpoint > warm start > random init
    if user_factors_cp is not None and item_factors_cp is not None:
        initial_factors: tuple[np.ndarray, np.ndarray] | None = (user_factors_cp, item_factors_cp)
    elif enable_warm_start and warm_start_model_stage:
        warm = _load_warm_start_factors(
            model_name=CFG_MODEL_NAME,
            artifact_name=CFG_MODEL_ARTIFACT_NAME,
            stage=warm_start_model_stage,
        )
        if warm is not None and user_encoder is not None and item_encoder is not None:
            initial_factors = _remap_warm_start_factors(
                old_user_factors=warm[0],
                old_item_factors=warm[1],
                old_user_encoder=warm[2],
                old_item_encoder=warm[3],
                new_user_encoder=user_encoder,
                new_item_encoder=item_encoder,
                seed=42,
            )
        elif warm is not None:
            logger.warning(
                "Warm start: user_encoder/item_encoder not provided — "
                "cannot safely remap factors when the dataset changes. "
                "Falling back to random init."
            )
            initial_factors = None
        else:
            initial_factors = None
    else:
        initial_factors = None

    factors = best_hyperparams.factors
    regularization = best_hyperparams.regularization
    alpha = best_hyperparams.alpha
    n_iter = best_hyperparams.n_iter

    warm_start_active = (
        enable_warm_start
        and warm_start_model_stage
        and initial_factors is not None
        and user_factors_cp is None
    )
    logger.info(
        "Training %s: factors=%d, regularization=%.4f, alpha=%.4f, n_iter=%d, "
        "start_epoch=%d, n_workers=%d, warm_start=%s",
        recommender_class_name,
        factors,
        regularization,
        alpha,
        n_iter,
        start_epoch,
        n_workers,
        f"{warm_start_model_stage!r}" if warm_start_active else "disabled",
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
        factors=factors,
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
                "warm_start_active": bool(warm_start_active),
                "warm_start_stage": str(warm_start_model_stage)
                if warm_start_active and warm_start_model_stage
                else "",
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
