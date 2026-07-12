"""
steps/training/als_epoch.py

ZenML steps for per-epoch ALS training.

Pipeline wires n_iter steps in a chain:
    load_or_init_training_factors → als_epoch_0 → training_checkpoint_0 → ...

Resumability is coordinated by checkpoint steps in steps/training/checkopoint.py.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import get_step_context, log_metadata, step
from zenml.client import Client

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logger = logging.getLogger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

# ── Steps ─────────────────────────────────────────────────────────────────────


@step(enable_cache=True)
def init_als_factors(
    train_data: pd.DataFrame,
    best_hyperparams: dict,
    seed: int = 42,
) -> tuple[
    Annotated[np.ndarray, "user_factors"],
    Annotated[np.ndarray, "item_factors"],
]:
    """
    Initialize random ALS factor matrices for the training run.

    Called once at the start of the training chain.

    Args:
        train_data: Training ratings DataFrame. Used to infer matrix dimensions.
        best_hyperparams: Dict with key 'rank' for factor dimensionality.

    Returns:
        user_factors: (n_users × rank) float32 array.
        item_factors: (n_items × rank) float32 array.
    """
    from workflows.matrix_factorization.utils.als_numba import warmup_jit

    rank = int(best_hyperparams.get("rank", 50))
    n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
    n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1

    logger.info("Initializing ALS factors: %d users × %d items, rank=%d", n_users, n_items, rank)
    warmup_jit(rank=min(rank, 20))

    user_factors, item_factors = ALSRecommender.initialize_factors(
        n_users=n_users,
        n_items=n_items,
        rank=rank,
        seed=seed,
    )
    return user_factors, item_factors


@step(enable_cache=True, experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def train_als_epoch(
    epoch: int,
    start_epoch: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    best_hyperparams: dict,
    n_workers: int = 4,
    checkpoint_val_every_n_epochs: int = 1,
) -> tuple[
    Annotated[np.ndarray, "user_factors"],
    Annotated[np.ndarray, "item_factors"],
]:
    """
    Run one ALS epoch: update user factors then item factors in parallel.

    Uses ProcessPoolExecutor to run n_workers partition updates concurrently.
    Numba (prange) handles thread-level parallelism inside each partition.

    Called in a chain from training_pipeline:
        load_or_init_training_factors → als_epoch_0 → training_checkpoint_0 → ...

    Args:
        epoch: Zero-based epoch index (used for logging and val RMSE schedule).
        start_epoch: First epoch index that should execute training.
        user_factors: User factor matrix from previous epoch or resumed checkpoint.
        item_factors: Item factor matrix from previous epoch or resumed checkpoint.
        train_data: Training ratings DataFrame.
        val_data: Validation ratings DataFrame.
        best_hyperparams: Dict with rank, regularization, alpha, n_iter.
        n_workers: Number of parallel partition workers (ProcessPoolExecutor).
        checkpoint_val_every_n_epochs: Log val RMSE every N epochs.

    Returns:
        user_factors: Updated (n_users × rank) float32 array.
        item_factors: Updated (n_items × rank) float32 array.
    """
    regularization = float(best_hyperparams.get("regularization", 0.01))
    alpha = float(best_hyperparams.get("alpha", 1.0))
    n_iter = int(best_hyperparams.get("n_iter", 15))

    if epoch < start_epoch:
        logger.info(
            "Skipping epoch %d/%d (already checkpointed; resume starts at epoch %d)",
            epoch + 1,
            n_iter,
            start_epoch + 1,
        )
        return user_factors, item_factors

    logger.info("Epoch %d/%d: updating user factors (%d workers)...", epoch + 1, n_iter, n_workers)
    user_factors, item_factors = ALSRecommender.train_epoch(
        train_data=train_data,
        user_factors=user_factors,
        item_factors=item_factors,
        regularization=regularization,
        alpha=alpha,
        n_workers=n_workers,
    )

    if checkpoint_val_every_n_epochs > 0 and (epoch + 1) % checkpoint_val_every_n_epochs == 0:
        rmse = ALSRecommender.compute_rmse(
            val_data=val_data,
            user_factors=user_factors,
            item_factors=item_factors,
        )
        logger.info("Epoch %d/%d: val RMSE = %.4f", epoch + 1, n_iter, rmse)
        try:
            ctx = get_step_context()
            run_id = ctx.pipeline_run.id

            # Log metadata to ZenML model version
            log_metadata(
                metadata={
                    "val_rmse": rmse,
                    "epoch": epoch + 1,
                    **best_hyperparams,
                },
                run_id_name_or_prefix=str(run_id),
                step_name=ctx.step_name,
            )

        except Exception as exc:
            logger.warning("Metadata logging skipped: %s", exc)

    logger.info("Epoch %d/%d complete.", epoch + 1, n_iter)
    return user_factors, item_factors
