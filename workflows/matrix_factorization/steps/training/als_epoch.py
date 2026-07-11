"""
steps/training/als_epoch.py

ZenML steps for per-epoch ALS training using ZenML fan-out/fan-in.

Pipeline wires n_iter steps in a chain:
    init_als_factors → als_epoch_0 → als_epoch_1 → ... → als_epoch_{n_iter-1}

Each epoch step uses ProcessPoolExecutor for within-epoch partition-level
parallelism. Numba handles thread-level parallelism inside each partition via prange.

ZenML caching provides epoch-level resume: if epoch N's inputs are unchanged from a
previous run, ZenML skips it and uses the cached output — equivalent to checkpointing
without manual checkpoint management.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import log_metadata, step

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logger = logging.getLogger(__name__)

# ── Steps ─────────────────────────────────────────────────────────────────────

@step(enable_cache=True)
def init_als_factors(
    train_data: pd.DataFrame,
    best_hyperparams: dict,
) -> tuple[
    Annotated[np.ndarray, "user_factors"],
    Annotated[np.ndarray, "item_factors"],
]:
    """
    Initialize random ALS factor matrices for the training run.

    Called once at the start of the training chain. ZenML caches the output
    so re-runs with unchanged data/hyperparams skip this step automatically.

    Args:
        train_data: Training ratings DataFrame. Used to infer matrix dimensions.
        best_hyperparams: Dict with key 'rank' for factor dimensionality.

    Returns:
        user_factors: (n_users × rank) float32 array, randomly initialized.
        item_factors: (n_items × rank) float32 array, randomly initialized.
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
        seed=42,
    )
    return user_factors, item_factors


@step(enable_cache=True)
def train_als_epoch(
    epoch: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    best_hyperparams: dict,
    n_workers: int = 4,
    checkpoint_val_every_n_epochs: int = 5,
) -> tuple[
    Annotated[np.ndarray, "user_factors"],
    Annotated[np.ndarray, "item_factors"],
]:
    """
    Run one ALS epoch: update user factors then item factors in parallel.

    Uses ProcessPoolExecutor to run n_workers partition updates concurrently.
    Numba (prange) handles thread-level parallelism inside each partition.

    Called in a chain from training_pipeline:
        init_als_factors → als_epoch_0 → als_epoch_1 → ... → als_epoch_{n_iter-1}

    ZenML caches the output keyed on (epoch, user_factors, item_factors, train_data,
    best_hyperparams, n_workers). On pipeline restart, completed epochs are skipped
    and the chain resumes from the first uncached epoch.

    Args:
        epoch: Zero-based epoch index (used for logging and val RMSE schedule).
        user_factors: User factor matrix from the previous epoch (or init).
        item_factors: Item factor matrix from the previous epoch (or init).
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
        log_metadata(metadata={"val_rmse": rmse, "epoch": epoch + 1})

    logger.info("Epoch %d/%d complete.", epoch + 1, n_iter)
    return user_factors, item_factors
