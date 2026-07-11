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
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import log_metadata, step

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.steps.training.als_partition import (
    update_item_partition,
    update_user_partition,
)

logger = logging.getLogger(__name__)


# ── Module-level wrappers required for ProcessPoolExecutor pickling ────────────

def _update_user_partition_worker(
    partition: np.ndarray, item_factors: np.ndarray, regularization: float, alpha: float
) -> np.ndarray:
    return update_user_partition(partition, item_factors, regularization, alpha)


def _update_item_partition_worker(
    partition: np.ndarray, user_factors: np.ndarray, regularization: float, alpha: float
) -> np.ndarray:
    return update_item_partition(partition, user_factors, regularization, alpha)


# ── Partition matrix builders ──────────────────────────────────────────────────

def _build_user_partition_matrices(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
    n_partitions: int,
) -> Generator[np.ndarray, None, None]:
    """Yield dense user-partition rating matrices for ALS user update."""
    from workflows.matrix_factorization.utils.als_numba import fill_user_partition
    partition_size = (n_users + n_partitions - 1) // n_partitions
    for p in range(n_partitions):
        u_start = p * partition_size
        u_end = min(u_start + partition_size, n_users)
        yield fill_user_partition(user_indices, item_indices, ratings, u_start, u_end, n_items)


def _build_item_partition_matrices(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
    n_partitions: int,
) -> Generator[np.ndarray, None, None]:
    """Yield dense item-partition rating matrices for ALS item update."""
    from workflows.matrix_factorization.utils.als_numba import fill_item_partition
    partition_size = (n_items + n_partitions - 1) // n_partitions
    for p in range(n_partitions):
        i_start = p * partition_size
        i_end = min(i_start + partition_size, n_items)
        yield fill_item_partition(user_indices, item_indices, ratings, i_start, i_end, n_users)


def _compute_val_rmse(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    ratings: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> float:
    from workflows.matrix_factorization.utils.als_numba import compute_rmse_block
    sse, count = compute_rmse_block(user_indices, item_indices, ratings, user_factors, item_factors)
    return float(np.sqrt(sse / count)) if count > 0 else float("inf")


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

    rng = np.random.default_rng(42)
    user_factors = (rng.standard_normal((n_users, rank)) * 0.01).astype(np.float32)
    item_factors = (rng.standard_normal((n_items, rank)) * 0.01).astype(np.float32)
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

    n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
    n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1

    train_user_idx = train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int64)
    train_item_idx = train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int64)
    train_ratings  = train_data[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)

    val_user_idx = np.clip(
        val_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int64), 0, n_users - 1
    )
    val_item_idx = np.clip(
        val_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int64), 0, n_items - 1
    )
    val_ratings = val_data[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)

    logger.info("Epoch %d/%d: updating user factors (%d workers)...", epoch + 1, n_iter, n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        user_partitions = list(_build_user_partition_matrices(
            train_user_idx, train_item_idx, train_ratings, n_users, n_items, n_workers
        ))
        user_blocks = list(executor.map(
            _update_user_partition_worker,
            user_partitions,
            [item_factors] * len(user_partitions),
            [regularization] * len(user_partitions),
            [alpha] * len(user_partitions),
        ))
        user_factors = np.vstack(user_blocks)[:n_users]

        logger.info("Epoch %d/%d: updating item factors...", epoch + 1, n_iter)

        item_partitions = list(_build_item_partition_matrices(
            train_user_idx, train_item_idx, train_ratings, n_users, n_items, n_workers
        ))
        item_blocks = list(executor.map(
            _update_item_partition_worker,
            item_partitions,
            [user_factors] * len(item_partitions),
            [regularization] * len(item_partitions),
            [alpha] * len(item_partitions),
        ))
        item_factors = np.vstack(item_blocks)[:n_items]

    if checkpoint_val_every_n_epochs > 0 and (epoch + 1) % checkpoint_val_every_n_epochs == 0:
        rmse = _compute_val_rmse(val_user_idx, val_item_idx, val_ratings, user_factors, item_factors)
        logger.info("Epoch %d/%d: val RMSE = %.4f", epoch + 1, n_iter, rmse)
        log_metadata(metadata={"val_rmse": rmse, "epoch": epoch + 1})

    logger.info("Epoch %d/%d complete.", epoch + 1, n_iter)
    return user_factors, item_factors
