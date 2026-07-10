"""
steps/training/train.py

ZenML step: train_als

Distributed ALS training with Dask (parallel per-partition factor updates)
and Numba (JIT-compiled per-user/item least-squares solve).

Checkpointing: After each epoch, saves user_factors.npy + item_factors.npy
+ a .done marker to base_path. On restart, automatically resumes from the
last complete epoch. See utils/checkpointing.py for the full protocol.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator
from typing import Annotated

import dask_expr as dd
import numpy as np
import pandas as pd
from zenml import get_step_context, step

from helpers.checkpointing import load_latest_checkpoint, save_checkpoint
from helpers.dask_cluster import get_client_mode_from_config, get_dask_client
from workflows.matrix_factorization.configs import (
    CFG_DASK_SCHEDULER_ADDRESS,
    CFG_FEATURES_FIELD_NAMES,
)
from workflows.matrix_factorization.steps.training.als_partition import (
    update_item_partition,
    update_user_partition,
)

logger = logging.getLogger(__name__)


def _build_user_partition_matrices(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
    n_partitions: int,
) -> Generator[np.ndarray, None, None]:
    """
    Yield dense user-partition rating matrices for ALS user update.
    Each partition covers a contiguous range of user_idx values.
    Yields (partition_n_users × n_items) float32 arrays one at a time.
    """
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
    """Yield dense item-partition rating matrices for ALS item update (transposed roles)."""
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
    """Compute RMSE on the validation set."""
    from workflows.matrix_factorization.utils.als_numba import compute_rmse_block

    sse, count = compute_rmse_block(user_indices, item_indices, ratings, user_factors, item_factors)
    return float(np.sqrt(sse / count)) if count > 0 else float("inf")


@step(enable_cache=True)
def train_als(
    train_data: dd.DataFrame,
    val_data: dd.DataFrame,
    best_hyperparams: dict,
    n_dask_partitions: int = 4,
    checkpoint_path: str = "./checkpoints",
    checkpoint_val_every_n_epochs: int = 5,
    scheduler_address: str | None = CFG_DASK_SCHEDULER_ADDRESS,
) -> tuple[
    Annotated[np.ndarray, "user_factors"],
    Annotated[np.ndarray, "item_factors"],
]:
    """
    Train ALS model with distributed Dask workers and epoch-level checkpointing.

    On first run:
      - Initializes random factor matrices
      - Trains for n_iter epochs
      - Saves checkpoint after each epoch

    On restart after failure:
      - Loads latest complete checkpoint automatically
      - Resumes from the next epoch (never re-trains completed epochs)

    Args:
        train_data: Training ratings Dask DataFrame (user_idx, item_idx, rating, timestamp).
        val_data: Validation ratings Dask DataFrame.
        best_hyperparams: Dict with keys: rank, regularization, alpha, n_iter.
            From run_hpo step or default config values.
        checkpoint_path: Base directory for epoch checkpoints.
            Local path (./checkpoints/) or S3 URI (s3://bucket/checkpoints/).
        n_dask_partitions: Number of parallel Dask tasks per ALS epoch.
        checkpoint_val_every_n_epochs: Compute and log val RMSE every N epochs.

    Returns:
        user_factors: (n_users × rank) float32 array.
        item_factors: (n_items × rank) float32 array.
    """
    from workflows.matrix_factorization.utils.als_numba import warmup_jit

    rank = int(best_hyperparams.get("rank", 50))
    regularization = float(best_hyperparams.get("regularization", 0.01))
    alpha = float(best_hyperparams.get("alpha", 1.0))
    n_iter = int(best_hyperparams.get("n_iter", 15))

    logger.info(
        "ALS hyperparams: rank=%d, reg=%.4f, alpha=%.4f, n_iter=%d",
        rank,
        regularization,
        alpha,
        n_iter,
    )

    # Warm up Numba JIT before the training loop
    warmup_jit(rank=min(rank, 20))

    # Materialize training data to pandas for partition building
    logger.info("Materializing training data...")
    train_pd: pd.DataFrame = train_data.compute()
    val_pd: pd.DataFrame = val_data.compute()

    if not isinstance(val_pd, pd.DataFrame):
        raise ValueError("train_data and val_data must be a pandas DataFrame after compute()")

    n_users = int(train_pd[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
    n_items = int(train_pd[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1
    logger.info("Matrix dimensions: %d users × %d items", n_users, n_items)

    # Extract numpy arrays once — avoids repeated pandas overhead in the training loop
    train_user_idx = train_pd[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int64)
    train_item_idx = train_pd[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int64)
    train_ratings = train_pd[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)

    val_user_idx = np.clip(
        val_pd[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int64),
        0,
        n_users - 1,
    )
    val_item_idx = np.clip(
        val_pd[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int64),
        0,
        n_items - 1,
    )
    val_ratings = val_pd[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)

    # Unique run ID for checkpoint directory scoping
    try:
        run_id = get_step_context().pipeline_run.id
    except Exception:
        run_id = str(uuid.uuid4())[:8]

    run_checkpoint_path = f"{checkpoint_path}/{run_id}"
    logger.info("Checkpoint path: %s", run_checkpoint_path)

    # ── Resume or initialize ──────────────────────────────────────────────────
    start_epoch, user_factors, item_factors = load_latest_checkpoint(run_checkpoint_path)

    if user_factors is None:
        logger.info("No checkpoint found. Initializing random factors...")
        rng = np.random.default_rng(42)
        user_factors = (rng.standard_normal((n_users, rank)) * 0.01).astype(np.float32)
        item_factors = (rng.standard_normal((n_items, rank)) * 0.01).astype(np.float32)
    else:
        logger.info("Resuming from epoch %d", start_epoch)

    # ── Training loop ─────────────────────────────────────────────────────────
    with get_dask_client(
        mode=get_client_mode_from_config(scheduler_address), scheduler_address=scheduler_address
    ) as client:
        for epoch in range(start_epoch, n_iter):
            logger.info("Epoch %d/%d: updating user factors...", epoch + 1, n_iter)

            user_futures = [
                client.submit(update_user_partition, partition, item_factors, regularization, alpha)
                for partition in _build_user_partition_matrices(
                    train_user_idx,
                    train_item_idx,
                    train_ratings,
                    n_users,
                    n_items,
                    n_dask_partitions,
                )
            ]
            user_blocks = client.gather(user_futures)
            user_factors = np.vstack(user_blocks)[:n_users]  # type: ignore

            logger.info("Epoch %d/%d: updating item factors...", epoch + 1, n_iter)
            item_futures = [
                client.submit(update_item_partition, partition, user_factors, regularization, alpha)
                for partition in _build_item_partition_matrices(
                    train_user_idx,
                    train_item_idx,
                    train_ratings,
                    n_users,
                    n_items,
                    n_dask_partitions,
                )
            ]
            item_blocks = client.gather(item_futures)
            item_factors = np.vstack(item_blocks)[:n_items]  # type: ignore

            # ── Checkpoint (atomic) ───────────────────────────────────────────
            save_checkpoint(epoch + 1, user_factors, item_factors, run_checkpoint_path)

            # ── Validation RMSE ───────────────────────────────────────────────
            if (
                checkpoint_val_every_n_epochs > 0
                and (epoch + 1) % checkpoint_val_every_n_epochs == 0
            ):
                rmse = _compute_val_rmse(
                    val_user_idx, val_item_idx, val_ratings, user_factors, item_factors
                )
                logger.info("Epoch %d/%d: val RMSE = %.4f", epoch + 1, n_iter, rmse)

    if item_factors is None or user_factors is None:
        raise RuntimeError("Training failed: final factors are None")

    logger.info(
        "ALS training complete. Final factors: user=%s item=%s",
        user_factors.shape,
        item_factors.shape,
    )
    return user_factors, item_factors
