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
import os
import uuid
from typing import Annotated

import dask_expr as dd
import numpy as np
import pandas as pd
from zenml import get_step_context, step

from helpers.checkpointing import load_latest_checkpoint, save_checkpoint
from helpers.dask_cluster import get_client_mode_from_config, get_dask_client

logger = logging.getLogger(__name__)


def _build_user_partition_matrices(
    train_pd: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_partitions: int,
) -> list[np.ndarray]:
    """
    Build a list of dense user-partition rating matrices for ALS user update.
    Each partition covers a contiguous range of user_idx values.
    Returns list of (partition_n_users × n_items) float32 arrays.
    """
    partition_size = (n_users + n_partitions - 1) // n_partitions
    partitions = []
    for p in range(n_partitions):
        u_start = p * partition_size
        u_end = min(u_start + partition_size, n_users)
        mask = (train_pd["user_idx"] >= u_start) & (train_pd["user_idx"] < u_end)
        sub = train_pd[mask]
        R_p = np.zeros((u_end - u_start, n_items), dtype=np.float32)
        local_u = sub["user_idx"] - u_start
        R_p[local_u, sub["item_idx"].values] = sub["rating"].values.astype(np.float32)
        partitions.append(R_p)
    return partitions


def _build_item_partition_matrices(
    train_pd: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_partitions: int,
) -> list[np.ndarray]:
    """Same as _build_user_partition_matrices but for item-factor update (transposed roles)."""
    partition_size = (n_items + n_partitions - 1) // n_partitions
    partitions = []
    for p in range(n_partitions):
        i_start = p * partition_size
        i_end = min(i_start + partition_size, n_items)
        mask = (train_pd["item_idx"] >= i_start) & (train_pd["item_idx"] < i_end)
        sub = train_pd[mask]
        R_p = np.zeros((i_end - i_start, n_users), dtype=np.float32)
        local_i = sub["item_idx"] - i_start
        R_p[local_i, sub["user_idx"].values] = sub["rating"].values.astype(np.float32)
        partitions.append(R_p)
    return partitions


def _compute_val_rmse(
    val_pd: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> float:
    """Compute RMSE on the validation set."""
    from workflows.matrix_factorization.utils.als_numba import compute_rmse_block

    u_idx = np.clip(val_pd["user_idx"], 0, user_factors.shape[0] - 1)
    i_idx = np.clip(val_pd["item_idx"], 0, item_factors.shape[0] - 1)
    r = np.asarray(val_pd["rating"], dtype=np.float32)
    sse, count = compute_rmse_block(u_idx, i_idx, r, user_factors, item_factors)
    return float(np.sqrt(sse / count)) if count > 0 else float("inf")


@step(enable_cache=True)
def train_als(
    train_data: dd.DataFrame,
    val_data: dd.DataFrame,
    best_hyperparams: dict,
    checkpoint_path: str = "./checkpoints",
    n_dask_partitions: int = 4,
    checkpoint_val_every_n_epochs: int = 5,
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
    from workflows.matrix_factorization.utils.als_numba import (
        solve_item_factors,
        solve_user_factors,
        warmup_jit,
    )

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
    train_pd = train_data.compute()
    val_pd = val_data.compute()

    n_users = int(train_pd["user_idx"].max()) + 1
    n_items = int(train_pd["item_idx"].max()) + 1
    logger.info("Matrix dimensions: %d users × %d items", n_users, n_items)

    # Build per-partition rating matrices (done once, reused every epoch)
    logger.info("Building %d user partition matrices...", n_dask_partitions)
    user_partitions = _build_user_partition_matrices(train_pd, n_users, n_items, n_dask_partitions)
    item_partitions = _build_item_partition_matrices(train_pd, n_users, n_items, n_dask_partitions)

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
    config = {"dask_scheduler_address": os.environ.get("DASK_SCHEDULER_ADDRESS")}
    with get_dask_client(mode=get_client_mode_from_config(config)) as client:
        for epoch in range(start_epoch, n_iter):
            logger.info("Epoch %d/%d: updating user factors...", epoch + 1, n_iter)

            # Compute user factor partitions counts for accurate stacking
            # partition_sizes = [p.shape[0] for p in user_partitions]
            user_futures = [
                client.submit(solve_user_factors, partition, item_factors, regularization, alpha)
                for partition in user_partitions
            ]
            user_blocks = client.gather(user_futures)
            user_factors = np.vstack(user_blocks)[:n_users]  # trim to exact n_users

            logger.info("Epoch %d/%d: updating item factors...", epoch + 1, n_iter)
            item_futures = [
                client.submit(solve_item_factors, partition, user_factors, regularization, alpha)
                for partition in item_partitions
            ]
            item_blocks = client.gather(item_futures)
            item_factors = np.vstack(item_blocks)[:n_items]

            # ── Checkpoint (atomic) ───────────────────────────────────────────
            save_checkpoint(epoch + 1, user_factors, item_factors, run_checkpoint_path)

            # ── Validation RMSE ───────────────────────────────────────────────
            if (
                checkpoint_val_every_n_epochs > 0
                and (epoch + 1) % checkpoint_val_every_n_epochs == 0
            ):
                rmse = _compute_val_rmse(val_pd, user_factors, item_factors)
                logger.info("Epoch %d/%d: val RMSE = %.4f", epoch + 1, n_iter, rmse)

    if item_factors is None or user_factors is None:
        raise RuntimeError("Training failed: final factors are None")

    logger.info(
        "ALS training complete. Final factors: user=%s item=%s",
        user_factors.shape,
        item_factors.shape,
    )
    return user_factors, item_factors
