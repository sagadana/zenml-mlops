"""
models/als_numba_recommender.py

ALSRecommender — numba-based ALS recommendation model.

Extends BaseRecommender. Training uses partitioned dense blocks with
ProcessPoolExecutor (process-level) + Numba JIT kernels (thread-level).

Inference, evaluation, and Pydantic types are inherited from BaseRecommender and
re-exported here for backward compatibility.

Serialized as cloudpickle by ALSRecommenderMaterializer.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat

import numpy as np
import pandas as pd

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
)

logger = logging.getLogger(__name__)
_MP_CONTEXT = mp.get_context("spawn")


def _update_user_partition_worker(
    partition: np.ndarray,
    item_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """Worker entrypoint: update user factors for one user-partition matrix."""
    from workflows.matrix_factorization.utils.als_numba import solve_user_factors

    return solve_user_factors(partition, item_factors, regularization, alpha)


def _update_item_partition_worker(
    partition: np.ndarray,
    user_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """Worker entrypoint: update item factors for one item-partition matrix."""
    from workflows.matrix_factorization.utils.als_numba import solve_item_factors

    return solve_item_factors(partition, user_factors, regularization, alpha)


def _build_user_partition_matrices(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
    n_partitions: int,
):
    """Yield dense user-partition matrices for ALS user-factor updates."""
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
):
    """Yield dense item-partition matrices for ALS item-factor updates."""
    from workflows.matrix_factorization.utils.als_numba import fill_item_partition

    partition_size = (n_items + n_partitions - 1) // n_partitions
    for p in range(n_partitions):
        i_start = p * partition_size
        i_end = min(i_start + partition_size, n_items)
        yield fill_item_partition(user_indices, item_indices, ratings, i_start, i_end, n_users)


@dataclass(repr=False)
class ALSRecommender(BaseRecommender):
    """
    Numba-based ALS recommendation model.

    Training uses ProcessPoolExecutor for partition-level parallelism and
    Numba JIT kernels for the ALS solve hot path.

    All inference methods (predict, batch_predict, get_similar_items, compute_rmse)
    are inherited from BaseRecommender.
    """

    @staticmethod
    def initialize_factors(
        n_users: int,
        n_items: int,
        rank: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize user/item latent factors with small Gaussian noise."""
        rng = np.random.default_rng(seed)
        user_factors = (rng.standard_normal((n_users, rank)) * 0.01).astype(np.float32)
        item_factors = (rng.standard_normal((n_items, rank)) * 0.01).astype(np.float32)
        return user_factors, item_factors

    @classmethod
    def train_epoch(
        cls,
        train_data: pd.DataFrame,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        regularization: float,
        alpha: float,
        n_workers: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one ALS epoch: update users then items.

        Uses partition-level parallelism via ProcessPoolExecutor when n_workers > 1.
        Uses streamed partition generators to avoid materializing all partitions at once.
        """
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")

        n_users = user_factors.shape[0]
        n_items = item_factors.shape[0]
        n_partitions = max(1, n_workers)

        train_user_idx = train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int64)
        train_item_idx = train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int64)
        train_ratings = train_data[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)

        user_partitions = _build_user_partition_matrices(
            train_user_idx, train_item_idx, train_ratings, n_users, n_items, n_partitions
        )
        item_partitions = _build_item_partition_matrices(
            train_user_idx, train_item_idx, train_ratings, n_users, n_items, n_partitions
        )

        if n_workers == 1:
            user_blocks = [
                _update_user_partition_worker(p, item_factors, regularization, alpha)
                for p in user_partitions
            ]
            updated_user_factors = np.vstack(user_blocks)[:n_users]
            item_blocks = [
                _update_item_partition_worker(p, updated_user_factors, regularization, alpha)
                for p in item_partitions
            ]
            updated_item_factors = np.vstack(item_blocks)[:n_items]
            return updated_user_factors, updated_item_factors

        # "spawn" so workers do not inherit ZenML signal handlers
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) as executor:
            user_blocks = list(
                executor.map(
                    _update_user_partition_worker,
                    user_partitions,
                    repeat(item_factors),
                    repeat(regularization),
                    repeat(alpha),
                )
            )
            updated_user_factors = np.vstack(user_blocks)[:n_users]
            item_blocks = list(
                executor.map(
                    _update_item_partition_worker,
                    item_partitions,
                    repeat(updated_user_factors),
                    repeat(regularization),
                    repeat(alpha),
                )
            )
            updated_item_factors = np.vstack(item_blocks)[:n_items]

        return updated_user_factors, updated_item_factors

    @classmethod
    def train(
        cls,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        rank: int,
        regularization: float,
        alpha: float,
        n_iter: int,
        n_workers: int = 1,
        initial_factors: tuple[np.ndarray, np.ndarray] | None = None,
        start_epoch: int = 0,
        seed: int = 42,
        eval_every_n_epochs: int = 1,
        epoch_end_callback: Callable[[int, float], None] | None = None,
        checkpoint_callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, float], float]:
        """
        Train ALS for n_iter epochs and return final factors + RMSE scores.

        Shared entrypoint for pipeline training and HPO. Supports checkpoint
        resume via initial_factors + start_epoch.
        """
        from workflows.matrix_factorization.utils.als_numba import warmup_jit

        if initial_factors is not None:
            user_factors, item_factors = initial_factors
        else:
            n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
            n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1
            warmup_jit(rank=min(rank, 20))
            user_factors, item_factors = cls.initialize_factors(
                n_users=n_users, n_items=n_items, rank=rank, seed=seed
            )

        rmse = float("inf")
        scores: dict[int, float] = {}

        for epoch in range(start_epoch, n_iter):
            user_factors, item_factors = cls.train_epoch(
                train_data=train_data,
                user_factors=user_factors,
                item_factors=item_factors,
                regularization=regularization,
                alpha=alpha,
                n_workers=n_workers,
            )

            if eval_every_n_epochs > 0 and (epoch + 1) % eval_every_n_epochs == 0:
                rmse = cls.compute_rmse(
                    val_data=val_data, user_factors=user_factors, item_factors=item_factors
                )
                scores[epoch] = rmse

            if epoch_end_callback is not None:
                epoch_end_callback(epoch, rmse)

            if checkpoint_callback is not None:
                checkpoint_callback(epoch, user_factors, item_factors)

        return user_factors, item_factors, scores, rmse
