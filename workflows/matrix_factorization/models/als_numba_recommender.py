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
from tqdm import tqdm

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    EpochMetricSource,
    EpochState,
    EpochStates,
)
from workflows.matrix_factorization.models.numba import (
    fill_item_partition,
    fill_user_partition,
    warmup_jit,
)

# Warm up the Numba JIT compiler
warmup_jit()

logger = logging.getLogger(__name__)
_MP_CONTEXT = mp.get_context("spawn")


def _update_user_partition_worker(
    partition: np.ndarray,
    item_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """Worker entrypoint: update user factors for one user-partition matrix."""
    from workflows.matrix_factorization.models.numba import solve_user_factors

    return solve_user_factors(partition, item_factors, regularization, alpha)


def _update_item_partition_worker(
    partition: np.ndarray,
    user_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """Worker entrypoint: update item factors for one item-partition matrix."""
    from workflows.matrix_factorization.models.numba import solve_item_factors

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

    partition_size = (n_items + n_partitions - 1) // n_partitions
    for p in range(n_partitions):
        i_start = p * partition_size
        i_end = min(i_start + partition_size, n_items)
        yield fill_item_partition(user_indices, item_indices, ratings, i_start, i_end, n_users)


@dataclass(repr=False)
class ALSNumbaRecommender(BaseRecommender):
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
        factors: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize user/item latent factors with small Gaussian noise."""
        rng = np.random.default_rng(seed)
        user_factors = (rng.standard_normal((n_users, factors)) * 0.01).astype(np.float32)
        item_factors = (rng.standard_normal((n_items, factors)) * 0.01).astype(np.float32)
        return user_factors, item_factors

    @classmethod
    def train_epoch(
        cls,
        train_data: pd.DataFrame,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        regularization: float,
        alpha: float,
        n_workers: int = 8,
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
            # Solve user factors for each partition sequentially
            user_blocks = [
                _update_user_partition_worker(p, item_factors, regularization, alpha)
                for p in user_partitions
            ]
            updated_user_factors = np.vstack(user_blocks)[:n_users]

            # Solve item factors for each partition sequentially using the updated user factors
            item_blocks = [
                _update_item_partition_worker(p, updated_user_factors, regularization, alpha)
                for p in item_partitions
            ]
            updated_item_factors = np.vstack(item_blocks)[:n_items]
            return updated_user_factors, updated_item_factors

        # "spawn" so workers do not inherit the main process's memory (e.g., large train_data DataFrame)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CONTEXT) as executor:
            # Solve user factors for each partition in parallel
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

            # Solve item factors for each partition in parallel using the updated user factors
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
        factors: int,
        regularization: float,
        alpha: float,
        n_iter: int,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        n_workers: int = 8,
        start_epoch: int = 0,
        seed: int = 42,
        k: int = 10,
        initial_factors: tuple[np.ndarray, np.ndarray] | None = None,
        use_cuda_gpu: bool = False,
        eval_every_n_epochs: int = 1,
        epoch_end_callback: Callable[[EpochState], None] | None = None,
        checkpoint_every_n_epochs: int = 1,
        checkpoint_callback: Callable[[EpochState, np.ndarray, np.ndarray], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, EpochStates]:

        remaining_iters = n_iter - start_epoch
        metrics_source: EpochMetricSource = "train"

        if remaining_iters <= 0:
            logger.info(
                "All %d epochs already completed (start_epoch=%d). Returning checkpoint factors.",
                n_iter,
                start_epoch,
            )
            if initial_factors is not None:
                return (initial_factors[0], initial_factors[1], EpochStates([]))

            raise ValueError(
                f"start_epoch={start_epoch} >= n_iter={n_iter} but no initial_factors provided."
            )

        # Warm-start: set factors from checkpoint so implicit skips re-initialisation
        if initial_factors is not None:
            user_factors = initial_factors[0].astype(np.float32)
            item_factors = initial_factors[1].astype(np.float32)
        else:
            n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
            n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1
            user_factors, item_factors = cls.initialize_factors(
                n_users=n_users, n_items=n_items, factors=factors, seed=seed
            )

        # If no validation data is provided, use a random sample of training data for validation metrics
        if val_data is None or val_data.empty:
            val_data = train_data.sample(frac=0.2, random_state=seed)
            logger.warning(
                "No validation data provided. Using a random sample of training data for validation metrics."
            )
        else:
            metrics_source = "val"
            logger.info(
                "Validation data provided with %d rows. Using for validation metrics.",
                len(val_data),
            )

        # Track per-epoch state
        states: EpochStates = EpochStates([])
        last_state: EpochState = EpochState(
            epoch=start_epoch,
            k=k,
            loss=0,
            elapsed_time=0,
            rmse=0,
            precision_at_k=0,
            recall_at_k=0,
            ndcg_at_k=0,
            metrics_source=metrics_source,
        )

        logger.debug("Running %i ALS iterations", remaining_iters)
        with tqdm(total=remaining_iters, disable=False) as progress:
            for epoch in range(start_epoch, n_iter):
                start_time = pd.Timestamp.now()
                user_factors, item_factors = cls.train_epoch(
                    train_data=train_data,
                    user_factors=user_factors,
                    item_factors=item_factors,
                    regularization=regularization,
                    alpha=alpha,
                    n_workers=n_workers,
                )

                # Compute training loss (WMSE) for this epoch
                loss = cls.compute_wmse(
                    user_indices=np.asarray(
                        train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].values, dtype=np.int32
                    ),
                    item_indices=np.asarray(
                        train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].values, dtype=np.int32
                    ),
                    ratings=np.asarray(
                        train_data[CFG_FEATURES_FIELD_NAMES.RATING.value].values, dtype=np.float32
                    ),
                    user_factors=user_factors,
                    item_factors=item_factors,
                    alpha=alpha,
                )

                # Update training state
                last_state = EpochState(
                    epoch=epoch,
                    k=k,
                    loss=loss,
                    elapsed_time=(pd.Timestamp.now() - start_time).total_seconds(),
                    rmse=0,
                    precision_at_k=0,
                    recall_at_k=0,
                    ndcg_at_k=0,
                    metrics_source=metrics_source,
                )

                progress.update(1)
                progress.set_postfix(
                    {
                        "loss": f"{last_state.loss:.4f}",
                    }
                )

                # Evaluate on validation set if requested
                if (
                    eval_every_n_epochs > 0
                    and (epoch + 1) % eval_every_n_epochs == 0
                    and val_data is not None
                    and not val_data.empty
                ):
                    # Compute ranking metrics on the validation set
                    rmse, precision, recall, ndcg = cls.compute_metrics(
                        user_indices=np.asarray(
                            val_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].values, dtype=np.int32
                        ),
                        item_indices=np.asarray(
                            val_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].values, dtype=np.int32
                        ),
                        ratings=np.asarray(
                            val_data[CFG_FEATURES_FIELD_NAMES.RATING.value].values, dtype=np.float32
                        ),
                        user_factors=user_factors,
                        item_factors=item_factors,
                        k=k,
                    )

                    # Update the last state with computed metrics
                    last_state.rmse = rmse
                    last_state.precision_at_k = precision
                    last_state.recall_at_k = recall
                    last_state.ndcg_at_k = ndcg

                    # Append the last state to the list of states after evaluation
                    states.append(last_state)

                    # Only invoke the epoch_end_callback if provided and should evaluate
                    if epoch_end_callback is not None:
                        epoch_end_callback(last_state)

                # Invoke the checkpoint callback if provided and should checkpoint
                if (
                    checkpoint_callback is not None
                    and checkpoint_every_n_epochs > 0
                    and (last_state.epoch + 1) % checkpoint_every_n_epochs == 0
                ):
                    ufs = np.array(user_factors, dtype=np.float32)
                    ifs = np.array(item_factors, dtype=np.float32)
                    checkpoint_callback(last_state, ufs, ifs)

        # Ensure the last state is appended if it wasn't already (e.g., if no evaluation was done on the last epoch)
        if len(states) == 0 or states[-1].epoch != last_state.epoch:
            states.append(last_state)

        return user_factors, item_factors, states
