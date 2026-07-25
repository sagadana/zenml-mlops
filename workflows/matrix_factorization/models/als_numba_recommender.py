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
from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from helpers.resource_monitor import capture_snapshot
from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    EpochMetricSource,
    EpochState,
    EpochStates,
)
from workflows.matrix_factorization.models.numba import (
    solve_factors_csr,
    warmup_jit,
)

# Warm up the Numba JIT compiler
warmup_jit()

logger = logging.getLogger(__name__)

# Maximum number of threads Numba can launch for parallel kernels (captured at
# import time; equal to NUMBA_NUM_THREADS). Used to clamp the per-run worker count.
_MAX_NUMBA_THREADS = numba.get_num_threads()

# CSR triple: (indptr, indices, data) describing a sparse rating matrix.
CSRMatrix = tuple[np.ndarray, np.ndarray, np.ndarray]


def _build_csr(
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    ratings: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> CSRMatrix:
    """
    Build a sorted CSR (indptr, indices, data) triple for the ALS solve kernel.

    Rows index the entity being updated (users or items); columns index the
    fixed factor matrix. Duplicate (row, col) pairs are summed and column
    indices sorted so the kernel can stream each row contiguously.
    """
    mat = csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )
    mat.sum_duplicates()
    mat.sort_indices()
    return (
        mat.indptr.astype(np.int64),
        mat.indices.astype(np.int64),
        mat.data.astype(np.float32),
    )


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
        user_csr: CSRMatrix,
        item_csr: CSRMatrix,
        item_factors: np.ndarray,
        regularization: float,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one ALS epoch: update user factors, then item factors.

        Both half-steps stream a sparse CSR matrix into ``solve_factors_csr``,
        which parallelises across rows internally via Numba ``prange`` — no
        dense partitions and no per-epoch process pool. The item update uses
        the freshly updated user factors (Gauss–Seidel style).
        """
        user_indptr, user_cols, user_data = user_csr
        updated_user_factors = solve_factors_csr(
            user_indptr, user_cols, user_data, item_factors, regularization, alpha
        )

        item_indptr, item_cols, item_data = item_csr
        updated_item_factors = solve_factors_csr(
            item_indptr, item_cols, item_data, updated_user_factors, regularization, alpha
        )

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

        # TODO: Add GPU support for the ALS solve kernel using Numba's CUDA target.
        # This would require implementing a separate CUDA kernel for the ALS solve and managing data transfer between host and device.

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

        n_users = user_factors.shape[0]
        n_items = item_factors.shape[0]

        # Configure Numba's parallel thread count for the ALS solve kernels.
        numba.set_num_threads(max(1, min(n_workers, _MAX_NUMBA_THREADS)))

        # Build both CSR orientations once — the interaction data is static
        # across epochs, so there is no need to rebuild them per epoch.
        csr_user_idx = train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int64)
        csr_item_idx = train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int64)
        csr_ratings = train_data[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)
        user_csr = _build_csr(csr_user_idx, csr_item_idx, csr_ratings, n_users, n_items)
        item_csr = _build_csr(csr_item_idx, csr_user_idx, csr_ratings, n_items, n_users)

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
                    user_csr=user_csr,
                    item_csr=item_csr,
                    item_factors=item_factors,
                    regularization=regularization,
                    alpha=alpha,
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

                # Capture resource usage for this epoch
                snap = capture_snapshot(use_gpu=use_cuda_gpu)

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
                    cpu_percent=snap.cpu_percent,
                    memory_mb=snap.memory_mb,
                    gpu_memory_mb=snap.gpu_memory_mb,
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
