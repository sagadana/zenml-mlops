"""
models/als_implicit_recommender.py

ALSImplicitRecommender — implicit-library-based ALS recommendation model.

Extends BaseRecommender. Training delegates to `implicit.als.AlternatingLeastSquares`,
which is highly optimised (BLAS-backed, GPU-optional). The confidence matrix follows
the Hu–Koren–Volinsky formulation: c_ui = 1 + alpha * r_ui.

Training inputs/outputs are identical in shape to ALSRecommender, so the two models
are drop-in replacements — change `recommender_class_name` in the train_als and
register_model steps and nothing else needs to change.

Warm-start / checkpoint resume:
  Setting user_factors / item_factors on the implicit model before fit() causes the
  library to skip random re-initialisation and continue from the provided matrices.
  The train_als step uses this to resume from epoch checkpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    EpochMetricSource,
    EpochState,
    EpochStates,
)

logger = logging.getLogger(__name__)


def _build_confidence_matrix(
    train_data: pd.DataFrame,
    alpha: float,
    n_users: int,
    n_items: int,
) -> csr_matrix:
    """
    Build a sparse user-item confidence matrix.

    Values = alpha * rating. The implicit library adds 1 internally to yield
    c_ui = 1 + alpha * r_ui, matching the H.K.V. formulation.

    Args:
        train_data: DataFrame with user_idx, item_idx, rating columns.
        alpha: Confidence scaling factor.
        n_users: Total number of users (matrix row count).
        n_items: Total number of items (matrix column count).

    Returns:
        CSR matrix of shape (n_users, n_items) with float32 values.
    """
    user_idx = train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int32)
    item_idx = train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int32)
    ratings = train_data[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)
    confidence = alpha * ratings
    return csr_matrix(
        (confidence, (user_idx, item_idx)), shape=(n_users, n_items), dtype=np.float32
    )


@dataclass(repr=False)
class ALSImplicitRecommender(BaseRecommender):
    """
    Implicit-library ALS recommendation model.

    Training uses `implicit.als.AlternatingLeastSquares` (CPU BLAS or GPU).
    All inference methods (predict, batch_predict, get_similar_items, compute_rmse)
    are inherited from BaseRecommender.

    Warm-start from checkpoints is supported by setting user_factors / item_factors
    on the implicit model before calling fit().
    """

    @classmethod
    def train(
        cls,
        factors: int,
        regularization: float,
        alpha: float,
        n_iter: int,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        n_workers: int = 1,
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

        import implicit
        import threadpoolctl

        from workflows.matrix_factorization.models.numba import warmup_jit

        # Limit the number of threads used by BLAS libraries (e.g., OpenBLAS, MKL) to 1.
        # This prevents oversubscription of CPU cores when using ThreadPoolExecutor for parallel evaluation.
        threadpoolctl.threadpool_limits(1, "blas")

        # Warm up the Numba JIT compiler for computing metrics
        warmup_jit()

        n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
        n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1
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

        user_item_csr = _build_confidence_matrix(train_data, alpha, n_users, n_items)

        model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=remaining_iters,
            num_threads=max(1, n_workers),
            use_gpu=use_cuda_gpu,  # GPU support is optional; requires cupy and a CUDA-capable GPU
            random_state=seed,
            # enables implicit's built-in loss callback
            calculate_training_loss=True,
        )

        # Warm-start: set factors from checkpoint so implicit skips re-initialisation
        if initial_factors is not None:
            model.user_factors = initial_factors[0].astype(np.float32)
            model.item_factors = initial_factors[1].astype(np.float32)

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

        # Define the callback function for implicit's fit() method. This is called after each epoch.
        def _on_iteration(iteration: int, elapsed: float, loss: float) -> None:
            nonlocal last_state
            nonlocal states
            nonlocal val_data

            epoch = start_epoch + iteration  # global epoch index

            # Update training state
            last_state = EpochState(
                epoch=epoch,
                k=k,
                loss=loss,
                elapsed_time=elapsed,
                rmse=0,
                precision_at_k=0,
                recall_at_k=0,
                ndcg_at_k=0,
                metrics_source=metrics_source,
            )

            # Evaluate on validation set if requested
            if (
                eval_every_n_epochs > 0
                and (epoch + 1) % eval_every_n_epochs == 0
                and val_data is not None
                and not val_data.empty
            ):
                ufs = np.array(model.user_factors, dtype=np.float32)
                ifs = np.array(model.item_factors, dtype=np.float32)

                # Compute ranking metrics on the validation set
                rmse, precision, recall, ndcg = cls.compute_metrics(
                    user_ids=np.asarray(
                        val_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].values, dtype=np.int32
                    ),
                    item_ids=np.asarray(
                        val_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].values, dtype=np.int32
                    ),
                    ratings=np.asarray(
                        val_data[CFG_FEATURES_FIELD_NAMES.RATING.value].values, dtype=np.float32
                    ),
                    user_factors=ufs,
                    item_factors=ifs,
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
                ufs = np.array(model.user_factors, dtype=np.float32)
                ifs = np.array(model.item_factors, dtype=np.float32)
                checkpoint_callback(last_state, ufs, ifs)

        # Train the model with the callback for per-epoch evaluation and checkpointing
        model.fit(user_item_csr, show_progress=True, callback=_on_iteration)

        user_factors = np.array(model.user_factors, dtype=np.float32)
        item_factors = np.array(model.item_factors, dtype=np.float32)

        return user_factors, item_factors, states
