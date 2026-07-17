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
from workflows.matrix_factorization.models.base_recommender import BaseRecommender

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
        Train via the implicit library and return final factors + RMSE scores.

        The confidence matrix follows H.K.V.: c_ui = 1 + alpha * r_ui.
        Warm-start from initial_factors is supported for checkpoint resume.
        Per-epoch callbacks are invoked via implicit's built-in callback hook.

        Args:
            initial_factors: (user_factors, item_factors) for warm-start / resume.
            start_epoch: Number of already-completed epochs. The model trains
                (n_iter - start_epoch) remaining iterations starting from initial_factors.
            epoch_end_callback: fn(epoch, rmse) — used by Optuna pruning in HPO.
            checkpoint_callback: fn(epoch, user_factors, item_factors) — used for checkpointing.
        """
        import implicit

        n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
        n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1
        remaining_iters = n_iter - start_epoch

        if remaining_iters <= 0:
            logger.info(
                "All %d epochs already completed (start_epoch=%d). Returning checkpoint factors.",
                n_iter,
                start_epoch,
            )
            if initial_factors is not None:
                return initial_factors[0], initial_factors[1], {}, float("inf")
            raise ValueError(
                f"start_epoch={start_epoch} >= n_iter={n_iter} but no initial_factors provided."
            )

        user_item_csr = _build_confidence_matrix(train_data, alpha, n_users, n_items)

        model = implicit.als.AlternatingLeastSquares(
            factors=rank,
            regularization=regularization,
            iterations=remaining_iters,
            num_threads=max(1, n_workers),
            # use_gpu=False, # GPU support is optional; requires cupy and a CUDA-capable GPU
            random_state=seed,
            # enables implicit's built-in loss callback
            calculate_training_loss=True,
        )

        # Warm-start: set factors from checkpoint so implicit skips re-initialisation
        if initial_factors is not None:
            model.user_factors = initial_factors[0].astype(np.float32)
            model.item_factors = initial_factors[1].astype(np.float32)

        val_rmse_scores: dict[int, float] = {}
        last_rmse = float("inf")

        def _on_iteration(iteration: int, elapsed: float, loss) -> None:
            nonlocal last_rmse
            epoch = start_epoch + iteration  # global epoch index

            # Evaluate on validation set if requested
            should_eval = eval_every_n_epochs > 0 and (iteration + 1) % eval_every_n_epochs == 0
            if should_eval:
                uf = np.array(model.user_factors, dtype=np.float32)
                if_arr = np.array(model.item_factors, dtype=np.float32)
                last_rmse = cls.compute_rmse(
                    val_data=val_data,
                    user_factors=uf,
                    item_factors=if_arr,
                )
                val_rmse_scores[epoch] = loss
                logger.info(
                    "Epoch %d/%d: Loss = %.4f, Val RMSE = %.4f, (elapsed %.1fs)",
                    epoch + 1,
                    n_iter,
                    loss,
                    last_rmse,
                    elapsed,
                )

            if epoch_end_callback is not None:
                epoch_end_callback(epoch, last_rmse)

            if checkpoint_callback is not None:
                uf = np.array(model.user_factors, dtype=np.float32)
                if_arr = np.array(model.item_factors, dtype=np.float32)
                checkpoint_callback(epoch, uf, if_arr)

        model.fit(user_item_csr, show_progress=False, callback=_on_iteration)

        user_factors = np.array(model.user_factors, dtype=np.float32)
        item_factors = np.array(model.item_factors, dtype=np.float32)

        return user_factors, item_factors, val_rmse_scores, last_rmse
