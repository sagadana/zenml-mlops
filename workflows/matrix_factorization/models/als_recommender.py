"""
models/als_recommender.py

ALSRecommender — serializable wrapper around trained ALS factor matrices.

Training algorithm:
  - Implicit-feedback ALS (Hu, Koren, Volinsky style confidence weighting).
  - Alternates user update then item update each epoch.
  - Uses partitioned dense blocks + ProcessPoolExecutor for process-level parallelism.
  - Uses Numba kernels inside each worker for numerical solve hot paths.

Holds:
  - user_factors: np.ndarray (n_users × rank)
  - item_factors: np.ndarray (n_items × rank)
  - user_encoder: pd.Series (raw_user_id → dense_user_idx)
  - item_encoder: pd.Series (raw_item_id → dense_item_idx)
  - item_decoder: pd.Series (dense_item_idx → raw_item_id)  [reverse map]

Serialized as cloudpickle by ALSRecommenderMaterializer.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat

import numpy as np
import pandas as pd

from workflows.matrix_factorization.configs import (
    CFG_BATCH_PREDICTION_FIELD_NAMES,
    CFG_FEATURES_FIELD_NAMES,
    CFG_PREDICTION_FIELD_NAMES,
)


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
) -> Generator[np.ndarray, None, None]:
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
) -> Generator[np.ndarray, None, None]:
    """Yield dense item-partition matrices for ALS item-factor updates."""
    from workflows.matrix_factorization.utils.als_numba import fill_item_partition

    partition_size = (n_items + n_partitions - 1) // n_partitions
    for p in range(n_partitions):
        i_start = p * partition_size
        i_end = min(i_start + partition_size, n_items)
        yield fill_item_partition(user_indices, item_indices, ratings, i_start, i_end, n_users)


@dataclass
class ALSRecommender:
    """
    ALS recommendation model.

    Attributes:
        user_factors: User latent factor matrix (n_users × rank). float32.
        item_factors: Item latent factor matrix (n_items × rank). float32.
        user_encoder: Maps raw user ID → dense integer index.
        item_encoder: Maps raw item ID → dense integer index.
        rank: Latent factor dimensionality.
        regularization: L2 regularization used during training.
        alpha: Confidence weighting parameter used during training.
        n_iter: Number of ALS iterations trained.
        model_version: Identifier string set by register_model step.
    """

    user_factors: np.ndarray
    item_factors: np.ndarray
    user_encoder: pd.Series
    item_encoder: pd.Series
    rank: int
    regularization: float = 0.01
    alpha: float = 1.0
    n_iter: int = 15
    model_version: str = "unknown"

    # Built lazily on first use
    _item_decoder: pd.Series | None = field(default=None, repr=False, compare=False)
    _user_decoder: pd.Series | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate shapes
        assert self.user_factors.ndim == 2, "user_factors must be 2D"
        assert self.item_factors.ndim == 2, "item_factors must be 2D"
        assert (
            self.user_factors.shape[1] == self.item_factors.shape[1]
        ), "user_factors and item_factors must have the same rank dimension"

    @property
    def n_users(self) -> int:
        return self.user_factors.shape[0]

    @property
    def n_items(self) -> int:
        return self.item_factors.shape[0]

    @property
    def item_decoder(self) -> pd.Series:
        if self._item_decoder is None:
            self._item_decoder = pd.Series(
                self.item_encoder.index.values, index=self.item_encoder.values
            )
        return self._item_decoder

    @property
    def user_decoder(self) -> pd.Series:
        if self._user_decoder is None:
            self._user_decoder = pd.Series(
                self.user_encoder.index.values, index=self.user_encoder.values
            )
        return self._user_decoder

    def predict(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_known: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Generate top-K item recommendations for a single user.

        Args:
            user_id: Raw (external) user ID.
            top_k: Number of recommendations to return.
            exclude_known: Array of raw item IDs the user has already interacted with.
                           These will be excluded from recommendations.

        Returns:
            List of dicts: [{"item_id": int, "score": float}, ...], descending by score.

        Raises:
            KeyError: If user_id is not in the encoder (unknown user).
        """
        if user_id not in self.user_encoder.index:
            raise KeyError(f"Unknown user_id: {user_id}. Not in training data.")

        user_idx = int(self.user_encoder[user_id])
        u = self.user_factors[user_idx]  # (rank,)

        # Compute scores for all items: u · Y^T  →  (n_items,)
        scores = self.item_factors @ u  # (n_items,)

        # Mask known items
        if exclude_known is not None and len(exclude_known) > 0:
            known_idxs = self.item_encoder[self.item_encoder.index.isin(exclude_known)].values
            scores[known_idxs] = -np.inf

        top_item_idxs = np.argpartition(scores, -top_k)[-top_k:]
        top_item_idxs = top_item_idxs[np.argsort(scores[top_item_idxs])[::-1]]

        return [
            {
                CFG_PREDICTION_FIELD_NAMES.ITEM_ID.value: int(self.item_decoder[idx]),
                CFG_PREDICTION_FIELD_NAMES.SCORE.value: float(scores[idx]),
            }
            for idx in top_item_idxs
        ]

    def batch_predict(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Generate top-K recommendations for a batch of users.

        Args:
            user_ids: Array of raw user IDs.
            top_k: Number of recommendations per user.

        Returns:
            List of dicts: [{"user_id": int, "recommendations": [{"item_id": int, "score": float}]}, ...]
        """
        results = []
        for uid in user_ids:
            try:
                recs = self.predict(uid, top_k=top_k)
                results.append(
                    {
                        CFG_BATCH_PREDICTION_FIELD_NAMES.USER_ID.value: int(uid),
                        CFG_BATCH_PREDICTION_FIELD_NAMES.RECOMMENDATIONS.value: recs,
                    }
                )
            except KeyError:
                results.append(
                    {
                        CFG_BATCH_PREDICTION_FIELD_NAMES.USER_ID.value: int(uid),
                        CFG_BATCH_PREDICTION_FIELD_NAMES.RECOMMENDATIONS.value: [],
                    }
                )
        return results

    def get_similar_items(self, item_id: int, top_k: int = 10) -> list[dict]:
        """
        Find the most similar items to a given item using cosine similarity
        on the item factor matrix.

        Args:
            item_id: Raw (external) item ID.
            top_k: Number of similar items to return.

        Returns:
            List of dicts: [{"item_id": int, "score": float}, ...], excluding item_id itself.

        Raises:
            KeyError: If item_id is not in the encoder.
        """
        if item_id not in self.item_encoder.index:
            raise KeyError(f"Unknown item_id: {item_id}. Not in training data.")

        item_idx = int(self.item_encoder[item_id])
        v = self.item_factors[item_idx]  # (rank,)
        norms = np.linalg.norm(self.item_factors, axis=1)
        v_norm = np.linalg.norm(v)

        # Cosine similarity: (Y · v) / (||Y|| * ||v||)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = (self.item_factors @ v) / (norms * v_norm + 1e-10)

        scores[item_idx] = -np.inf  # exclude self

        top_idxs = np.argpartition(scores, -top_k)[-top_k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        return [
            {
                CFG_PREDICTION_FIELD_NAMES.ITEM_ID.value: int(self.item_decoder[idx]),
                CFG_PREDICTION_FIELD_NAMES.SCORE.value: float(scores[idx]),
            }
            for idx in top_idxs
        ]

    def __repr__(self) -> str:
        return (
            f"ALSRecommender("
            f"n_users={self.n_users}, n_items={self.n_items}, rank={self.rank}, "
            f"version={self.model_version!r})"
        )

    @staticmethod
    def initialize_factors(
        n_users: int,
        n_items: int,
        rank: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize user/item latent factors with small Gaussian noise.

        Args:
            n_users: Number of users in the factor matrix.
            n_items: Number of items in the factor matrix.
            rank: Latent dimension.
            seed: RNG seed for reproducible initialization.

        Returns:
            Tuple of (user_factors, item_factors), both float32 arrays.
        """
        rng = np.random.default_rng(seed)
        user_factors = (rng.standard_normal((n_users, rank)) * 0.01).astype(np.float32)
        item_factors = (rng.standard_normal((n_items, rank)) * 0.01).astype(np.float32)
        return user_factors, item_factors

    @staticmethod
    def compute_rmse(
        val_data: pd.DataFrame,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
    ) -> float:
        """
        Compute validation RMSE for current factor matrices.

        Validation IDs are clipped to matrix bounds to handle subsampled/HPO runs.
        """
        from workflows.matrix_factorization.utils.als_numba import compute_rmse_block

        n_users = user_factors.shape[0]
        n_items = item_factors.shape[0]

        val_user_idx = np.clip(
            val_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].to_numpy(dtype=np.int64),
            0,
            n_users - 1,
        )
        val_item_idx = np.clip(
            val_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].to_numpy(dtype=np.int64),
            0,
            n_items - 1,
        )
        val_ratings = val_data[CFG_FEATURES_FIELD_NAMES.RATING.value].to_numpy(dtype=np.float32)
        sse, count = compute_rmse_block(
            val_user_idx,
            val_item_idx,
            val_ratings,
            user_factors,
            item_factors,
        )
        return float(np.sqrt(sse / count)) if count > 0 else float("inf")

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
            train_user_idx,
            train_item_idx,
            train_ratings,
            n_users,
            n_items,
            n_partitions,
        )

        item_partitions = _build_item_partition_matrices(
            train_user_idx,
            train_item_idx,
            train_ratings,
            n_users,
            n_items,
            n_partitions,
        )

        if n_workers == 1:
            user_blocks = [
                _update_user_partition_worker(partition, item_factors, regularization, alpha)
                for partition in user_partitions
            ]
            updated_user_factors = np.vstack(user_blocks)[:n_users]
            item_blocks = [
                _update_item_partition_worker(
                    partition, updated_user_factors, regularization, alpha
                )
                for partition in item_partitions
            ]
            updated_item_factors = np.vstack(item_blocks)[:n_items]
            return updated_user_factors, updated_item_factors

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
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
        seed: int = 42,
        eval_every_n_epochs: int = 1,
        epoch_end_callback: Callable[[int, float], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Train ALS for n_iter epochs and return final factors + latest RMSE.

        This is the shared training entrypoint used by both pipeline training
        and HPO trial execution. Optional epoch callback supports Optuna pruning.
        """
        if initial_factors is None:
            n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
            n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1
            user_factors, item_factors = cls.initialize_factors(
                n_users=n_users,
                n_items=n_items,
                rank=rank,
                seed=seed,
            )
        else:
            user_factors, item_factors = initial_factors

        rmse = float("inf")
        for epoch in range(n_iter):
            user_factors, item_factors = cls.train_epoch(
                train_data=train_data,
                user_factors=user_factors,
                item_factors=item_factors,
                regularization=regularization,
                alpha=alpha,
                n_workers=n_workers,
            )

            should_eval = eval_every_n_epochs > 0 and (epoch + 1) % eval_every_n_epochs == 0
            if should_eval:
                rmse = cls.compute_rmse(
                    val_data=val_data,
                    user_factors=user_factors,
                    item_factors=item_factors,
                )
                if epoch_end_callback is not None:
                    epoch_end_callback(epoch, rmse)

        return user_factors, item_factors, rmse
