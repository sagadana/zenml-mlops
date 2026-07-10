"""
models/als_recommender.py

ALSRecommender — serializable wrapper around trained ALS factor matrices.

Holds:
  - user_factors: np.ndarray (n_users × rank)
  - item_factors: np.ndarray (n_items × rank)
  - user_encoder: pd.Series (raw_user_id → dense_user_idx)
  - item_encoder: pd.Series (raw_item_id → dense_item_idx)
  - item_decoder: pd.Series (dense_item_idx → raw_item_id)  [reverse map]

Serialized as cloudpickle by ALSRecommenderMaterializer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from workflows.matrix_factorization.configs import (
    CFG_BATCH_PREDICTION_FIELD_NAMES,
    CFG_PREDICTION_FIELD_NAMES,
)


@dataclass
class ALSRecommender:
    """
    Trained ALS recommendation model.

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
