"""
models/base_recommender.py

BaseRecommender — abstract base class for recommendation models.

Provides the shared inference interface (predict, batch_predict, get_similar_items,
compute_rmse) and the abstract train() classmethod that each subclass implements.

Pydantic output types are defined here and re-exported from als_numba_recommender.py
for backward compatibility.

To swap the model used by the training pipeline, change the `recommender_class_name`
parameter in the train_als and register_model steps — no other code changes needed.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES


class PredictionUser(BaseModel):
    """Single user recommendation with score."""

    user_id: int
    score: float


class PredictionItem(BaseModel):
    """Single item recommendation with score."""

    item_id: int
    score: float


class BatchPredictions(BaseModel):
    """Batch prediction results mapping user IDs to their recommendations."""

    predictions: dict[str, list[PredictionItem]]


class PredictionLog(BaseModel):
    """Log entry for a single user recommendation request."""

    timestamp: str
    user_id: int
    top_k: int
    latency_ms: float
    count: int
    predictions: list[PredictionItem]


@dataclass
class BaseRecommender(ABC):
    """
    Abstract base recommender.

    Subclasses implement train() to produce user/item factor matrices via different
    algorithms (e.g. numba ALS, implicit ALS). All inference methods are shared.

    Swap between implementations by changing recommender_class_name in
    the train_als and register_model steps.
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

    _item_decoder: pd.Series | None = field(default=None, repr=False, compare=False)
    _user_decoder: pd.Series | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        assert self.user_factors.ndim == 2, "user_factors must be 2D"
        assert self.item_factors.ndim == 2, "item_factors must be 2D"
        assert self.user_factors.shape[1] == self.item_factors.shape[1], (
            "user_factors and item_factors must have the same rank dimension"
        )

    # ── Properties ────────────────────────────────────────────────────────────

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_users={self.n_users}, n_items={self.n_items}, rank={self.rank}, "
            f"version={self.model_version!r})"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_known: np.ndarray | None = None,
    ) -> list[PredictionItem]:
        """
        Generate top-K item recommendations for a single user.

        Args:
            user_id: Raw (external) user ID.
            top_k: Number of recommendations to return.
            exclude_known: Array of raw item IDs to exclude (already interacted).

        Returns:
            List of PredictionItem objects, descending by score.

        Raises:
            KeyError: If user_id is not in the encoder.
        """
        if user_id not in self.user_encoder.index:
            raise KeyError(f"Unknown user_id: {user_id}. Not in training data.")

        user_idx = int(self.user_encoder[user_id])
        u = self.user_factors[user_idx]
        scores = self.item_factors @ u

        if exclude_known is not None and len(exclude_known) > 0:
            known_idxs = self.item_encoder[self.item_encoder.index.isin(exclude_known)].values
            scores[known_idxs] = -np.inf

        top_item_idxs = np.argpartition(scores, -top_k)[-top_k:]
        top_item_idxs = top_item_idxs[np.argsort(scores[top_item_idxs])[::-1]]

        return [
            PredictionItem(
                item_id=int(self.item_decoder[idx]),
                score=float(scores[idx]),
            )
            for idx in top_item_idxs
        ]

    def batch_predict(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
    ) -> BatchPredictions:
        """
        Generate top-K recommendations for a batch of users.

        Args:
            user_ids: Array of raw user IDs.
            top_k: Number of recommendations per user.

        Returns:
            BatchPredictions mapping user_id (str) to list of PredictionItem.
            Unknown users map to empty lists.
        """
        predictions_dict: dict[str, list[PredictionItem]] = {}
        for uid in user_ids:
            try:
                predictions_dict[str(uid)] = self.predict(uid, top_k=top_k)
            except KeyError:
                predictions_dict[str(uid)] = []
        return BatchPredictions(predictions=predictions_dict)

    def get_similar_items(self, item_id: int, top_k: int = 10) -> list[PredictionItem]:
        """
        Find the most similar items using cosine similarity on item factor matrix.

        Args:
            item_id: Raw (external) item ID.
            top_k: Number of similar items to return (excluding item_id itself).

        Returns:
            List of PredictionItem objects.

        Raises:
            KeyError: If item_id is not in the encoder.
        """
        if item_id not in self.item_encoder.index:
            raise KeyError(f"Unknown item_id: {item_id}. Not in training data.")

        item_idx = int(self.item_encoder[item_id])
        v = self.item_factors[item_idx]
        norms = np.linalg.norm(self.item_factors, axis=1)
        v_norm = np.linalg.norm(v)

        with np.errstate(invalid="ignore", divide="ignore"):
            scores = (self.item_factors @ v) / (norms * v_norm + 1e-10)

        scores[item_idx] = -np.inf

        top_idxs = np.argpartition(scores, -top_k)[-top_k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        return [
            PredictionItem(
                item_id=int(self.item_decoder[idx]),
                score=float(scores[idx]),
            )
            for idx in top_idxs
        ]

    def get_similar_users(self, user_id: int, top_k: int = 10) -> list[PredictionUser]:
        """
        Find the most similar users using cosine similarity on user factor matrix.

        Args:
            user_id: Raw (external) user ID.
            top_k: Number of similar users to return (excluding user_id itself).

        Returns:
            List of PredictionUser objects.

        Raises:
            KeyError: If user_id is not in the encoder.
        """
        if user_id not in self.user_encoder.index:
            raise KeyError(f"Unknown user_id: {user_id}. Not in training data.")

        user_idx = int(self.user_encoder[user_id])
        u = self.user_factors[user_idx]
        norms = np.linalg.norm(self.user_factors, axis=1)
        u_norm = np.linalg.norm(u)

        with np.errstate(invalid="ignore", divide="ignore"):
            scores = (self.user_factors @ u) / (norms * u_norm + 1e-10)

        scores[user_idx] = -np.inf

        top_idxs = np.argpartition(scores, -top_k)[-top_k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]

        return [
            PredictionUser(
                user_id=int(self.user_decoder[idx]),
                score=float(scores[idx]),
            )
            for idx in top_idxs
        ]

    # ── Evaluation ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_rmse(
        val_data: pd.DataFrame,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
    ) -> float:
        """
        Compute validation RMSE for the given factor matrices.

        Validation IDs are clipped to matrix bounds to handle subsampled / HPO runs.
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

    # ── Training (abstract) ───────────────────────────────────────────────────

    @classmethod
    @abstractmethod
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
        Train the model and return factors + validation scores.

        Args:
            train_data: Training ratings DataFrame.
            val_data: Validation ratings DataFrame.
            rank: Latent factor dimensionality.
            regularization: L2 regularization coefficient.
            alpha: Implicit feedback confidence weight (c_ui = 1 + alpha * r_ui).
            n_iter: Total number of training epochs.
            n_workers: Number of parallel workers (interpretation depends on subclass).
            initial_factors: (user_factors, item_factors) for warm-start / checkpoint resume.
            start_epoch: First epoch to execute (0 = fresh start, k = resume after epoch k-1).
            seed: Random seed for reproducible initialization.
            eval_every_n_epochs: Compute validation RMSE every N epochs.
            epoch_end_callback: Called as fn(epoch, rmse) after each evaluated epoch.
                Used by Optuna HPO for pruning decisions.
            checkpoint_callback: Called as fn(epoch, user_factors, item_factors).
                Used by train_als step to persist intermediate factor matrices.

        Returns:
            Tuple of (user_factors, item_factors, {epoch: rmse}, final_rmse).
        """
        ...


def load_recommender_class(class_path: str) -> type[BaseRecommender]:
    """
    Dynamically import a BaseRecommender subclass from a fully-qualified dotted path.

    Callers only need to know about BaseRecommender — the concrete class is resolved
    at runtime via importlib, so new implementations can be added without touching
    any existing code.

    Args:
        class_path: Fully-qualified dotted path to the class, e.g.
            'workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender'

    Returns:
        The resolved class (verified to be a BaseRecommender subclass).

    Raises:
        ValueError: If class_path has no module component.
        ImportError: If the module cannot be imported.
        AttributeError: If the class name is not found in the module.
        TypeError: If the resolved object is not a BaseRecommender subclass.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"class_path must be a fully-qualified dotted path "
            f"(e.g. 'my.module.MyClass'), got: {class_path!r}"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseRecommender)):
        raise TypeError(
            f"{class_path!r} resolved to {cls!r}, which is not a BaseRecommender subclass."
        )
    return cls
