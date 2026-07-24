"""
models/base_recommender.py

BaseRecommender — abstract base class for recommendation models.

Provides the shared inference interface (predict, batch_predict, get_similar_items,
compute_rmse) and the abstract train() classmethod that each subclass implements.

Pydantic output types are defined here.

To swap the model used by the training pipeline, change the `recommender_class_name`
parameter in the train_als and register_model steps — no other code changes needed.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, RootModel

from workflows.matrix_factorization.models.numba import (
    compute_ranking_metrics,
    compute_rmse,
)

type EpochMetricSource = Literal[
    "train", "val"
]  # Indicates whether metrics are from training or validation


class EpochState(BaseModel):
    """State of a single training epoch."""

    epoch: int
    loss: float
    k: int
    rmse: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    elapsed_time: float
    metrics_source: EpochMetricSource


class EpochStates(RootModel):
    root: list[EpochState]

    def __iter__(self) -> Iterator[EpochState]:  # pyright: ignore[reportIncompatibleMethodOverride]
        yield from self.root

    def __getitem__(self, index: int) -> EpochState:
        return self.root[index]

    def __len__(self) -> int:
        return len(self.root)

    def append(self, state: EpochState) -> None:
        self.root.append(state)


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
    model_name: str
    model_version: str


class Hyperparameters(BaseModel):
    """Hyperparameters for training a recommender."""

    factors: int
    regularization: float
    alpha: float
    n_iter: int


class ModelMetrics(BaseModel):
    """Evaluation metrics for a trained recommender."""

    rmse: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float


@dataclass
class BaseRecommender(ABC):
    """
    Abstract base recommender.

    Subclasses implement train() to produce user/item factor matrices via different
    algorithms. All inference methods are shared.

    """

    user_factors: np.ndarray
    item_factors: np.ndarray

    user_encoder: pd.Series
    item_encoder: pd.Series

    params: Hyperparameters

    metrics: ModelMetrics | None = None

    _item_decoder: pd.Series | None = field(default=None, repr=False, compare=False)
    _user_decoder: pd.Series | None = field(default=None, repr=False, compare=False)

    name: str = ""
    version: str = ""
    promoted: bool = False

    def __post_init__(self) -> None:
        assert self.user_factors.ndim == 2, "user_factors must be 2D"
        assert self.item_factors.ndim == 2, "item_factors must be 2D"
        assert self.user_factors.shape[1] == self.item_factors.shape[1], (
            "user_factors and item_factors must have the same factors dimension"
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
            f"name={self.name!r}, version={self.version!r}, promoted={self.promoted}, "
            f"n_users={self.n_users}, n_items={self.n_items}, "
            f"hyperparameters={self.params.model_dump_json()}, "
            f"metrics={self.metrics.model_dump_json() if self.metrics else None})"
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def scores(self, user_idx: int) -> np.ndarray:
        """
        Compute scores for all items for a given user index.

        Args:
            user_idx: User index (0 <= user_idx < n_users).

        Returns:
            Array of scores for all items (length n_items).
        """
        u = self.user_factors[user_idx]
        return self.item_factors @ u

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
        scores = self.scores(user_idx)

        if exclude_known is not None and len(exclude_known) > 0:
            known_idxs = np.asarray(
                self.item_encoder[self.item_encoder.index.isin(exclude_known)].values
            )
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
        max_size_per_worker: int = 64,
    ) -> BatchPredictions:
        """
        Generate top-K recommendations for a batch of users.

        Splits user_ids into chunks of at most max_size_per_worker and processes
        each chunk in a separate thread. The per-user item scoring (item_factors @ u)
        uses BLAS and releases the GIL, so threads run truly in parallel.

        Args:
            user_ids: Array of raw user IDs.
            top_k: Number of recommendations per user.
            max_size_per_worker: Max users per thread. n_threads is derived as
                ceil(len(user_ids) / max_size_per_worker). No threading when
                len(user_ids) <= max_size_per_worker.

        Returns:
            BatchPredictions mapping user_id (str) to list of PredictionItem.
            Unknown users map to empty lists.
        """

        def _predict_chunk(chunk: np.ndarray) -> dict[str, list[PredictionItem]]:
            result: dict[str, list[PredictionItem]] = {}
            for uid in chunk:
                try:
                    result[str(uid)] = self.predict(uid, top_k=top_k)
                except KeyError:
                    result[str(uid)] = []
            return result

        n_users = len(user_ids)
        n_threads = (n_users + max_size_per_worker - 1) // max_size_per_worker

        if n_threads <= 1:
            return BatchPredictions(predictions=_predict_chunk(user_ids))

        chunks = np.array_split(user_ids, n_threads)
        predictions_dict: dict[str, list[PredictionItem]] = {}
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for chunk_result in pool.map(_predict_chunk, chunks):
                predictions_dict.update(chunk_result)

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

    @classmethod
    def compute_metrics(
        cls,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        k: int = 10,
    ) -> tuple[float, float, float, float]:
        """
        Compute RMSE, Precision@K, Recall@K, NDCG@K averaged over users.

        Args:
            user_ids: (n_ratings,) int32 array of user indices (must be within bounds).
            item_ids: (n_ratings,) int32 array of item indices (must be within bounds).
            ratings: (n_ratings,) float32 array of ratings.
            user_factors: (n_users, factors) float32 array of user factors.
            item_factors: (n_items, factors) float32 array of item factors.
            k: K for ranking metrics.

        Items with index >= n_items are silently skipped by the ranking kernel (OOV guard).

        Returns:
            (rmse, precision_at_k, recall_at_k, ndcg_at_k) averaged over users that have
            at least one relevant item.
        """
        sse, count = compute_rmse(user_ids, item_ids, ratings, user_factors, item_factors)
        precision, recall, ndcg = compute_ranking_metrics(
            user_ids, item_ids, user_factors, item_factors, k
        )

        rmse = float(np.sqrt(sse / count)) if count > 0 else float("inf")
        return rmse, precision, recall, ndcg

    # ── Training (abstract) ───────────────────────────────────────────────────

    @classmethod
    @abstractmethod
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
        """
        Train the model and return factors + validation scores.

        Args:
            factors: Latent factor dimensionality.
            regularization: L2 regularization coefficient.
            alpha: Learning rate / confidence scaling factor.
            n_iter: Total number of training epochs.
            train_data: Training ratings DataFrame.
            val_data: Validation ratings DataFrame (optional, can be None if no validation required).
            n_workers: Number of parallel workers (interpretation depends on subclass).
            start_epoch: First epoch to execute (0 = fresh start, k = resume after epoch k-1).
            seed: Random seed for reproducible initialization.
            k: K for ranking metrics (Precision@K, Recall@K, NDCG@K).
            initial_factors: (user_factors, item_factors) for warm-start / checkpoint resume.
            use_cuda_gpu: If True, use GPU for training (requires cupy and CUDA-capable GPU).
            eval_every_n_epochs: Compute validation RMSE every N epochs (0 = only at end, 1 = every epoch).
            epoch_end_callback: Called as fn(EpochState) after each evaluated epoch.
                Used by Optuna HPO for pruning decisions.
            checkpoint_every_n_epochs: Save checkpoint every N epochs (0 = disable, 1 = every epoch).
            checkpoint_callback: Called as fn(EpochState, user_factors, item_factors).
                Used by train_als step to persist intermediate factor matrices.

        Returns:
            Tuple of (user_factors, item_factors, list of EpochState).
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
