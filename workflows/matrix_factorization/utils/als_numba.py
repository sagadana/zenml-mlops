"""
utils/als_numba.py

Numba-JIT-compiled evaluation kernels.

Provides RMSE and ranking metric computation (Precision@K, Recall@K, NDCG@K)
used during model evaluation and training callbacks.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# ── RMSE computation ────────────────────────────────────────────────────────


@njit(parallel=True, nogil=True, cache=True)
def compute_rmse_block(
    user_indices: np.ndarray,  # (n_ratings,)  int32
    item_indices: np.ndarray,  # (n_ratings,)  int32
    ratings: np.ndarray,  # (n_ratings,)  float32
    user_factors: np.ndarray,  # (n_users, rank)
    item_factors: np.ndarray,  # (n_items, rank)
) -> tuple[float, int]:
    """
    Compute sum of squared errors and count for a block of ratings.
    Returns (sse, count) — caller aggregates across blocks for RMSE.

    Args:
        user_indices: (n_ratings,) int32 array of user indices.
        item_indices: (n_ratings,) int32 array of item indices.
        ratings: (n_ratings,) float32 array of ratings.
        user_factors: (n_users, rank) float32 array of user factors.
        item_factors: (n_items, rank) float32 array of item factors.

    Returns:
        (sse, count) where sse is the sum of squared errors and count is the number of ratings.
    """
    n = len(ratings)
    sse = 0.0
    for i in prange(n):
        u = user_indices[i]
        v = item_indices[i]
        pred = 0.0
        for r in range(user_factors.shape[1]):
            pred += user_factors[u, r] * item_factors[v, r]
        diff = pred - ratings[i]
        sse += diff * diff
    return sse, n


# ── Ranking metrics ─────────────────────────────────────────────────────────


@njit(parallel=True, nogil=True, cache=True)
def compute_ranking_metrics(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    k: int = 10,
    rating_threshold: float = 3.5,
) -> tuple[float, float, float]:
    """
    Compute Precision@K, Recall@K, NDCG@K averaged over users (parallel over users).

    Args:
        user_ids: (n_ratings,) int32 array of user indices.
        item_ids: (n_ratings,) int32 array of item indices.
        ratings: (n_ratings,) float32 array of ratings.
        user_factors: (n_users, rank) float32 array of user factors.
        item_factors: (n_items, rank) float32 array of item factors.
        k: K for ranking metrics.
        rating_threshold: A rating >= rating_threshold is considered a "relevant" item.

    A rating >= rating_threshold is considered a "relevant" item.
    Items with index >= n_items are silently skipped (out-of-vocabulary guard).

    Returns:
        (precision_at_k, recall_at_k, ndcg_at_k) averaged over users that have
        at least one relevant item.
    """

    # Build offsets from sorted user_ids (CSR layout assumed) — np.unique is
    # not supported with return_index=True in Numba nopython mode.
    n_ratings = len(user_ids)
    n_unique_users = 0
    for i in range(n_ratings):
        if i == 0 or user_ids[i] != user_ids[i - 1]:
            n_unique_users += 1

    user_offsets = np.empty(n_unique_users + 1, dtype=np.int64)
    user_factor_indices = np.empty(n_unique_users, dtype=np.int32)
    uid = -1
    for i in range(n_ratings):
        if i == 0 or user_ids[i] != user_ids[i - 1]:
            uid += 1
            user_offsets[uid] = i
            user_factor_indices[uid] = user_ids[i]
    user_offsets[n_unique_users] = n_ratings
    n_items = item_factors.shape[0]

    precisions = np.zeros(n_unique_users, dtype=np.float64)
    recalls = np.zeros(n_unique_users, dtype=np.float64)
    ndcgs = np.zeros(n_unique_users, dtype=np.float64)
    valid = np.zeros(n_unique_users, dtype=np.bool_)

    # Parallel over users
    for u_local in prange(n_unique_users):
        start = user_offsets[u_local]
        end = user_offsets[u_local + 1]
        u_idx = user_factor_indices[u_local]

        if u_idx >= user_factors.shape[0]:
            continue

        # Mark relevant items and count them
        is_relevant = np.zeros(n_items, dtype=np.bool_)
        n_relevant = 0
        for i in range(start, end):
            if ratings[i] >= rating_threshold and item_ids[i] < n_items:
                is_relevant[item_ids[i]] = True
                n_relevant += 1

        if n_relevant == 0:
            continue

        # Score all items: (n_items,) = item_factors @ u
        scores = item_factors @ user_factors[u_idx]

        # argsort ascending; highest scores are at the tail
        sorted_idxs = np.argsort(scores)
        top_k = min(k, n_items)

        hits = 0
        dcg = 0.0
        for rank_pos in range(top_k):
            item_idx = sorted_idxs[n_items - 1 - rank_pos]
            if is_relevant[item_idx]:
                hits += 1
                dcg += 1.0 / np.log2(rank_pos + 2.0)

        ideal_hits = min(k, n_relevant)
        idcg = 0.0
        for i in range(ideal_hits):
            idcg += 1.0 / np.log2(i + 2.0)

        precisions[u_local] = hits / k
        recalls[u_local] = hits / n_relevant
        ndcgs[u_local] = dcg / idcg if idcg > 0.0 else 0.0
        valid[u_local] = True

    # Aggregate over valid users
    p_sum = 0.0
    r_sum = 0.0
    n_sum = 0.0
    n_valid = 0
    for i in range(n_unique_users):
        if valid[i]:
            p_sum += precisions[i]
            r_sum += recalls[i]
            n_sum += ndcgs[i]
            n_valid += 1

    if n_valid == 0:
        return 0.0, 0.0, 0.0

    return p_sum / n_valid, r_sum / n_valid, n_sum / n_valid


# ── JIT warmup ──────────────────────────────────────────────────────────────


def warmup_jit(rank: int = 10) -> None:
    """
    Pre-compile all JIT functions with small dummy data.
    Call this at Docker image build time to eliminate cold-start latency.
    """
    n_users, n_items = 8, 12
    rng = np.random.default_rng(0)

    item_factors = rng.random((n_items, rank)).astype(np.float32)
    user_factors = rng.random((n_users, rank)).astype(np.float32)

    # Warmup compute_rmse_block: 5 ratings, 8 users, 12 items
    _ = compute_rmse_block(
        np.array([0, 0, 1, 1, 1], dtype=np.int32),
        np.array([0, 1, 0, 1, 2], dtype=np.int32),
        np.array([5.0, 3.0, 4.0, 2.0, 5.0], dtype=np.float32),
        user_factors,
        item_factors,
    )

    # Warmup compute_ranking_metrics: 2 users, 5 ratings, 3 items
    _ = compute_ranking_metrics(
        np.array([0, 0, 1, 1, 1], dtype=np.int32),
        np.array([0, 1, 0, 1, 2], dtype=np.int32),
        np.array([5.0, 3.0, 4.0, 2.0, 5.0], dtype=np.float32),
        user_factors,
        item_factors,
        k=2,
        rating_threshold=3.5,
    )
