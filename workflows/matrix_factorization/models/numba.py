"""
models/numba.py

Numba-JIT-compiled evaluation kernels.

Provides RMSE and ranking metric computation (Precision@K, Recall@K, NDCG@K)
used during model evaluation and training callbacks.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# ── RMSE computation ────────────────────────────────────────────────────────


@njit(parallel=True, nogil=True, cache=True)
def compute_rmse(
    user_indices: np.ndarray,  # (n_ratings,)  int32
    item_indices: np.ndarray,  # (n_ratings,)  int32
    ratings: np.ndarray,  # (n_ratings,)  float32
    user_factors: np.ndarray,  # (n_users, rank)
    item_factors: np.ndarray,  # (n_items, rank)
) -> tuple[float, int]:
    """
    Compute sum of squared errors and count ratings.

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


@njit(cache=True)
def _build_user_offsets(
    user_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Convert a sorted user_ids array into CSR-style row offsets.

    np.unique(..., return_index=True) is not supported in Numba nopython mode,
    so we build the offsets manually with a single pass over user_ids.

    Returns:
        offsets: (n_unique + 1,) int64 — offsets[i] is the start position of
                 user i's ratings; offsets[n_unique] == len(user_ids).
        factor_indices: (n_unique,) int32 — the actual user-factor row index
                        for each local user.
        n_unique: number of distinct users.
    """
    n_ratings = len(user_ids)
    n_unique = 0
    for i in range(n_ratings):
        if i == 0 or user_ids[i] != user_ids[i - 1]:
            n_unique += 1

    offsets = np.empty(n_unique + 1, dtype=np.int64)
    factor_indices = np.empty(n_unique, dtype=np.int32)
    uid = -1
    for i in range(n_ratings):
        if i == 0 or user_ids[i] != user_ids[i - 1]:
            uid += 1
            offsets[uid] = i
            factor_indices[uid] = user_ids[i]
    offsets[n_unique] = n_ratings
    return offsets, factor_indices, n_unique


@njit(cache=True)
def _user_metrics_at_k(
    u_factor: np.ndarray,  # (rank,) — this user's latent vector
    user_item_ids: np.ndarray,  # item indices this user interacted with
    item_factors: np.ndarray,  # (n_items, rank) — all item latent vectors
    k: int,
) -> tuple[float, float, float, bool]:
    """
    Compute Precision@K, Recall@K, and NDCG@K for a single user.

    Steps:
      1. Build a boolean mask of which items the user interacted with (the "relevant" set).
      2. Score every item via a dot product with the user's latent vector.
      3. Walk the top-K items by score and count how many are relevant (hits).
      4. Derive the three metrics from hits, DCG, and IDCG.

    Returns:
        (precision, recall, ndcg, is_valid)
        is_valid is False when the user has no relevant items (nothing to evaluate).
    """
    n_items = item_factors.shape[0]

    # Step 1 — build the relevant-item mask
    is_relevant = np.zeros(n_items, dtype=np.bool_)
    n_relevant = 0
    for item_idx in user_item_ids:
        if item_idx < n_items:  # skip out-of-vocabulary items
            is_relevant[item_idx] = True
            n_relevant += 1

    if n_relevant == 0:
        return 0.0, 0.0, 0.0, False

    # Step 2 — score every item; argsort ascending so best scores sit at the tail
    scores = item_factors @ u_factor
    sorted_idxs = np.argsort(scores)
    top_k = min(k, n_items)

    # Step 3 — walk top-K (descending) and accumulate hits and DCG
    hits = 0
    dcg = 0.0
    for rank_pos in range(top_k):
        item_idx = sorted_idxs[n_items - 1 - rank_pos]
        if is_relevant[item_idx]:
            hits += 1
            dcg += 1.0 / np.log2(rank_pos + 2.0)

    # Step 4 — ideal DCG assumes all relevant items appear at the very top
    ideal_hits = min(k, n_relevant)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1.0 / np.log2(i + 2.0)

    precision = hits / k
    recall = hits / n_relevant
    ndcg = dcg / idcg if idcg > 0.0 else 0.0
    return precision, recall, ndcg, True


@njit(parallel=True, nogil=True, cache=True)
def compute_ranking_metrics(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    k: int = 10,
) -> tuple[float, float, float]:
    """
    Compute Precision@K, Recall@K, NDCG@K averaged over all users (parallel).

    Args:
        user_ids: (n_ratings,) int32 — user index per rating, sorted ascending.
        item_ids: (n_ratings,) int32 — item index per rating.
        user_factors: (n_users, rank) float32 — user latent vectors.
        item_factors: (n_items, rank) float32 — item latent vectors.
        k: cut-off rank for the metrics.

    Returns:
        (precision_at_k, recall_at_k, ndcg_at_k) averaged over users with at
        least one relevant item.
    """
    offsets, factor_indices, n_unique_users = _build_user_offsets(user_ids)

    precisions = np.zeros(n_unique_users, dtype=np.float64)
    recalls = np.zeros(n_unique_users, dtype=np.float64)
    ndcgs = np.zeros(n_unique_users, dtype=np.float64)
    valid = np.zeros(n_unique_users, dtype=np.bool_)

    for u_local in prange(n_unique_users):
        u_idx = factor_indices[u_local]
        if u_idx >= user_factors.shape[0]:
            continue

        start, end = offsets[u_local], offsets[u_local + 1]
        p, r, n, ok = _user_metrics_at_k(
            user_factors[u_idx],
            item_ids[start:end],
            item_factors,
            k,
        )
        precisions[u_local] = p
        recalls[u_local] = r
        ndcgs[u_local] = n
        valid[u_local] = ok

    # Average over users that had at least one relevant item
    p_sum = r_sum = n_sum = 0.0
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

    # Warmup compute_rmse: 5 ratings, 8 users, 12 items
    _ = compute_rmse(
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
        user_factors,
        item_factors,
        k=2,
    )
