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
    user_factors: np.ndarray,  # (n_users, factors)
    item_factors: np.ndarray,  # (n_items, factors)
) -> float:
    """
    Compute root mean squared error.

    Args:
        user_indices: (n_ratings,) int32 array of user indices.
        item_indices: (n_ratings,) int32 array of item indices.
        ratings: (n_ratings,) float32 array of ratings.
        user_factors: (n_users, factors) float32 array of user factors.
        item_factors: (n_items, factors) float32 array of item factors.

    Returns:
        (rmse) where rmse is the root mean squared error
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

    rmse = float(np.sqrt(sse / n)) if n > 0 else float("inf")
    return rmse


# ── Weighted MSE computation ────────────────────────────────────


@njit(parallel=True, nogil=True, cache=True)
def compute_wmse(
    user_indices: np.ndarray,  # (n_ratings,)  int32
    item_indices: np.ndarray,  # (n_ratings,)  int32
    ratings: np.ndarray,  # (n_ratings,)  float32
    user_factors: np.ndarray,  # (n_users, factors)
    item_factors: np.ndarray,  # (n_items, factors)
    alpha: float = 40.0,
) -> float:
    """
    Compute the confidence-weighted mean squared error (implicit-feedback ALS).

    Implements the implicit ALS reconstruction objective:
        WMSE = Σ c_ui (p_ui - x_u·y_i)^2  /  Σ c_ui
    where the preference p_ui = 1 for every observed interaction and the
    confidence weight c_ui = 1 + alpha * r_ui (matching solve_user_factors).

    Args:
        user_indices: (n_ratings,) int32 array of user indices.
        item_indices: (n_ratings,) int32 array of item indices.
        ratings: (n_ratings,) float32 array of ratings (implicit strengths).
        user_factors: (n_users, factors) float32 array of user factors.
        item_factors: (n_items, factors) float32 array of item factors.
        alpha: Confidence weighting parameter (c_ui = 1 + alpha * r_ui).

    Returns:
        (wmse) where wmse is the confidence-weighted mean squared error.
    """
    n = len(ratings)
    weighted_sse = 0.0
    weight_sum = 0.0
    for i in prange(n):
        u = user_indices[i]
        v = item_indices[i]
        pred = 0.0
        for r in range(user_factors.shape[1]):
            pred += user_factors[u, r] * item_factors[v, r]
        confidence = 1.0 + alpha * ratings[i]
        diff = pred - 1.0  # preference p_ui = 1 for observed interactions
        weighted_sse += confidence * diff * diff
        weight_sum += confidence

    wmse = float(weighted_sse / weight_sum) if weight_sum > 0.0 else float("inf")
    return wmse


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
    u_factor: np.ndarray,  # (factors,) — this user's latent vector
    user_item_ids: np.ndarray,  # item indices this user interacted with
    item_factors: np.ndarray,  # (n_items, factors) — all item latent vectors
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

    # Step 1 — build the relevant-item mask (dedup: each item counts once)
    is_relevant = np.zeros(n_items, dtype=np.bool_)
    n_relevant = 0
    for item_idx in user_item_ids:
        if item_idx < n_items and not is_relevant[item_idx]:  # skip OOV / duplicates
            is_relevant[item_idx] = True
            n_relevant += 1

    if n_relevant == 0:
        return 0.0, 0.0, 0.0, False

    # Step 2 — score every item via a dot product with the user's latent vector
    scores = item_factors @ u_factor
    top_k = min(k, n_items)

    # Step 3 — bounded top-K selection kept in descending order.
    # A single pass with insertion into a fixed-size buffer avoids a full
    # O(n log n) argsort (and its n-length index allocation); expected cost is
    # ~O(n_items) since items rarely displace an existing top-K entry.
    top_items = np.empty(top_k, dtype=np.int64)
    top_scores = np.empty(top_k, dtype=scores.dtype)
    filled = 0
    for item_idx in range(n_items):
        s = scores[item_idx]
        if filled < top_k:
            pos = filled
            filled += 1
        elif s > top_scores[top_k - 1]:
            pos = top_k - 1
        else:
            continue
        while pos > 0 and top_scores[pos - 1] < s:
            top_scores[pos] = top_scores[pos - 1]
            top_items[pos] = top_items[pos - 1]
            pos -= 1
        top_scores[pos] = s
        top_items[pos] = item_idx

    # Step 4 — walk the selected top-K (descending) and accumulate hits and DCG
    hits = 0
    dcg = 0.0
    for rank_pos in range(top_k):
        if is_relevant[top_items[rank_pos]]:
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
        user_factors: (n_users, factors) float32 — user latent vectors.
        item_factors: (n_items, factors) float32 — item latent vectors.
        k: cut-off factors for the metrics.

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


# ── User factor update ──────────────────────────────────────────────────────


@njit(nogil=True, cache=True)
def _solve_linear_system_gaussian(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax=b with Gaussian elimination + partial pivoting.
    Implemented manually to stay fully compatible with Numba nopython mode.
    """
    n = A.shape[0]
    mat = A.copy()
    rhs = b.copy()

    # Forward elimination
    for k in range(n):
        pivot_row = k
        pivot_abs = np.abs(mat[k, k])
        for i in range(k + 1, n):
            cand = np.abs(mat[i, k])
            if cand > pivot_abs:
                pivot_abs = cand
                pivot_row = i

        if pivot_abs <= 1e-12:
            raise ValueError("Singular matrix in ALS linear solve.")

        if pivot_row != k:
            for j in range(k, n):
                tmp = mat[k, j]
                mat[k, j] = mat[pivot_row, j]
                mat[pivot_row, j] = tmp
            tmp_rhs = rhs[k]
            rhs[k] = rhs[pivot_row]
            rhs[pivot_row] = tmp_rhs

        pivot = mat[k, k]
        for i in range(k + 1, n):
            factor = mat[i, k] / pivot
            mat[i, k] = 0.0
            for j in range(k + 1, n):
                mat[i, j] -= factor * mat[k, j]
            rhs[i] -= factor * rhs[k]

    # Back substitution
    x = np.zeros(n, dtype=rhs.dtype)
    for i in range(n - 1, -1, -1):
        acc = rhs[i]
        for j in range(i + 1, n):
            acc -= mat[i, j] * x[j]
        x[i] = acc / mat[i, i]

    return x


@njit(parallel=True, nogil=True, cache=True)
def solve_user_factors(
    user_ratings: np.ndarray,  # (n_users_in_block, n_items)  float32
    item_factors: np.ndarray,  # (n_items, rank)               float32
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute updated user factor matrix for a block of users.

    For each user, solves:
        (Y^T C^u Y + λI) u = Y^T C^u p
    where C^u_ii = 1 + alpha * r_ui  (confidence weighting).

    Args:
        user_ratings: Dense rating matrix block (users × items). Zero means no rating.
        item_factors: Current item factor matrix (items × rank).
        regularization: L2 regularization parameter λ.
        alpha: Confidence weighting parameter. Higher = more weight on observed interactions.

    Returns:
        Updated user factors (n_users_in_block × rank).
    """
    n_users, _ = user_ratings.shape
    rank = item_factors.shape[1]
    user_factors = np.zeros((n_users, rank), dtype=np.float32)

    # Precompute Y^T Y (shared across all users in this block)
    YtY = item_factors.T @ item_factors  # (rank × rank)
    reg_eye = regularization * np.eye(rank, dtype=np.float32)

    for u in prange(n_users):  # parallel over users
        # Build confidence-weighted system for user u
        # Rated items only (sparse loop)
        rated_indices = np.where(user_ratings[u] > 0)[0]
        if len(rated_indices) == 0:
            continue

        Y_u = item_factors[rated_indices]  # (n_rated × rank)
        r_u = user_ratings[u, rated_indices]  # (n_rated,)
        c_u = 1.0 + alpha * r_u  # confidence weights

        # A_u = Y^T C^u Y + λI  =  Y^T Y + Y^T (C^u - I) Y + λI
        # Efficient: only loop over rated items for the correction term
        A_u = YtY + reg_eye
        for idx in range(len(rated_indices)):
            y = Y_u[idx]
            # Add (c_u - 1) * y * y^T to A_u
            delta = c_u[idx] - 1.0
            for r1 in range(rank):
                for r2 in range(rank):
                    A_u[r1, r2] += delta * y[r1] * y[r2]

        # b_u = Y^T C^u p_u  (p_u = 1 for all rated items in implicit setting)
        b_u = np.zeros(rank, dtype=np.float32)
        for idx in range(len(rated_indices)):
            for r in range(rank):
                b_u[r] += c_u[idx] * Y_u[idx, r]

        user_factors[u] = _solve_linear_system_gaussian(A_u, b_u)

    return user_factors


# ── Item factor update ──────────────────────────────────────────────────────


def solve_item_factors(
    item_ratings: np.ndarray,  # (n_items_in_block, n_users)  float32  (transposed view)
    user_factors: np.ndarray,  # (n_users, rank)               float32
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute updated item factor matrix for a block of items.
    Symmetric to solve_user_factors with roles of users/items swapped.

    Args:
        item_ratings: Dense rating matrix block (items × users). Transposed from user_ratings.
        user_factors: Current user factor matrix (users × rank).
        regularization: L2 regularization parameter λ.
        alpha: Confidence weighting parameter.

    Returns:
        Updated item factors (n_items_in_block × rank).
    """
    return solve_user_factors(item_ratings, user_factors, regularization, alpha)


# ── Partition matrix construction ───────────────────────────────────────────


@njit(nogil=True, cache=True)
def fill_user_partition(
    user_indices: np.ndarray,  # (n_ratings,)  int64
    item_indices: np.ndarray,  # (n_ratings,)  int64
    ratings: np.ndarray,  # (n_ratings,)  float32
    u_start: int,
    u_end: int,
    n_items: int,
) -> np.ndarray:
    """
    Build a dense user-partition rating matrix for a contiguous range [u_start, u_end).
    Returns a (u_end - u_start) × n_items float32 array.
    """
    R_p = np.zeros((u_end - u_start, n_items), dtype=np.float32)
    for k in range(len(user_indices)):
        u = user_indices[k]
        if u_start <= u < u_end:
            R_p[u - u_start, item_indices[k]] = ratings[k]
    return R_p


@njit(nogil=True, cache=True)
def fill_item_partition(
    user_indices: np.ndarray,  # (n_ratings,)  int64
    item_indices: np.ndarray,  # (n_ratings,)  int64
    ratings: np.ndarray,  # (n_ratings,)  float32
    i_start: int,
    i_end: int,
    n_users: int,
) -> np.ndarray:
    """
    Build a dense item-partition rating matrix (transposed) for a contiguous range [i_start, i_end).
    Returns a (i_end - i_start) × n_users float32 array.
    """
    R_p = np.zeros((i_end - i_start, n_users), dtype=np.float32)
    for k in range(len(item_indices)):
        i = item_indices[k]
        if i_start <= i < i_end:
            R_p[i - i_start, user_indices[k]] = ratings[k]
    return R_p


# ── JIT warmup ──────────────────────────────────────────────────────────────


def warmup_jit(factors: int = 10) -> None:
    """
    Pre-compile all JIT functions with small dummy data.
    Call this at Docker image build time to eliminate cold-start latency.
    """
    n_users, n_items = 8, 12
    rng = np.random.default_rng(0)

    item_factors = rng.random((n_items, factors)).astype(np.float32)
    user_factors = rng.random((n_users, factors)).astype(np.float32)

    # Warmup compute_rmse: 5 ratings, 8 users, 12 items
    _ = compute_rmse(
        np.array([0, 0, 1, 1, 1], dtype=np.int32),
        np.array([0, 1, 0, 1, 2], dtype=np.int32),
        np.array([5.0, 3.0, 4.0, 2.0, 5.0], dtype=np.float32),
        user_factors,
        item_factors,
    )

    # Warmup compute_wmse: 5 ratings, 8 users, 12 items
    _ = compute_wmse(
        np.array([0, 0, 1, 1, 1], dtype=np.int32),
        np.array([0, 1, 0, 1, 2], dtype=np.int32),
        np.array([5.0, 3.0, 4.0, 2.0, 5.0], dtype=np.float32),
        user_factors,
        item_factors,
        alpha=40.0,
    )

    # Warmup compute_ranking_metrics: 2 users, 5 ratings, 3 items
    _ = compute_ranking_metrics(
        np.array([0, 0, 1, 1, 1], dtype=np.int32),
        np.array([0, 1, 0, 1, 2], dtype=np.int32),
        user_factors,
        item_factors,
        k=2,
    )
