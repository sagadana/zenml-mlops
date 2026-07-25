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


# ── Factor update (sparse CSR + Cholesky) ───────────────────────────────────


@njit(nogil=True, cache=True)
def _solve_spd_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b for a symmetric positive-definite matrix A via Cholesky.

    Factorises A = L Lᵀ, then solves L y = b (forward) and Lᵀ x = y (backward).

    The ALS normal-equation matrix (YᵀY + λI + Σ (c-1) yyᵀ) is SPD by
    construction, so Cholesky is valid, ~2× cheaper than LU, and needs no
    pivoting. Implemented in scalar loops to stay LAPACK-free and therefore
    safe to call from inside a prange parallel region.
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=A.dtype)
    for j in range(n):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        # Guard against tiny non-positive pivots from float round-off
        if s <= 1e-10:
            s = 1e-10
        L[j, j] = np.sqrt(s)
        inv_ljj = 1.0 / L[j, j]
        for i in range(j + 1, n):
            t = A[i, j]
            for k in range(j):
                t -= L[i, k] * L[j, k]
            L[i, j] = t * inv_ljj

    # Forward substitution: L y = b
    y = np.zeros(n, dtype=A.dtype)
    for i in range(n):
        t = b[i]
        for k in range(i):
            t -= L[i, k] * y[k]
        y[i] = t / L[i, i]

    # Back substitution: Lᵀ x = y
    x = np.zeros(n, dtype=A.dtype)
    for i in range(n - 1, -1, -1):
        t = y[i]
        for k in range(i + 1, n):
            t -= L[k, i] * x[k]
        x[i] = t / L[i, i]

    return x


@njit(parallel=True, nogil=True, cache=True)
def solve_factors_csr(
    indptr: np.ndarray,  # (n_rows + 1,) int64   — CSR row offsets
    indices: np.ndarray,  # (nnz,)       int64   — column (other-entity) id per interaction
    data: np.ndarray,  # (nnz,)          float32 — rating strength per interaction
    other_factors: np.ndarray,  # (n_cols, rank) float32 — fixed factor matrix (Y)
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """
    Solve one ALS half-step for every row entity from a sparse CSR matrix.

    For each row u, solves the confidence-weighted normal equations
        (YᵀY + λI + Σ_i (c_ui - 1) y_i y_iᵀ) x_u = Σ_i c_ui y_i
    with c_ui = 1 + alpha * r_ui and preference p_ui = 1 for observed items.

    Using the Hu–Koren–Volinsky decomposition, YᵀY is computed once and shared
    across all rows; only the rated items of each row contribute the low-rank
    correction, so the per-row cost scales with its number of interactions
    rather than n_cols. Rows are solved in parallel via prange.

    Args:
        indptr: CSR row offsets; row u spans indices[indptr[u]:indptr[u+1]].
        indices: Column (other-entity) index for each interaction.
        data: Rating strength for each interaction.
        other_factors: Fixed factor matrix Y (n_cols × rank).
        regularization: L2 regularization λ.
        alpha: Confidence weighting parameter.

    Returns:
        Updated factor matrix (n_rows × rank) float32.
    """
    n_rows = indptr.shape[0] - 1
    rank = other_factors.shape[1]
    out = np.zeros((n_rows, rank), dtype=np.float32)

    # YᵀY + λI is identical for every row → compute once, reuse via copy.
    base = other_factors.T @ other_factors
    for r in range(rank):
        base[r, r] += regularization

    for u in prange(n_rows):  # parallel over rows
        start = indptr[u]
        end = indptr[u + 1]
        if start == end:  # row with no interactions
            continue

        A = base.copy()
        b = np.zeros(rank, dtype=np.float32)
        for idx in range(start, end):
            col = indices[idx]
            c = 1.0 + alpha * data[idx]
            delta = c - 1.0
            y = other_factors[col]
            # A += delta * y yᵀ   and   b += c * y  (only over rated items)
            for r1 in range(rank):
                yr1 = y[r1]
                b[r1] += c * yr1
                dyr1 = delta * yr1
                for r2 in range(rank):
                    A[r1, r2] += dyr1 * y[r2]

        out[u] = _solve_spd_cholesky(A, b)

    return out


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

    # Warmup solve_factors_csr: 8 rows over a small CSR matrix
    indptr = np.array([0, 2, 3, 3, 5, 6, 7, 8, 9], dtype=np.int64)
    indices = np.array([0, 1, 2, 0, 1, 3, 4, 5, 6], dtype=np.int64)
    data = np.array([5.0, 3.0, 4.0, 2.0, 5.0, 1.0, 4.0, 3.0, 2.0], dtype=np.float32)
    _ = solve_factors_csr(indptr, indices, data, item_factors, 0.1, 40.0)
