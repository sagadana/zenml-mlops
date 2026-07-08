"""
utils/als_numba.py

Numba-JIT-compiled ALS (Alternating Least Squares) solver kernels.

These functions are the performance-critical inner loop of ALS training.
Each Dask worker calls these on a numpy block (materialized partition).

Design:
  - @njit(parallel=True, nogil=True, cache=True) on the per-user/item solve
  - prange over users/items within a block — one level of parallelism only
  - numpy arrays in, numpy arrays out — Dask workers call these as regular callables
  - cache=True: compiled bytecode persists in __pycache__; avoids per-invocation JIT cost

Correctness reference:
  Implicit feedback ALS from Hu, Koren & Volinsky (2008).
  For each user i, the update is:
      u_i = (Y^T C^i Y + λI)^{-1} Y^T C^i p_i
  where:
      Y         = item factor matrix (n_items × rank)
      C^i       = diagonal confidence matrix for user i
      p_i       = preference vector (1 if rating > 0, else 0)
      λ         = regularization parameter
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ── User factor update ──────────────────────────────────────────────────────

@njit(parallel=True, nogil=True, cache=True)
def solve_user_factors(
    user_ratings: np.ndarray,   # (n_users_in_block, n_items)  float32
    item_factors: np.ndarray,   # (n_items, rank)               float32
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
    n_users, n_items = user_ratings.shape
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

        Y_u = item_factors[rated_indices]          # (n_rated × rank)
        r_u = user_ratings[u, rated_indices]       # (n_rated,)
        c_u = 1.0 + alpha * r_u                    # confidence weights

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

        user_factors[u] = np.linalg.solve(A_u, b_u)

    return user_factors


# ── Item factor update ──────────────────────────────────────────────────────

@njit(parallel=True, nogil=True, cache=True)
def solve_item_factors(
    item_ratings: np.ndarray,   # (n_items_in_block, n_users)  float32  (transposed view)
    user_factors: np.ndarray,   # (n_users, rank)               float32
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


# ── RMSE computation ────────────────────────────────────────────────────────

@njit(parallel=True, nogil=True, cache=True)
def compute_rmse_block(
    user_indices: np.ndarray,   # (n_ratings,)  int32
    item_indices: np.ndarray,   # (n_ratings,)  int32
    ratings: np.ndarray,        # (n_ratings,)  float32
    user_factors: np.ndarray,   # (n_users, rank)
    item_factors: np.ndarray,   # (n_items, rank)
) -> tuple[float, int]:
    """
    Compute sum of squared errors and count for a block of ratings.
    Returns (sse, count) — caller aggregates across blocks for RMSE.
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


# ── JIT warmup ──────────────────────────────────────────────────────────────

def warmup_jit(rank: int = 10) -> None:
    """
    Pre-compile all JIT functions with small dummy data.
    Call this at Docker image build time to eliminate cold-start latency.
    """
    n_users, n_items = 8, 12
    rng = np.random.default_rng(0)

    user_ratings = rng.random((n_users, n_items)).astype(np.float32) * 5
    # Sparsify: 90% zeros
    mask = rng.random((n_users, n_items)) > 0.1
    user_ratings[mask] = 0.0

    item_factors = rng.random((n_items, rank)).astype(np.float32)
    user_factors = solve_user_factors(user_ratings, item_factors, 0.01, 1.0)

    item_ratings = user_ratings.T.copy()
    _ = solve_item_factors(item_ratings, user_factors, 0.01, 1.0)

    u_idx = np.array([0, 1, 2], dtype=np.int32)
    i_idx = np.array([0, 1, 2], dtype=np.int32)
    r = np.array([4.0, 3.0, 5.0], dtype=np.float32)
    _ = compute_rmse_block(u_idx, i_idx, r, user_factors, item_factors)
