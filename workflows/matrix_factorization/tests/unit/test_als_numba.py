"""
tests/unit/test_als_numba.py

Unit tests for Numba-JIT ALS solver kernels.
Validates correctness by comparing Numba output against a naive numpy implementation.
"""

import numpy as np
import pytest

from workflows.matrix_factorization.utils.als_numba import (
    compute_rmse_block,
    solve_item_factors,
    solve_user_factors,
    warmup_jit,
)


def _naive_solve_user_factors(
    user_ratings: np.ndarray,
    item_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> np.ndarray:
    """Reference implementation using pure numpy (no Numba)."""
    n_users, n_items = user_ratings.shape
    rank = item_factors.shape[1]
    user_factors = np.zeros((n_users, rank), dtype=np.float32)

    for u in range(n_users):
        rated = np.where(user_ratings[u] > 0)[0]
        if len(rated) == 0:
            continue
        Y_u = item_factors[rated]
        r_u = user_ratings[u, rated]
        c_u = 1.0 + alpha * r_u
        A_u = Y_u.T @ np.diag(c_u) @ Y_u + regularization * np.eye(rank)
        b_u = Y_u.T @ (c_u * np.ones_like(r_u))  # p_u = 1 for implicit
        user_factors[u] = np.linalg.solve(A_u.astype(np.float64), b_u.astype(np.float64)).astype(
            np.float32
        )
    return user_factors


@pytest.fixture(scope="module", autouse=True)
def warmup():
    """Warm up JIT compilation before all tests in this module."""
    warmup_jit(rank=10)


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(42)
    n_users, n_items, rank = 20, 30, 10
    R = rng.random((n_users, n_items)).astype(np.float32)
    # Sparsify: ~80% zeros
    R[rng.random((n_users, n_items)) > 0.2] = 0.0
    item_factors = rng.standard_normal((n_items, rank)).astype(np.float32) * 0.1
    return R, item_factors, rank


def test_solve_user_factors_shape(small_problem):
    R, item_factors, rank = small_problem
    n_users = R.shape[0]
    result = solve_user_factors(R, item_factors, regularization=0.01, alpha=1.0)
    assert result.shape == (n_users, rank), f"Expected ({n_users}, {rank}), got {result.shape}"
    assert result.dtype == np.float32


def test_solve_user_factors_correctness(small_problem):
    """Numba output should match naive numpy within float32 tolerance."""
    R, item_factors, rank = small_problem
    numba_result = solve_user_factors(R, item_factors, regularization=0.01, alpha=1.0)
    naive_result = _naive_solve_user_factors(R, item_factors, regularization=0.01, alpha=1.0)
    np.testing.assert_allclose(numba_result, naive_result, rtol=1e-3, atol=1e-4)


def test_solve_user_factors_all_zeros():
    """Users with no ratings should get zero factor vectors."""
    R = np.zeros((5, 10), dtype=np.float32)
    item_factors = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    result = solve_user_factors(R, item_factors, regularization=0.01, alpha=1.0)
    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_solve_item_factors_shape(small_problem):
    R, item_factors, rank = small_problem
    n_items = R.shape[1]
    # For item update: R.T is (n_items × n_users), user_factors = solve_user_factors first
    user_factors = solve_user_factors(R, item_factors, 0.01, 1.0)
    result = solve_item_factors(R.T.copy(), user_factors, regularization=0.01, alpha=1.0)
    assert result.shape == (n_items, rank)
    assert result.dtype == np.float32


def test_compute_rmse_block():
    rng = np.random.default_rng(0)
    n, rank = 100, 10
    user_factors = rng.standard_normal((50, rank)).astype(np.float32)
    item_factors = rng.standard_normal((60, rank)).astype(np.float32)
    u_idx = rng.integers(0, 50, n).astype(np.int32)
    i_idx = rng.integers(0, 60, n).astype(np.int32)
    ratings = rng.uniform(1, 5, n).astype(np.float32)

    sse, count = compute_rmse_block(u_idx, i_idx, ratings, user_factors, item_factors)
    assert count == n
    assert sse >= 0.0

    # Verify against numpy
    preds = np.array(
        [float(user_factors[u_idx[i]] @ item_factors[i_idx[i]]) for i in range(n)], dtype=np.float32
    )
    expected_sse = float(np.sum((preds - ratings) ** 2))
    np.testing.assert_allclose(sse, expected_sse, rtol=1e-3)


def test_warmup_jit_runs():
    """warmup_jit should complete without error."""
    warmup_jit(rank=8)
