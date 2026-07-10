"""
steps/training/als_partition_steps.py

ZenML steps for individual ALS factor-update partitions.

Each step wraps one Numba-JIT kernel call so that every Dask task submitted
during the training loop is a first-class ZenML step.  When called from a
Dask worker (outside a pipeline context), ZenML executes the underlying
function directly with no tracking overhead.  When wired into a pipeline the
steps are individually tracked, cached, and assignable to separate step
operators (e.g. SageMaker Processing jobs).
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from zenml import step


@step(enable_cache=False)
def update_user_partition(
    user_ratings: np.ndarray,
    item_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> Annotated[np.ndarray, "user_factors_block"]:
    """
    Solve the ALS user-factor update for one partition of users.

    Wraps ``solve_user_factors`` (Numba JIT) so the computation can be
    submitted as a ZenML step to a Dask worker.

    Args:
        user_ratings: Dense rating block (n_users_in_partition × n_items) float32.
        item_factors: Current item factor matrix (n_items × rank) float32.
        regularization: L2 regularisation parameter λ.
        alpha: Confidence weighting parameter.

    Returns:
        Updated user factor block (n_users_in_partition × rank) float32.
    """
    from workflows.matrix_factorization.utils.als_numba import solve_user_factors

    return solve_user_factors(user_ratings, item_factors, regularization, alpha)


@step(enable_cache=False)
def update_item_partition(
    item_ratings: np.ndarray,
    user_factors: np.ndarray,
    regularization: float,
    alpha: float,
) -> Annotated[np.ndarray, "item_factors_block"]:
    """
    Solve the ALS item-factor update for one partition of items.

    Wraps ``solve_item_factors`` (Numba JIT) so the computation can be
    submitted as a ZenML step to a Dask worker.

    Args:
        item_ratings: Dense rating block (n_items_in_partition × n_users) float32
            (transposed view used by the item update).
        user_factors: Current user factor matrix (n_users × rank) float32.
        regularization: L2 regularisation parameter λ.
        alpha: Confidence weighting parameter.

    Returns:
        Updated item factor block (n_items_in_partition × rank) float32.
    """
    from workflows.matrix_factorization.utils.als_numba import solve_item_factors

    return solve_item_factors(item_ratings, user_factors, regularization, alpha)
