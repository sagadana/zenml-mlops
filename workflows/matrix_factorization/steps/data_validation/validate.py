"""
steps/data_validation/validate.py

ZenML step: validate_data

Runs data quality checks on the raw ratings DataFrame.
Raises DataValidationError if any check fails, halting the pipeline.
"""

from __future__ import annotations

import logging
from typing import Annotated

import dask.dataframe as dd
from zenml import step

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the ratings dataset fails a quality check."""


@step(enable_cache=True)
def validate_data(
    raw_ratings: dd.DataFrame,
    min_sparsity: float = 0.95,
    min_ratings: int = 100_000,
) -> Annotated[dict, "validation_report"]:
    """
    Validate the raw ratings DataFrame.

    Checks:
      1. Required columns exist: userId, movieId, rating, timestamp
      2. No null values in any required column
      3. Rating values in [0.5, 5.0]
      4. No duplicate (userId, movieId) pairs
      5. Dataset size >= min_ratings
      6. Sparsity >= min_sparsity (i.e., the user-item matrix is sufficiently sparse)

    Args:
        raw_ratings: Raw ratings Dask DataFrame from ingest_data.
        min_sparsity: Minimum required sparsity of the user-item matrix (default 0.95).
        min_ratings: Minimum number of ratings required (default 100,000).

    Returns:
        Validation report dict with counts and computed statistics.

    Raises:
        DataValidationError: If any check fails.
    """
    errors: list[str] = []
    report: dict = {}

    # Materialize summary stats (computed once)
    required_cols = {"userId", "movieId", "rating", "timestamp"}
    actual_cols = set(raw_ratings.columns.tolist())

    # Check 1: Required columns
    missing_cols = required_cols - actual_cols
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        raise DataValidationError("\n".join(errors))

    # Compute all stats in one pass where possible
    n_ratings = len(raw_ratings)
    n_users = raw_ratings["userId"].nunique().compute()
    n_items = raw_ratings["movieId"].nunique().compute()
    null_counts = raw_ratings[list(required_cols)].isnull().sum().compute().to_dict()
    rating_min = raw_ratings["rating"].min().compute()
    rating_max = raw_ratings["rating"].max().compute()

    report["n_ratings"] = int(n_ratings)
    report["n_users"] = int(n_users)
    report["n_items"] = int(n_items)
    report["null_counts"] = {k: int(v) for k, v in null_counts.items()}
    report["rating_min"] = float(rating_min)
    report["rating_max"] = float(rating_max)

    # Check 2: No nulls
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        errors.append(f"Found {total_nulls} null values: {null_counts}")

    # Check 3: Rating range
    if rating_min < 0.5 or rating_max > 5.0:
        errors.append(f"Ratings out of [0.5, 5.0] range: min={rating_min}, max={rating_max}")

    # Check 4: Dataset size
    if n_ratings < min_ratings:
        errors.append(f"Too few ratings: {n_ratings} < {min_ratings}")

    # Check 5: Sparsity
    max_possible = n_users * n_items
    sparsity = 1.0 - (n_ratings / max_possible) if max_possible > 0 else 0.0
    report["sparsity"] = float(sparsity)

    if sparsity < min_sparsity:
        errors.append(f"Insufficient sparsity: {sparsity:.4f} < {min_sparsity}")

    # Check 6: Duplicate (userId, movieId) pairs
    n_unique_pairs = (
        raw_ratings[["userId", "movieId"]]
        .drop_duplicates()
        .shape[0]
    )
    n_duplicates = n_ratings - n_unique_pairs
    report["n_duplicate_pairs"] = int(n_duplicates)

    if n_duplicates > 0:
        errors.append(f"Found {n_duplicates} duplicate (userId, movieId) pairs")

    if errors:
        logger.error("Data validation FAILED:\n%s", "\n".join(f"  - {e}" for e in errors))
        raise DataValidationError("\n".join(errors))

    logger.info(
        "Data validation PASSED: %d ratings, %d users, %d items, sparsity=%.4f",
        n_ratings, n_users, n_items, sparsity,
    )
    return report
