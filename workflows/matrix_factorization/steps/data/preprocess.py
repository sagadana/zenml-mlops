"""
steps/data/preprocess.py

ZenML step: preprocess_data

Applies common MovieLens preprocessing to the raw ratings DataFrame:
  1. Drop duplicate (userId, movieId) pairs — keep the highest-rated interaction.
  2. Remove users with fewer than `min_user_ratings` interactions.
  3. Remove items with fewer than `min_item_ratings` interactions.
  4. Keep only the top `top_ratings_per_user` ratings per user (by rating
     descending, then by timestamp descending as a tie-breaker).
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from zenml import step

from workflows.matrix_factorization.configs import CFG_DATASET_FIELD_NAMES

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def preprocess_data(
    raw_ratings: pd.DataFrame,
    min_user_ratings: int = 5,
    min_item_ratings: int = 1,
    top_ratings_per_user: int = 10,
) -> Annotated[pd.DataFrame, "processed_ratings"]:
    """
    Apply standard MovieLens preprocessing to the raw ratings DataFrame.

    Steps applied in order:
      1. Deduplication: for duplicate (userId, movieId) pairs keep the entry
         with the highest rating (latest timestamp as tie-breaker).
      2. User activity filter: drop users with fewer than `min_user_ratings`
         total ratings (cold-start pruning).
      3. Item popularity filter: drop items with fewer than `min_item_ratings`
         total ratings (long-tail pruning).
      4. Top-N selection: for each user, keep only the `top_ratings_per_user`
         highest-rated interactions (latest timestamp as tie-breaker), so the
         ALS model focuses on the most relevant signal per user.

    Args:
        raw_ratings: Raw ratings DataFrame (userId, movieId, rating, timestamp).
        min_user_ratings: Minimum number of ratings a user must have to be kept
            (default: 5).
        min_item_ratings: Minimum number of ratings an item must have to be kept
            (default: 1).
        top_ratings_per_user: Maximum number of ratings to retain per user,
            selected by highest rating then most recent timestamp (default: 10).

    Returns:
        processed_ratings: Preprocessed DataFrame with the same columns as the
            input, sorted by userId and timestamp, with a reset integer index.
    """
    user_col = CFG_DATASET_FIELD_NAMES.USER_ID.value
    item_col = CFG_DATASET_FIELD_NAMES.ITEM_ID.value
    rating_col = CFG_DATASET_FIELD_NAMES.RATING.value
    ts_col = CFG_DATASET_FIELD_NAMES.TIMESTAMP.value

    n_before = len(raw_ratings)

    # Step 1 — Deduplication: keep the highest rating for each (user, item) pair.
    df = raw_ratings.sort_values([rating_col, ts_col], ascending=[False, False]).drop_duplicates(
        subset=[user_col, item_col], keep="first"
    )
    n_after_dedup = len(df)
    logger.info(
        "Deduplication: %d → %d rows (removed %d duplicates)",
        n_before,
        n_after_dedup,
        n_before - n_after_dedup,
    )

    # Step 2 — User activity filter: remove cold-start users.
    user_counts = df[user_col].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df[user_col].isin(active_users)]
    n_after_user_filter = len(df)
    logger.info(
        "User filter (min=%d): %d → %d rows, %d → %d users",
        min_user_ratings,
        n_after_dedup,
        n_after_user_filter,
        len(user_counts),
        len(active_users),
    )

    # Step 3 — Item popularity filter: remove long-tail items.
    item_counts = df[item_col].value_counts()
    popular_items = item_counts[item_counts >= min_item_ratings].index
    df = df[df[item_col].isin(popular_items)]
    n_after_item_filter = len(df)
    logger.info(
        "Item filter (min=%d): %d → %d rows, %d → %d items",
        min_item_ratings,
        n_after_user_filter,
        n_after_item_filter,
        len(item_counts),
        len(popular_items),
    )

    # Step 4 — Top-N per user: retain only the highest-rated (most recent as
    # tie-breaker) interactions for each user.
    df = (
        df.sort_values([user_col, rating_col, ts_col], ascending=[True, False, False])
        .groupby(user_col, sort=False)
        .head(top_ratings_per_user)
    )
    n_after_topn = len(df)
    logger.info(
        "Top-%d per user: %d → %d rows",
        top_ratings_per_user,
        n_after_item_filter,
        n_after_topn,
    )

    df = df.sort_values([user_col, ts_col]).reset_index(drop=True)

    logger.info(
        "Preprocessing complete: %d → %d rows (%.1f%% retained), %d users, %d items",
        n_before,
        n_after_topn,
        100.0 * n_after_topn / n_before if n_before > 0 else 0.0,
        df[user_col].nunique(),
        df[item_col].nunique(),
    )

    return df
