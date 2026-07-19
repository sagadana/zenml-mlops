"""
steps/feature_engineering/encoders.py

ZenML step: build_encoders

Maps raw userId/movieId values to dense integer indices starting from 0.
These encoders are required to build the dense factor matrices used in ALS.
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from zenml import step

from workflows.matrix_factorization.configs import CFG_DATASET_FIELD_NAMES

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def build_encoders(
    raw_ratings: pd.DataFrame,
    power_scaling_alpha: float = 0.5,
) -> tuple[
    Annotated[pd.Series, "user_encoder"],
    Annotated[pd.Series, "item_encoder"],
    Annotated[pd.DataFrame, "scaled_ratings"],
]:
    """
    Build dense integer encoders for users and items and apply power scaling to ratings.

    Args:
        raw_ratings: Raw ratings pandas DataFrame (userId, movieId, rating, timestamp).
        power_scaling_alpha: Exponent for power scaling applied to ratings (default: 0.5).
            scaled_rating = (rating ** power_scaling_alpha - min) / (max - min).

    Returns:
        user_encoder: pd.Series mapping raw userId → dense int index [0, n_users-1].
                      Index = raw userId, values = dense index.
        item_encoder: pd.Series mapping raw movieId → dense int index [0, n_items-1].
                      Index = raw movieId, values = dense index.
        scaled_ratings: Copy of raw_ratings with the rating column power-scaled and min-max normalized to [0, 1].
    """
    # --- Build encoders ---
    # Collect unique IDs and sort them so the mapping is deterministic across runs.
    user_ids = sorted(raw_ratings[CFG_DATASET_FIELD_NAMES.USER_ID.value].unique().tolist())
    item_ids = sorted(raw_ratings[CFG_DATASET_FIELD_NAMES.ITEM_ID.value].unique().tolist())

    # Map each raw ID to a contiguous integer index starting at 0.
    # These dense indices are used to index into the ALS factor matrices directly.
    user_encoder = pd.Series(
        data=range(len(user_ids)),
        index=user_ids,
        name="user_dense_idx",
        dtype="int32",
    )
    item_encoder = pd.Series(
        data=range(len(item_ids)),
        index=item_ids,
        name="item_dense_idx",
        dtype="int32",
    )

    # --- Scale ratings ---
    rating_field = CFG_DATASET_FIELD_NAMES.RATING.value
    scaled_ratings = raw_ratings.copy()

    # Step 1 — Power scaling: compress the rating range and reduce the influence
    # of high ratings relative to low ones (similar to a square-root transform
    # when alpha=0.5).
    scaled_ratings[rating_field] = scaled_ratings[rating_field] ** power_scaling_alpha

    # Step 2 — Min-max normalization: shift the power-scaled values to [0, 1]
    # so the ALS confidence weights are on a consistent scale regardless of
    # the original rating magnitude.
    power_min = scaled_ratings[rating_field].min()
    power_max = scaled_ratings[rating_field].max()
    scaled_ratings[rating_field] = (scaled_ratings[rating_field] - power_min) / (
        power_max - power_min
    )

    logger.info(
        "Encoders built: %d users, %d items. Power scaling (alpha=%.3f) + min-max normalization applied: "
        "ratings range [%.4f, %.4f] → [%.4f, %.4f] → [%.4f, %.4f]",
        len(user_encoder),
        len(item_encoder),
        power_scaling_alpha,
        raw_ratings[rating_field].min(),
        raw_ratings[rating_field].max(),
        power_min,
        power_max,
        scaled_ratings[rating_field].min(),
        scaled_ratings[rating_field].max(),
    )
    return user_encoder, item_encoder, scaled_ratings
