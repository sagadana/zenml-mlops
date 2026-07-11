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
) -> tuple[
    Annotated[pd.Series, "user_encoder"],
    Annotated[pd.Series, "item_encoder"],
]:
    """
    Build dense integer encoders for users and items.

    Args:
        raw_ratings: Raw ratings pandas DataFrame (userId, movieId, rating, timestamp).

    Returns:
        user_encoder: pd.Series mapping raw userId → dense int index [0, n_users-1].
                      Index = raw userId, values = dense index.
        item_encoder: pd.Series mapping raw movieId → dense int index [0, n_items-1].
                      Index = raw movieId, values = dense index.
    """
    # Compute unique sorted user/item IDs (sorted for deterministic mapping)
    user_ids = sorted(raw_ratings[CFG_DATASET_FIELD_NAMES.USER_ID.value].unique().tolist())
    item_ids = sorted(raw_ratings[CFG_DATASET_FIELD_NAMES.ITEM_ID.value].unique().tolist())

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

    logger.info(
        "Encoders built: %d users, %d items",
        len(user_encoder),
        len(item_encoder),
    )
    return user_encoder, item_encoder
