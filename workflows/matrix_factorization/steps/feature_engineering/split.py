"""
steps/feature_engineering/split.py

ZenML step: split_data

Stratified train/val/test split by user (each user's ratings split proportionally).
Applies user/item encoders to produce integer-indexed DataFrames.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step

from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_FEATURES_FIELD_NAMES,
)

logger = logging.getLogger(__name__)


def _split_user_ratings(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Assign a split label to each rating for a single user.
    Splits are done chronologically (by timestamp) to avoid data leakage.
    """
    df = df.sort_values(CFG_DATASET_FIELD_NAMES.TIMESTAMP.value)
    n = len(df)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    df = df.copy()
    df["split"] = labels
    return df


@step(enable_cache=True)
def split_data(
    raw_ratings: pd.DataFrame,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[
    Annotated[pd.DataFrame, "train_data"],
    Annotated[pd.DataFrame, "val_data"],
    Annotated[pd.DataFrame, "test_data"],
]:
    """
    Split ratings into train/val/test sets with stratification by user.

    Uses chronological ordering within each user's ratings (temporal split)
    to avoid data leakage — training always uses older ratings.

    Args:
        raw_ratings: Raw ratings pandas DataFrame.
        user_encoder: Mapping raw userId → dense int index.
        item_encoder: Mapping raw movieId → dense int index.
        train_ratio: Fraction of each user's ratings for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test (= 1 - train_ratio - val_ratio).

    Returns:
        (train_data, val_data, test_data) — pandas DataFrames with columns:
        user_idx (int32), item_idx (int32), rating (float32), timestamp (int64).
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "train_ratio + val_ratio + test_ratio must sum to 1.0"

    df = raw_ratings

    # Apply encoder maps
    df[CFG_FEATURES_FIELD_NAMES.USER_ID.value] = user_encoder[
        df[CFG_DATASET_FIELD_NAMES.USER_ID.value]
    ].values.astype("int32")
    df[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value] = item_encoder[
        df[CFG_DATASET_FIELD_NAMES.ITEM_ID.value]
    ].values.astype("int32")

    rng = np.random.default_rng(42)

    # Stratified split per user
    df = (
        df.groupby(CFG_DATASET_FIELD_NAMES.USER_ID.value, group_keys=False)
        .apply(_split_user_ratings, train_ratio=train_ratio, val_ratio=val_ratio, rng=rng)
        .reset_index(drop=True)
    )

    output_cols = [
        CFG_FEATURES_FIELD_NAMES.USER_ID.value,
        CFG_FEATURES_FIELD_NAMES.ITEM_ID.value,
        CFG_FEATURES_FIELD_NAMES.RATING.value,
        CFG_FEATURES_FIELD_NAMES.TIMESTAMP.value,
    ]

    train_pd = df[df["split"] == "train"][output_cols].reset_index(drop=True)
    val_pd = df[df["split"] == "val"][output_cols].reset_index(drop=True)
    test_pd = df[df["split"] == "test"][output_cols].reset_index(drop=True)

    logger.info(
        "Split complete: train=%d, val=%d, test=%d (total=%d)",
        len(train_pd),
        len(val_pd),
        len(test_pd),
        len(df),
    )

    return train_pd, val_pd, test_pd
