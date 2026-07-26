"""
steps/feature_engineering/split.py

ZenML step: split_data

Stratified train/val split by user (each user's ratings split proportionally).
Expects pre-encoded features DataFrame (output of prepare_features).
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


@step(enable_cache=True)
def prepare_features(
    raw_ratings: pd.DataFrame,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
) -> Annotated[pd.DataFrame, "features"]:
    """
    Apply user/item encoders to the full ratings dataset.

    Args:
        raw_ratings: Raw ratings pandas DataFrame.
        user_encoder: Mapping raw userId → dense int index.
        item_encoder: Mapping raw movieId → dense int index.

    Returns:
        features — pandas DataFrame with columns:
        user_idx (int32), item_idx (int32), rating (float32), timestamp (int64).
    """
    df = raw_ratings
    df[CFG_FEATURES_FIELD_NAMES.USER_ID.value] = user_encoder[
        df[CFG_DATASET_FIELD_NAMES.USER_ID.value]
    ].values.astype("int32")
    df[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value] = item_encoder[
        df[CFG_DATASET_FIELD_NAMES.ITEM_ID.value]
    ].values.astype("int32")

    logger.info("Features: %d ratings", len(df))
    output_cols = df.columns.tolist()
    return df[output_cols].reset_index(drop=True)


@step(enable_cache=True)
def split_data(
    features: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> tuple[
    Annotated[pd.DataFrame, "train_data"],
    Annotated[pd.DataFrame, "val_data"],
]:
    """
    Split encoded features into train/val sets with per-user temporal stratification.

    Uses chronological ordering within each user's ratings so that training always
    uses older ratings and validation always uses newer ones. This intentionally
    introduces item- and rating-distribution drift between the splits, which is the
    correct behaviour for HPO: it simulates the train-on-history / evaluate-on-future
    scenario that the deployed model faces, producing more realistic hyperparameter
    scores than a random split would. Every user with at least one rating is
    guaranteed to appear in the training split (``clip(lower=1)``).

    Expects the pre-encoded features DataFrame produced by ``prepare_features``.

    Args:
        features: Encoded features DataFrame (output of prepare_features).
        train_ratio: Fraction of each user's ratings for training.
        val_ratio: Fraction for validation (= 1 - train_ratio).

    Returns:
        (train_data, val_data) — pandas DataFrames with columns:
        user_idx (int32), item_idx (int32), rating (float32), timestamp (int64).
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must sum to 1.0"

    user_col = CFG_DATASET_FIELD_NAMES.USER_ID.value
    ts_col = CFG_DATASET_FIELD_NAMES.TIMESTAMP.value

    # Sort globally by user then timestamp (chronological per user, no per-group apply)
    df = features.sort_values([user_col, ts_col]).reset_index(drop=True)

    # Per-user 0-based rank and total count — both O(n), no Python loops
    rank = df.groupby(user_col).cumcount()
    count = df.groupby(user_col)[user_col].transform("count")
    n_train = (count * train_ratio).astype(int).clip(lower=1)

    df = df.copy()
    df["split"] = np.where(rank < n_train, "train", "val")

    output_cols = df.columns.tolist()

    train_pd = df[df["split"] == "train"][output_cols].reset_index(drop=True)
    val_pd = df[df["split"] == "val"][output_cols].reset_index(drop=True)

    logger.info(
        "Split complete: train=%d, val=%d (total=%d)",
        len(train_pd),
        len(val_pd),
        len(df),
    )

    return train_pd, val_pd
