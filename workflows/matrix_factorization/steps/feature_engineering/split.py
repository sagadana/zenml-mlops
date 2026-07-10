"""
steps/feature_engineering/split.py

ZenML step: split_data

Stratified train/val/test split by user (each user's ratings split proportionally).
Applies user/item encoders to produce integer-indexed DataFrames.
"""

from __future__ import annotations

import logging
from typing import Annotated

import dask_expr as dd
import numpy as np
import pandas as pd
from zenml import step

from workflows.matrix_factorization.materializers.dask_dataframe_materializer import (
    DaskDataFrameMaterializer,
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
    df = df.sort_values("timestamp")
    n = len(df)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    df = df.copy()
    df["split"] = labels
    return df


@step(
    enable_cache=True,
    output_materializers={
        "train_data": DaskDataFrameMaterializer,
        "val_data": DaskDataFrameMaterializer,
        "test_data": DaskDataFrameMaterializer,
    },
)
def split_data(
    raw_ratings: dd.DataFrame,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_dask_partitions: int = 4,
) -> tuple[
    Annotated[dd.DataFrame, "train_data"],
    Annotated[dd.DataFrame, "val_data"],
    Annotated[dd.DataFrame, "test_data"],
]:
    """
    Split ratings into train/val/test sets with stratification by user.

    Uses chronological ordering within each user's ratings (temporal split)
    to avoid data leakage — training always uses older ratings.

    Args:
        raw_ratings: Raw ratings Dask DataFrame.
        user_encoder: Mapping raw userId → dense int index.
        item_encoder: Mapping raw movieId → dense int index.
        train_ratio: Fraction of each user's ratings for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test (= 1 - train_ratio - val_ratio).

    Returns:
        (train_data, val_data, test_data) — Dask DataFrames with columns:
        user_idx (int32), item_idx (int32), rating (float32), timestamp (int64).
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "train_ratio + val_ratio + test_ratio must sum to 1.0"

    # Compute to pandas for the split operation (groupby + apply)
    df: pd.DataFrame = raw_ratings.compute()

    # Apply encoder maps
    df["user_idx"] = user_encoder[df["userId"]].values.astype("int32")
    df["item_idx"] = item_encoder[df["movieId"]].values.astype("int32")

    rng = np.random.default_rng(42)

    # Stratified split per user
    df = (
        df.groupby("userId", group_keys=False)
        .apply(_split_user_ratings, train_ratio=train_ratio, val_ratio=val_ratio, rng=rng)
        .reset_index(drop=True)
    )

    output_cols = ["user_idx", "item_idx", "rating", "timestamp"]

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

    # Convert back to Dask (partitioned by user_idx range for ALS efficiency)
    train_ddf = dd.from_pandas(train_pd, npartitions=n_dask_partitions)
    val_ddf = dd.from_pandas(val_pd, npartitions=max(1, n_dask_partitions // 4))
    test_ddf = dd.from_pandas(test_pd, npartitions=max(1, n_dask_partitions // 4))

    return train_ddf, val_ddf, test_ddf
