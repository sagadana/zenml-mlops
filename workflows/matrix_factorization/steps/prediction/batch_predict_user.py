"""
steps/serving/batch_predict_user.py

ZenML steps: get_user_ids, get_user_batch_slice, predict_user_batch

Generates top-K recommendations for a contiguous slice of users.
Fan-out pattern: get_user_ids computes the full id list and batch size;
get_user_batch_slice slices per batch; predict_user_batch predicts per slice.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step
from zenml.enums import StepRuntime

from workflows.matrix_factorization.configs import (
    CFG_BATCH_USER_PREDICTION_OUTPUT,
    CFG_RECS_FIELD_NAMES,
)
from workflows.matrix_factorization.models.base_recommender import BaseRecommender, PredictionItem


def _iter_recommendation_rows(
    batch_predictions: dict[str, list[PredictionItem]],
    model_name: str,
    model_version: str,
) -> Iterator[dict]:
    """Yield flat recommendation rows for one user batch."""
    model_id_prefix = model_name.replace("_", "-").lower()
    for user_id_str, recommendations in batch_predictions.items():
        uid = int(user_id_str)
        for rank_pos, rec in enumerate(recommendations):
            yield {
                CFG_RECS_FIELD_NAMES.RECORD_ID.value: f"{model_id_prefix}-{uid}",
                CFG_RECS_FIELD_NAMES.USER_ID.value: uid,
                CFG_RECS_FIELD_NAMES.REC_ITEM_ID.value: rec.item_id,
                CFG_RECS_FIELD_NAMES.REC_SCORE.value: rec.score,
                CFG_RECS_FIELD_NAMES.REC_RANK.value: rank_pos + 1,
                CFG_RECS_FIELD_NAMES.VERSION.value: model_version,
            }


@step(enable_cache=True, runtime=StepRuntime.INLINE)
def get_user_ids(
    model: BaseRecommender,
    n_batches: int,
    min_user_batch_size: int,
) -> tuple[
    Annotated[np.ndarray, "user_ids"],
    Annotated[int, "batch_size"],
]:
    """
    Get all user IDs from the model and compute the effective batch size.

    Args:
        model: Loaded ALS recommender (passed from load_als_model step).
        n_batches: Number of fan-out batches.
        min_user_batch_size: Minimum users per batch; actual batch size is
            max(min_user_batch_size, ceil(total_users / n_batches)).

    Returns:
        user_ids: Numpy array of all user IDs.
        batch_size: Effective number of users per batch.
    """
    user_ids = np.asarray(model.user_encoder.index.tolist(), dtype=np.int64)
    batch_size = max(min_user_batch_size, math.ceil(len(user_ids) / n_batches))
    return user_ids, batch_size


@step(enable_cache=True, runtime=StepRuntime.INLINE)
def get_user_batch_slice(
    user_ids: np.ndarray,
    batch_size: int,
    batch_idx: int,
) -> Annotated[np.ndarray, "batch_ids"]:
    """
    Get a contiguous slice of user IDs for the given batch index.

    Args:
        user_ids: Numpy array of all user IDs (from get_user_ids).
        batch_size: Effective users per batch (from get_user_ids).
        batch_idx: Zero-based batch index.

    Returns:
        Numpy array of user IDs for the specified batch.
    """
    batch_start = batch_idx * batch_size
    return user_ids[batch_start : batch_start + batch_size]


@step(enable_cache=True, runtime=StepRuntime.ISOLATED)
def predict_user_batch(
    batch_ids: np.ndarray,
    model: BaseRecommender,
    model_name: str,
    model_version: str,
    batch_top_k: int,
) -> Annotated[pd.DataFrame, CFG_BATCH_USER_PREDICTION_OUTPUT]:
    """
    Generate top-K recommendations for one batch of users.

    Args:
        batch_ids: Numpy array of user IDs for this batch.
        model: Loaded ALS recommender (passed from load_als_model step).
        model_name: Name of the ALS model (passed from load_als_model step).
        model_version: Version of the ALS model (passed from load_als_model step).
        batch_top_k: Number of top recommendations to generate per user.

    Returns:
        DataFrame with columns: id, userId, itemId, score, rank, version.
    """
    batch_predictions = model.batch_predict(batch_ids, top_k=batch_top_k)
    return pd.DataFrame.from_records(
        _iter_recommendation_rows(
            batch_predictions.predictions,
            model_name=model.name or model_name,
            model_version=model.version or model_version,
        )
    )
