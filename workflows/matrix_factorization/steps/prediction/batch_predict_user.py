"""
steps/serving/batch_predict_user.py

ZenML step: predict_user_batch

Generates top-K recommendations for a contiguous slice of users.
Identified by batch_idx — the pipeline fans out n_batches instances of this step.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step

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


@step(enable_cache=False)
def predict_user_batch(
    batch_idx: int,
    model: BaseRecommender,
    model_name: str,
    model_version: str,
    user_batch_size: int,
    batch_top_k: int,
) -> Annotated[pd.DataFrame, CFG_BATCH_USER_PREDICTION_OUTPUT]:
    """
    Generate top-K recommendations for one batch of users.

    The serving_pipeline fans out n_batches instances of this step in parallel
    (id="batch_0", "batch_1", ...). Each determines its user slice from
    batch_idx × user_batch_size.

    Args:
        model: Loaded ALS recommender (passed from load_als_model step).
        batch_idx: Zero-based batch index. Determines which user slice to process.
        user_batch_size: Users per batch. Same value used by serving_pipeline fan-out.
        batch_top_k: Number of recommendations per user.
        model_version: Version string embedded in each output row.

    Returns:
        DataFrame with columns: id, userId, itemId, score, rank, version.
    """
    all_user_ids = np.asarray(model.user_encoder.index.tolist(), dtype=np.int64)
    batch_start = batch_idx * user_batch_size
    batch_ids = all_user_ids[batch_start : batch_start + user_batch_size]

    batch_predictions = model.batch_predict(batch_ids, top_k=batch_top_k)
    return pd.DataFrame.from_records(
        _iter_recommendation_rows(
            batch_predictions.predictions,
            model_name=model.name or model_name,
            model_version=model.version or model_version,
        )
    )
