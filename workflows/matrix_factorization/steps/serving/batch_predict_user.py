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
    CFG_BATCH_PREDICTION_FIELD_NAMES,
    CFG_MODEL_NAME,
    CFG_PREDICTION_FIELD_NAMES,
    CFG_RECS_FIELD_NAMES,
)
from workflows.matrix_factorization.models.als_recommender import ALSRecommender


def _iter_recommendation_rows(
    batch_results: list[dict],
    model_version_name: str,
) -> Iterator[dict]:
    """Yield flat recommendation rows for one user batch."""
    model_id_prefix = CFG_MODEL_NAME.replace("_", "-").lower()
    for result in batch_results:
        uid = result[CFG_BATCH_PREDICTION_FIELD_NAMES.USER_ID.value]
        for rank_pos, rec in enumerate(
            result[CFG_BATCH_PREDICTION_FIELD_NAMES.RECOMMENDATIONS.value]
        ):
            yield {
                CFG_RECS_FIELD_NAMES.RECORD_ID.value: f"{model_id_prefix}-{uid}",
                CFG_RECS_FIELD_NAMES.USER_ID.value: uid,
                CFG_RECS_FIELD_NAMES.REC_ITEM_ID.value: int(rec[CFG_PREDICTION_FIELD_NAMES.ITEM_ID.value]),
                CFG_RECS_FIELD_NAMES.REC_SCORE.value: float(rec[CFG_PREDICTION_FIELD_NAMES.SCORE.value]),
                CFG_RECS_FIELD_NAMES.REC_RANK.value: rank_pos + 1,
                CFG_RECS_FIELD_NAMES.VERSION.value: model_version_name,
            }


@step(enable_cache=False)
def predict_user_batch(
    als_model: ALSRecommender,
    batch_idx: int,
    user_batch_size: int,
    batch_top_k: int,
    model_version_name: str,
) -> Annotated[pd.DataFrame, "batch_recommendations"]:
    """
    Generate top-K recommendations for one batch of users.

    The serving_pipeline fans out n_batches instances of this step in parallel
    (id="batch_0", "batch_1", ...). Each determines its user slice from
    batch_idx × user_batch_size.

    Args:
        als_model: Loaded ALS recommender (passed from load_als_model step).
        batch_idx: Zero-based batch index. Determines which user slice to process.
        user_batch_size: Users per batch. Same value used by serving_pipeline fan-out.
        batch_top_k: Number of recommendations per user.
        model_version_name: Version string embedded in each output row.

    Returns:
        DataFrame with columns: id, userId, itemId, score, rank, version.
    """
    all_user_ids = np.asarray(als_model.user_encoder.index.tolist(), dtype=np.int64)
    batch_start = batch_idx * user_batch_size
    batch_ids = all_user_ids[batch_start : batch_start + user_batch_size]

    batch_results = als_model.batch_predict(batch_ids, top_k=batch_top_k)
    return pd.DataFrame.from_records(_iter_recommendation_rows(batch_results, model_version_name))
