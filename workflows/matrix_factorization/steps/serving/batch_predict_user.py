"""
steps/serving/batch_predict_step.py

ZenML step for a single user-batch recommendation inference.

Extracted from ``generate_batch_recommendations`` so every Dask task
submitted during the batch loop is a first-class ZenML step.  When called
from a Dask worker (outside a pipeline context), ZenML executes the
underlying function directly.  When wired into a pipeline the step is
individually tracked and assignable to a separate step operator.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step

from workflows.matrix_factorization.configs import CFG_MODEL_NAME
from workflows.matrix_factorization.models.als_recommender import ALSRecommender


def _iter_recommendation_rows(
    batch_results: list[dict],
    model_version_name: str,
) -> Iterator[dict]:
    """Yield flat recommendation rows for one user batch."""
    model_id_prefix = CFG_MODEL_NAME.replace("_", "-").lower()
    for result in batch_results:
        uid = int(result["user_id"])
        for rank_pos, rec in enumerate(result["recommendations"]):
            yield {
                "id": f"{model_id_prefix}-{uid}",
                "userId": uid,
                "itemId": int(rec["item_id"]),
                "score": float(rec["score"]),
                "rank": rank_pos + 1,
                "version": model_version_name,
            }


@step(enable_cache=False)
def predict_user_batch(
    als_model: ALSRecommender,
    user_ids: np.ndarray,
    batch_top_k: int,
    model_version_name: str,
) -> Annotated[pd.DataFrame, "batch_recommendations"]:
    """
    Generate top-K recommendation rows for a single batch of users.

    Wraps ``ALSRecommender.batch_predict`` so the computation can be
    submitted as a ZenML step to a Dask worker.

    Args:
        als_model: Loaded ALS recommender model.
        user_ids: 1-D array of raw user IDs to score.
        batch_top_k: Number of recommendations per user.
        model_version_name: Model version string stored in each output row.

    Returns:
        DataFrame with columns: id, userId, itemId, score, rank, version.
    """
    batch_results = als_model.batch_predict(user_ids, top_k=batch_top_k)
    return pd.DataFrame.from_records(
        _iter_recommendation_rows(batch_results, model_version_name)
    )
