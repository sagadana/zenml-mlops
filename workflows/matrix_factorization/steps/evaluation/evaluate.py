"""
steps/model_evaluation/evaluate.py

ZenML step: compute_metrics

Distributed evaluation of the trained ALS model on the held-out test set.
Computes RMSE, MAE, Precision@K, Recall@K, NDCG@K.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import log_metadata, step

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.base_recommender import BaseRecommender
from workflows.matrix_factorization.utils.als_numba import warmup_jit

logger = logging.getLogger(__name__)

warmup_jit()  # Warm up the Numba JIT compiler for compute_rmse_block


@step(enable_cache=True)
def compute_metrics(
    test_data: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    best_hyperparams: dict,
    top_k: int = 10,
    sample_seed: int = 42,
    sample_size: int = 50_000,  # defult: sample up to 50k users for efficiency
) -> Annotated[dict, "eval_metrics"]:
    """
    Evaluate the trained ALS model on the test set.

    Args:
        test_data: Test split pandas DataFrame.
        user_factors: Trained user factor matrix (n_users × rank).
        item_factors: Trained item factor matrix (n_items × rank).
        best_hyperparams: Hyperparams dict.
        top_k: K for ranking metrics.
        sample_seed: Random seed for sampling users for ranking metrics.
        sample_size: Max number of users to sample for ranking metrics (for efficiency).

    Returns:
        eval_metrics dict with RMSE, MAE, Precision@K, Recall@K, NDCG@K.
    """

    test_pd = test_data
    n_users = user_factors.shape[0]
    n_items = item_factors.shape[0]

    # Sample users for efficiency if the test set is large
    sampled = test_pd.copy()
    unique_users = sampled[CFG_FEATURES_FIELD_NAMES.USER_ID.value].unique()
    if len(unique_users) > sample_size:
        sampled_users = np.random.default_rng(sample_seed).choice(
            unique_users, sample_size, replace=False
        )
        sampled = sampled[sampled[CFG_FEATURES_FIELD_NAMES.USER_ID.value].isin(sampled_users)]

    # Compute RMSE and Ranking metrics on the sampled test set
    sorted_df = sampled.sort_values(CFG_FEATURES_FIELD_NAMES.USER_ID.value)
    user_ids = np.clip(
        np.asarray(sorted_df[CFG_FEATURES_FIELD_NAMES.USER_ID.value].values, dtype=np.int32),
        0,
        n_users - 1,
    )
    item_ids = np.clip(
        np.asarray(sorted_df[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].values, dtype=np.int32),
        0,
        n_items - 1,
    )
    ratings = np.asarray(sorted_df[CFG_FEATURES_FIELD_NAMES.RATING.value].values, dtype=np.float32)

    rmse, precision, recall, ndcg = BaseRecommender.compute_scores(
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        user_factors=user_factors,
        item_factors=item_factors,
        k=top_k,
    )

    metrics = {
        "top_k": top_k,
        "rmse": rmse,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "ndcg_at_k": ndcg,
        "n_test_ratings": len(ratings),
        "n_test_users": len(np.unique(user_ids)),
        "n_test_items": len(np.unique(item_ids)),
    }

    log_metadata(
        metadata={"metrics": metrics, "best_hyperparams": best_hyperparams},
        infer_model=True,
    )

    logger.info(
        "Evaluation: RMSE=%.4f P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f",
        rmse,
        top_k,
        precision,
        top_k,
        recall,
        top_k,
        ndcg,
    )

    return metrics
