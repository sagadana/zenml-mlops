"""
steps/model_evaluation/evaluate.py

ZenML step: compute_metrics

Distributed evaluation of the trained ALS model on the held-out test set.
Computes RMSE, MAE, Precision@K, Recall@K, NDCG@K.
Logs all metrics to MLflow.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import log_metadata, step

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES

logger = logging.getLogger(__name__)


def _compute_precision_recall_ndcg(
    test_pd: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    k: int = 10,
    rating_threshold: float = 3.5,
) -> tuple[float, float, float]:
    """
    Compute Precision@K, Recall@K, NDCG@K averaged over users.

    A rating >= rating_threshold is considered a "relevant" item.
    """
    precisions, recalls, ndcgs = [], [], []

    for user_idx, group in test_pd.groupby(CFG_FEATURES_FIELD_NAMES.USER_ID.value):
        if int(user_idx) >= user_factors.shape[0]:  # type: ignore[arg-type]
            continue

        u = user_factors[int(user_idx)]  # type: ignore[arg-type]
        scores = item_factors @ u  # (n_items,)

        relevant_items = set(
            group[group[CFG_FEATURES_FIELD_NAMES.RATING.value] >= rating_threshold][
                CFG_FEATURES_FIELD_NAMES.ITEM_ID.value
            ].tolist()
        )
        if not relevant_items:
            continue

        # Top-K predicted items
        top_k_idxs = np.argpartition(scores, -k)[-k:]
        top_k_idxs = top_k_idxs[np.argsort(scores[top_k_idxs])[::-1]]

        hits = [1 if idx in relevant_items else 0 for idx in top_k_idxs]
        n_relevant = len(relevant_items)

        # Precision@K
        precisions.append(sum(hits) / k)

        # Recall@K
        recalls.append(sum(hits) / n_relevant)

        # NDCG@K
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = min(k, n_relevant)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    p_at_k = float(np.mean(precisions)) if precisions else 0.0
    r_at_k = float(np.mean(recalls)) if recalls else 0.0
    ndcg_at_k = float(np.mean(ndcgs)) if ndcgs else 0.0
    return p_at_k, r_at_k, ndcg_at_k


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
        best_hyperparams: Hyperparams dict (logged to MLflow as parameters).
        top_k: K for ranking metrics.

    Returns:
        eval_metrics dict with RMSE, MAE, Precision@K, Recall@K, NDCG@K.
    """
    from workflows.matrix_factorization.utils.als_numba import compute_rmse_block

    test_pd = test_data
    n_users = user_factors.shape[0]
    n_items = item_factors.shape[0]

    # Clip indices to factor matrix bounds
    u_idx = np.clip(
        test_pd[CFG_FEATURES_FIELD_NAMES.USER_ID.value].values.astype(np.int32),
        0,
        n_users - 1,
    )
    i_idx = np.clip(
        test_pd[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].values.astype(np.int32),
        0,
        n_items - 1,
    )
    r = test_pd[CFG_FEATURES_FIELD_NAMES.RATING.value].values.astype(np.float32)

    # RMSE and MAE
    sse, count = compute_rmse_block(u_idx, i_idx, r, user_factors, item_factors)
    rmse = float(np.sqrt(sse / count)) if count > 0 else float("inf")

    # MAE (compute manually)
    preds = np.einsum("ij,ij->i", user_factors[u_idx], item_factors[i_idx])
    mae = float(np.abs(preds - r).mean())

    # Ranking metrics
    sampled = test_pd.copy()
    unique_users = sampled[CFG_FEATURES_FIELD_NAMES.USER_ID.value].unique()
    if len(unique_users) > sample_size:
        sampled_users = np.random.default_rng(sample_seed).choice(
            unique_users, sample_size, replace=False
        )
        sampled = sampled[sampled[CFG_FEATURES_FIELD_NAMES.USER_ID.value].isin(sampled_users)]

    precision, recall, ndcg = _compute_precision_recall_ndcg(
        sampled, user_factors, item_factors, k=top_k
    )

    metrics = {
        "rmse": rmse,
        "mae": mae,
        f"precision_at_{top_k}": precision,
        f"recall_at_{top_k}": recall,
        f"ndcg_at_{top_k}": ndcg,
        "n_test_ratings": int(count),
        "rank": int(best_hyperparams.get("rank", 0)),
    }

    log_metadata(
        metadata={"metrics": metrics, "best_hyperparams": best_hyperparams},
        infer_model=True,
    )

    # Log to MLflow if tracker is active
    try:
        import mlflow

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_params({k: v for k, v in best_hyperparams.items()})
        logger.info("Metrics logged to MLflow")
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)

    logger.info(
        "Evaluation: RMSE=%.4f MAE=%.4f P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f",
        rmse,
        mae,
        top_k,
        precision,
        top_k,
        recall,
        top_k,
        ndcg,
    )

    return metrics
