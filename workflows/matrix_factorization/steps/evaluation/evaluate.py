"""
steps/model_evaluation/evaluate.py

ZenML step: compute_metrics

Distributed evaluation of the trained ALS model on the held-out test set.
Computes RMSE, MAE, Precision@K, Recall@K, NDCG@K.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated, Any, cast

import numpy as np
import pandas as pd
from evidently.legacy.pipeline.column_mapping import TaskType
from zenml import log_metadata, step
from zenml.integrations.evidently.column_mapping import EvidentlyColumnMapping
from zenml.integrations.evidently.data_validators import EvidentlyDataValidator
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.types import HTMLString

from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.base_recommender import BaseRecommender
from workflows.matrix_factorization.models.numba import warmup_jit

logger = logging.getLogger(__name__)

warmup_jit()  # Warm up the Numba JIT compiler for compute_rmse


@step(enable_cache=True)
def compute_metrics(
    test_data: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    top_k: int = 10,
    sample_seed: int = 42,
    sample_size: int = 50_000,  # defult: sample up to 50k users for efficiency
) -> Annotated[dict, "eval_metrics"]:
    """
    Evaluate the trained ALS model on the test set.

    Args:
        test_data: Test split pandas DataFrame.
        user_factors: Trained user factor matrix (n_users × factors).
        item_factors: Trained item factor matrix (n_items × factors).
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

    rmse, precision, recall, ndcg = BaseRecommender.compute_metrics(
        user_indices=user_ids,
        item_indices=item_ids,
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
        metadata=metrics,
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


@step
def evidently_report(
    reference_dataset: pd.DataFrame,
    comparison_dataset: pd.DataFrame | None = None,
    column_mapping: EvidentlyColumnMapping | None = None,
    user_id_column: str | None = None,
    item_id_column: str | None = None,
    ignored_cols: list[str] | None = None,
    metrics: list[EvidentlyMetricConfig] | None = None,
    report_options: Sequence[tuple[str, dict[str, Any]]] | None = None,
    download_nltk_data: bool = False,
) -> tuple[Annotated[str, "report_json"], Annotated[HTMLString, "report_html"]]:
    """Generate an Evidently report on one or two pandas datasets.

    Args:
        reference_dataset: a Pandas DataFrame
        comparison_dataset: a Pandas DataFrame of new data you wish to
            compare against the reference data
        column_mapping: properties of the DataFrame columns used
        ignored_cols: columns to ignore during the Evidently report step
        metrics: a list of Evidently metric configurations to use for the
            report.
        report_options: a list of tuples containing the name of the report
            and a dictionary of options for the report.
        download_nltk_data: whether to download the NLTK data for the report
            step. Defaults to False.

    Returns:
        A tuple containing the Evidently report in JSON and HTML
        formats.
    """
    if not metrics:
        metrics = EvidentlyMetricConfig.default_metrics()

    data_validator = cast(
        EvidentlyDataValidator,
        EvidentlyDataValidator.get_active_data_validator(),
    )

    if ignored_cols:
        exception_msg = (
            "Columns {extra_cols} configured in the `ignored_cols` "
            "parameter are not found in the {dataset} dataset. "
        )
        extra_cols = set(ignored_cols) - set(reference_dataset.columns)
        if extra_cols:
            logger.warning(exception_msg.format(extra_cols=extra_cols, dataset="reference"))
        reference_dataset = reference_dataset.drop(
            labels=list(set(ignored_cols) - extra_cols), axis=1
        )

        if comparison_dataset is not None:
            extra_cols = set(ignored_cols) - set(comparison_dataset.columns)
            if extra_cols:
                logger.warning(exception_msg.format(extra_cols=extra_cols, dataset="comparison"))

            comparison_dataset = comparison_dataset.drop(
                labels=list(set(ignored_cols) - extra_cols), axis=1
            )

    if column_mapping:
        evidently_column_mapping = column_mapping.to_evidently_column_mapping()
        evidently_column_mapping.user_id = user_id_column or evidently_column_mapping.user_id
        evidently_column_mapping.item_id = item_id_column or evidently_column_mapping.item_id
        evidently_column_mapping.task = (
            evidently_column_mapping.task or TaskType.RECOMMENDER_SYSTEMS
        )
    else:
        evidently_column_mapping = None

    report = data_validator.data_profiling(
        dataset=reference_dataset,
        comparison_dataset=comparison_dataset,
        profile_list=metrics,
        column_mapping=evidently_column_mapping,
        report_options=report_options or [],
        download_nltk_data=download_nltk_data,
    )
    return report.json(), HTMLString(report.get_html())
