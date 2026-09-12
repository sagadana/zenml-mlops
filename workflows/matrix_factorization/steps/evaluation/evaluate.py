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
from zenml import step
from zenml.client import Client
from zenml.enums import ModelStages, StepRuntime
from zenml.integrations.evidently.column_mapping import EvidentlyColumnMapping
from zenml.integrations.evidently.data_validators import EvidentlyDataValidator
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.types import HTMLString

from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_FEATURES_FIELD_NAMES,
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_NAME,
)
from workflows.matrix_factorization.models.base_recommender import BaseRecommender
from workflows.matrix_factorization.models.numba import warmup_jit

logger = logging.getLogger(__name__)

warmup_jit()  # Warm up the Numba JIT compiler for compute_rmse


@step(enable_cache=False, runtime=StepRuntime.INLINE)
def fetch_previous_model_factors(
    model_stage: ModelStages = ModelStages.STAGING,
) -> tuple[
    Annotated[np.ndarray, "previous_user_factors"],
    Annotated[np.ndarray, "previous_item_factors"],
    Annotated[pd.Series, "previous_user_encoder"],
    Annotated[pd.Series, "previous_item_encoder"],
    Annotated[bool, "previous_model_available"],
]:
    """Load factors and encoders from the model currently at ``model_stage``."""
    try:
        model_version = Client().get_model_version(CFG_MODEL_NAME, model_stage)
        artifact = model_version.get_artifact(CFG_MODEL_ARTIFACT_NAME)
        if artifact is None:
            raise ValueError(
                f"Model artifact '{CFG_MODEL_ARTIFACT_NAME}' not found for {CFG_MODEL_NAME}"
            )
        model: BaseRecommender = artifact.load()
    except Exception as exc:
        logger.warning(
            "No previous model could be loaded from stage '%s': %s. "
            "The regression check will be skipped.",
            model_stage,
            exc,
        )
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            pd.Series(dtype="int32"),
            pd.Series(dtype="int32"),
            False,
        )

    logger.info("Loaded previous model '%s' from stage '%s'", model.version, model_stage)
    return (
        model.user_factors,
        model.item_factors,
        model.user_encoder,
        model.item_encoder,
        True,
    )


@step(enable_cache=True)
def compute_metrics(
    test_data: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    model_available: bool = True,
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
        user_encoder: Mapping from raw user IDs to model factor indices.
        item_encoder: Mapping from raw item IDs to model factor indices.
        model_available: Whether the model factors were loaded successfully.
        top_k: K for ranking metrics.
        sample_seed: Random seed for sampling users for ranking metrics.
        sample_size: Max number of users to sample for ranking metrics (for efficiency).

    Returns:
        eval_metrics dict with RMSE, MAE, Precision@K, Recall@K, NDCG@K.
    """

    if not model_available:
        return {"available": False, "top_k": top_k}

    user_col = CFG_DATASET_FIELD_NAMES.USER_ID.value
    item_col = CFG_DATASET_FIELD_NAMES.ITEM_ID.value
    rating_col = CFG_FEATURES_FIELD_NAMES.RATING.value

    known_rows = test_data[user_col].isin(user_encoder.index) & test_data[item_col].isin(
        item_encoder.index
    )
    test_pd = test_data.loc[known_rows].copy()
    if test_pd.empty:
        logger.warning("No evaluation rows are known to this model; metrics are unavailable")
        return {"available": False, "top_k": top_k}

    # Sample users for efficiency if the test set is large
    sampled = test_pd.copy()
    unique_users = sampled[user_col].unique()
    if len(unique_users) > sample_size:
        sampled_users = np.random.default_rng(sample_seed).choice(
            unique_users, sample_size, replace=False
        )
        sampled = sampled[sampled[user_col].isin(sampled_users)]

    # Compute RMSE and Ranking metrics on the sampled test set
    sorted_df = sampled.sort_values(user_col)
    user_ids = np.asarray(
        user_encoder.loc[sorted_df[user_col]].values,
        dtype=np.int32,
    )
    item_ids = np.asarray(
        item_encoder.loc[sorted_df[item_col]].values,
        dtype=np.int32,
    )
    ratings = np.asarray(sorted_df[rating_col].values, dtype=np.float32)

    rmse, precision, recall, ndcg = BaseRecommender.compute_metrics(
        user_indices=user_ids,
        item_indices=item_ids,
        ratings=ratings,
        user_factors=user_factors,
        item_factors=item_factors,
        k=top_k,
    )

    metrics = {
        "available": True,
        "top_k": top_k,
        "rmse": rmse,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "ndcg_at_k": ndcg,
        "n_test_ratings": len(ratings),
        "n_test_users": len(np.unique(user_ids)),
        "n_test_items": len(np.unique(item_ids)),
    }

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


@step(enable_cache=False)
def quality_check(
    new_metrics: dict,
    previous_metrics: dict,
    precision_at_k_threshold: float = 0.1,
    recall_at_k_threshold: float = 0.1,
    ndcg_at_k_threshold: float = 0.1,
    force_promote: bool = False,
) -> Annotated[bool, "quality_check_passed"]:
    """Apply absolute quality thresholds and previous-model regression checks."""
    threshold_failures: list[str] = []
    for metric_name, threshold in (
        ("precision_at_k", precision_at_k_threshold),
        ("recall_at_k", recall_at_k_threshold),
        ("ndcg_at_k", ndcg_at_k_threshold),
    ):
        value = float(new_metrics[metric_name])
        if value < threshold:
            threshold_failures.append(f"{metric_name} {value:.4f} < threshold {threshold:.4f}")

    regressions: list[str] = []
    if previous_metrics.get("available", False):
        if previous_metrics["top_k"] != new_metrics["top_k"]:
            logger.warning(
                "Previous and new model metrics use different K values (%s and %s); "
                "the regression check cannot be performed.",
                previous_metrics["top_k"],
                new_metrics["top_k"],
            )
            regressions.append("evaluation K differs from the previous model")
        else:
            for metric_name in ("precision_at_k", "recall_at_k", "ndcg_at_k"):
                current = float(new_metrics[metric_name])
                previous = float(previous_metrics[metric_name])
                if current < previous:
                    regressions.append(f"{metric_name} {current:.4f} < previous {previous:.4f}")
    else:
        logger.info("No previous model metrics are available; skipping regression checks")

    passed = not threshold_failures and not regressions
    if force_promote and not passed:
        logger.warning(
            "Quality check was overridden despite: %s",
            "; ".join(threshold_failures + regressions),
        )
        return True

    if passed:
        logger.info("Model quality check PASSED")
    else:
        logger.warning(
            "Model quality check FAILED: %s",
            "; ".join(threshold_failures + regressions),
        )
    return passed


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
