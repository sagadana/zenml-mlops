"""
steps/serving/batch_predict.py

ZenML steps for fan-out batch recommendation serving:

  load_als_model              → als_model, model_version_name
  predict_user_batch (×N)     → batch_summary  [fan-out — each step stores its own shard]
  collect_batch_inference_report → batch_job_report  [fan-in — aggregates summaries]

The serving_pipeline fans out predict_user_batch for n_batches parallel steps.
Each step independently writes its Parquet shard and optionally loads DynamoDB.
collect_batch_inference_report fans in the per-batch summary dicts and returns
an aggregated report without any storage work of its own.
"""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import BaseModel
from zenml import get_step_context, step
from zenml.client import Client
from zenml.enums import ModelStages, StepRuntime

from workflows.matrix_factorization.configs import (
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_NAME,
)
from workflows.matrix_factorization.models.base_recommender import BaseRecommender
from workflows.matrix_factorization.steps.prediction.batch_predict_user import (
    BatchPredictUserSummary,
)

logger = logging.getLogger(__name__)


class BatchPredictReport(BaseModel):
    n_batches: int
    n_users: int
    n_records: int
    shard_paths: list[str]
    dynamodb_loaded: bool


@step(enable_cache=False, runtime=StepRuntime.INLINE)
def load_als_model(
    model_stage: ModelStages = ModelStages.STAGING,
) -> tuple[
    Annotated[BaseRecommender, "model"],
    Annotated[str, "model_name"],
    Annotated[str, "model_version"],
]:
    """
    Load the ALS model from the ZenML Model Control Plane.

    Args:
        model_stage: ZenML model stage ("production" or "staging").

    Returns:
        als_model: Loaded ALSRecommender instance.
        model_version_name: Model version string (used to label batch outputs).
    """
    client = Client()
    model_version = client.get_model_version(CFG_MODEL_NAME, model_stage)
    artifact = model_version.get_artifact(CFG_MODEL_ARTIFACT_NAME)
    if artifact is None:
        raise ValueError(
            f"Model artifact '{CFG_MODEL_ARTIFACT_NAME}' not found for {CFG_MODEL_NAME}"
        )

    model: BaseRecommender = artifact.load()
    model_name = model.name or str(model_version.model.name)
    model_version = model.version or str(model_version.name)

    logger.info(
        "Loaded model version %s - '%s' (%s stage)",
        model_name,
        model_version,
        model_stage,
    )

    return model, model_name, model_version


@step(enable_cache=False, runtime=StepRuntime.ISOLATED)
def collect_batch_inference_report(
    n_batches: int,
    summaries: list[BatchPredictUserSummary],
) -> Annotated[BatchPredictReport, "batch_predict_report"]:
    """
    Fan-in: collect batch_summary dicts from all predict_user_batch steps and
    return an aggregated job report.

    Each predict_user_batch step independently stores its own Parquet shard and
    optionally loads DynamoDB, so this step only aggregates metadata.

    Args:
        n_batches: Expected number of batch steps (used for validation logging).
        summaries: List of BatchPredictUserSummary instances from all batch steps.

    Returns:
        Aggregated batch prediction report as a BatchPredictReport instance.
    """

    total_users = 0
    total_records = 0
    batches_collected = 0
    shard_paths: list[str] = []
    dynamodb_loaded = False

    for summary in summaries:

        n_users = summary.n_users
        n_records = summary.n_records
        shard_path = summary.shard_path

        total_users += summary.n_users
        total_records += summary.n_records
        dynamodb_loaded = dynamodb_loaded or summary.dynamodb_loaded
        if summary.shard_path:
            shard_paths.append(summary.shard_path)
            batches_collected += 1

        logger.info(
            "Collected batch %d: %d users, %d rows → %s",
            batches_collected,
            n_users,
            n_records,
            shard_path,
        )

    if batches_collected < n_batches:
        logger.warning(
            "Expected %d batches but only collected %d", n_batches, batches_collected
        )

    return BatchPredictReport(
        n_batches=batches_collected,
        n_users=total_users,
        n_records=total_records,
        shard_paths=shard_paths,
        dynamodb_loaded=dynamodb_loaded,
    )
