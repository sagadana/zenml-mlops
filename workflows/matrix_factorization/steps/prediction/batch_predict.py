"""
steps/serving/batch_predict.py

ZenML steps for fan-out batch recommendation serving:

  load_als_model              → als_model, model_version_name
  predict_user_batch (×N)     → batch_summary  [fan-out — each step stores its own shard]
  collect_batch_recommendations → batch_job_report  [fan-in — aggregates summaries]

The serving_pipeline fans out predict_user_batch for n_batches parallel steps.
Each step independently writes its Parquet shard and optionally loads DynamoDB.
collect_batch_recommendations fans in the per-batch summary dicts and returns
an aggregated report without any storage work of its own.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import get_step_context, step
from zenml.client import Client
from zenml.enums import ModelStages, StepRuntime

from workflows.matrix_factorization.configs import (
    CFG_BATCH_USER_SUMMARY_OUTPUT,
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_NAME,
)
from workflows.matrix_factorization.models.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


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

    logger.info("Loaded model version %s - '%s' (%s stage)", model_name, model_version, model_stage)

    return model, model_name, model_version


@step(enable_cache=False, runtime=StepRuntime.ISOLATED)
def collect_batch_recommendations(
    n_batches: int,
    step_prefix: str = "predict_user_batch_",
) -> Annotated[dict, "batch_job_report"]:
    """
    Fan-in: collect batch_summary dicts from all predict_user_batch steps and
    return an aggregated job report.

    Each predict_user_batch step independently stores its own Parquet shard and
    optionally loads DynamoDB, so this step only aggregates metadata.

    Args:
        n_batches: Expected number of batch steps (used for validation logging).
        step_prefix: Prefix for batch step names (default: "predict_user_batch_").

    Returns:
        Aggregated batch job report dict.
    """
    client = Client()
    step_ctx = get_step_context()
    run = client.get_pipeline_run(step_ctx.pipeline_run.name)

    total_users = 0
    total_records = 0
    batches_collected = 0
    shard_paths: list[str] = []
    dynamodb_loaded = False

    for step_name, step_info in run.steps.items():
        if not step_name.startswith(step_prefix):
            continue
        if CFG_BATCH_USER_SUMMARY_OUTPUT not in step_info.outputs:
            continue

        output = step_info.outputs[CFG_BATCH_USER_SUMMARY_OUTPUT][0]
        summary: dict = output.load()

        n_users = summary.get("n_users", 0)
        n_records = summary.get("n_records", 0)
        shard_path = summary.get("shard_path", "")

        total_users += n_users
        total_records += n_records
        shard_paths.append(shard_path)
        dynamodb_loaded = dynamodb_loaded or summary.get("dynamodb_loaded", False)
        batches_collected += 1

        logger.info(
            "Collected '%s': %d users, %d rows → %s",
            step_name,
            n_users,
            n_records,
            shard_path,
        )

    if batches_collected < n_batches:
        logger.warning("Expected %d batches but only collected %d", n_batches, batches_collected)

    return {
        "n_batches": batches_collected,
        "n_users": total_users,
        "n_records": total_records,
        "shard_paths": shard_paths,
        "dynamodb_loaded": dynamodb_loaded,
    }
