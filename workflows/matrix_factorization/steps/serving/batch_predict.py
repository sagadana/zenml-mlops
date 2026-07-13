"""
steps/serving/batch_predict.py

ZenML steps for fan-out batch recommendation serving:

  load_als_model              → als_model, model_version_name
  predict_user_batch (×N)     → batch_recommendations  [fan-out in serving_pipeline]
  collect_batch_recommendations → batch_job_report     [fan-in]

The serving_pipeline fans out predict_user_batch for n_batches parallel steps,
then fans in with collect_batch_recommendations which reads all batch outputs
via the ZenML Client API.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

import pandas as pd
from zenml import get_step_context, step
from zenml.client import Client
from zenml.enums import ModelStages

from workflows.matrix_factorization.configs import (
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_NAME,
    CFG_RECS_FIELD_NAMES,
)
from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def load_als_model(
    model_stage: ModelStages = ModelStages.STAGING,
) -> tuple[
    Annotated[ALSRecommender, "als_model"],
    Annotated[str, "model_version_name"],
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
    als_model: ALSRecommender = artifact.load()
    model_version_name = str(model_version.model.latest_version_name)
    logger.info("Loaded model version '%s' (%s stage)", model_version_name, model_stage)
    return als_model, model_version_name


@step(enable_cache=False)
def collect_batch_recommendations(
    n_batches: int,
    model_version_name: str,
    batch_output_path: str = "s3://zenml-predictions/batch",
    batch_top_k: int = 50,
    dynamodb_table: str | None = None,
    dynamodb_partition_key: str = CFG_RECS_FIELD_NAMES.RECORD_ID.value,
) -> Annotated[dict, "batch_job_report"]:
    """
    Fan-in: collect all batch_* step outputs and write to S3 / DynamoDB.

    Reads batch_recommendations artifacts from all steps whose name starts
    with "batch_" in the current pipeline run (set via after= in serving_pipeline).

    Args:
        n_batches: Expected number of batch steps (used for logging/validation).
        model_version_name: Model version string for the output path.
        batch_output_path: Base path (local or S3) for Parquet shards.
        batch_top_k: Number of recommendations per user (logged in report).
        dynamodb_table: DynamoDB table name. If set, loads recommendations there.
        dynamodb_partition_key: DynamoDB partition key attribute name.

    Returns:
        Batch job report dict with n_users, n_records, output_path, date.
    """
    run_name = get_step_context().pipeline_run.name
    run = Client().get_pipeline_run(run_name)

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    output_path = f"{batch_output_path}/{date_str}/{model_version_name}-recommendations"

    records_written = 0
    batches_collected = 0

    for step_name, step_info in run.steps.items():
        if not step_name.startswith("batch_"):
            continue
        if "batch_recommendations" not in step_info.outputs:
            continue

        output = step_info.outputs["batch_recommendations"][0]
        batch_df: pd.DataFrame = output.load()

        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            logger.warning("Step '%s' produced empty or invalid output, skipping", step_name)
            continue

        shard_path = f"{output_path}/{step_name}.parquet"
        batch_df.to_parquet(shard_path, index=False)
        records_written += len(batch_df)
        batches_collected += 1

        if dynamodb_table:
            _load_to_dynamodb(
                df=batch_df,
                table_name=dynamodb_table,
                partition_key_name=dynamodb_partition_key,
                top_k=batch_top_k,
            )

        n_users_in_batch = batch_df[CFG_RECS_FIELD_NAMES.USER_ID.value].nunique()
        logger.info(
            "Collected '%s': %d users, %d rows → %s",
            step_name,
            n_users_in_batch,
            len(batch_df),
            shard_path,
        )

    if batches_collected < n_batches:
        logger.warning("Expected %d batches but only collected %d", n_batches, batches_collected)

    return {
        "n_batches": batches_collected,
        "n_records": records_written,
        "top_k": batch_top_k,
        "output_path": output_path,
        "date": date_str,
        "dynamodb_loaded": dynamodb_table is not None,
    }


def _load_to_dynamodb(
    df: pd.DataFrame,
    table_name: str,
    partition_key_name: str,
    top_k: int,
) -> None:
    """Write user recommendation lists to DynamoDB."""
    import json
    import time

    import boto3

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)  # type: ignore[arg-type]
    ttl_seconds = int(time.time()) + 48 * 3600
    count = 0

    grouped = df.sort_values(
        [CFG_RECS_FIELD_NAMES.USER_ID.value, CFG_RECS_FIELD_NAMES.REC_RANK.value]
    ).groupby(CFG_RECS_FIELD_NAMES.USER_ID.value)

    with table.batch_writer() as batch:
        for user_id, group in grouped:
            recs = group[
                [
                    CFG_RECS_FIELD_NAMES.REC_ITEM_ID.value,
                    CFG_RECS_FIELD_NAMES.REC_SCORE.value,
                    CFG_RECS_FIELD_NAMES.REC_RANK.value,
                ]
            ].to_dict("records")
            batch.put_item(
                Item={
                    partition_key_name: str(user_id),
                    CFG_RECS_FIELD_NAMES.RECS.value: json.dumps(recs[:top_k]),
                    CFG_RECS_FIELD_NAMES.UPDATED_AT.value: ttl_seconds,
                }
            )
            count += 1

    logger.info("Loaded %d user recommendation lists to DynamoDB '%s'", count, table_name)
