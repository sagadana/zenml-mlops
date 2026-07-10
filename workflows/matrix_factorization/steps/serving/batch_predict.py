"""
steps/serving/batch_predict.py

ZenML step: generate_batch_recommendations

For each user in the encoder, generates top-K recommendations using the
production ALSRecommender model. Writes results to S3 as Parquet and
optionally loads them into a DynamoDB table for real-time lookup.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import pandas as pd
from dask.distributed import Future, as_completed
from dask_expr import DataFrame
from zenml import get_step_context, step
from zenml.client import Client

from helpers.checkpointing import load_latest_checkpoint, save_checkpoint
from helpers.dask_cluster import get_client_mode_from_config, get_dask_client
from workflows.matrix_factorization.configs import (
    CFG_BATCH_PREDICTION_FIELD_NAMES,
    CFG_DASK_SCHEDULER_ADDRESS,
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_NAME,
    CFG_PREDICTION_FIELD_NAMES,
    CFG_RECS_FIELD_NAMES,
)
from workflows.matrix_factorization.models.als_recommender import ALSRecommender
from workflows.matrix_factorization.steps.serving.batch_predict_user import predict_user_batch

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def generate_batch_recommendations(
    batch_top_k: int = 50,
    batch_output_path: str = "s3://aips-recs-zenml-predictions/batch",
    checkpoint_path: str = "./checkpoints",
    model_stage: str = "production",
    user_batch_size: int = 10_000,
    n_parallel_batches: int = 4,
    dynamodb_table: str | None = None,
    dynamodb_partition_key: str = CFG_RECS_FIELD_NAMES.RECORD_ID.value,
    scheduler_address: str | None = CFG_DASK_SCHEDULER_ADDRESS,
) -> Annotated[dict, "batch_job_report"]:
    """
    Pre-compute top-K recommendations for all users and write to S3.

    Args:
        batch_top_k: Number of recommendations per user.
        batch_output_path: S3 path (or local path) for Parquet output.
        dynamodb_table: DynamoDB table name. If set, loads recommendations there.
        dynamodb_partition_key: DynamoDB partition key attribute name.
            Table schema: <partition_key> (PK, String), recommendations (JSON), updated_at (TTL).
        model_stage: ZenML model stage to load from ("production" or "staging").
        user_batch_size: Number of users per prediction batch.
        n_parallel_batches: Maximum concurrently processed user batches on Dask.
        checkpoint_path: Base path (local/S3) for batch progress checkpoints.

    Returns:
        Batch job report dict with counts and output path.
    """
    if user_batch_size <= 0:
        raise ValueError("user_batch_size must be > 0")
    if n_parallel_batches <= 0:
        raise ValueError("n_parallel_batches must be > 0")

    # Load model from ZenML Model Control Plane
    client = Client()
    model_version = client.get_model_version(CFG_MODEL_NAME, model_stage)
    artifact = model_version.get_artifact(CFG_MODEL_ARTIFACT_NAME)
    if artifact is None:
        raise ValueError(
            f"Model artifact '${CFG_MODEL_ARTIFACT_NAME}' not found for {CFG_MODEL_NAME}"
        )

    als_model: ALSRecommender = artifact.load()
    logger.info("Loaded %s for batch prediction", als_model)

    all_user_ids = np.asarray(als_model.user_encoder.index.tolist(), dtype=np.int64)
    logger.info("Generating top-%d recs for %d users...", batch_top_k, len(all_user_ids))
    model_version_name = str(model_version.model.latest_version_name)

    # Unique run ID for checkpoint directory scoping (same pattern as train_als).
    try:
        run_id = get_step_context().pipeline_run.id
    except Exception:
        run_id = str(uuid.uuid4())[:8]

    run_checkpoint_path = f"{checkpoint_path}/{run_id}/batch_recommendations"
    start_batch, checkpoint_meta, _ = load_latest_checkpoint(run_checkpoint_path)
    records_written = int(checkpoint_meta[0]) if checkpoint_meta is not None else 0

    total_users = len(all_user_ids)
    total_batches = (total_users + user_batch_size - 1) // user_batch_size if total_users else 0

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    output_path = f"{batch_output_path}/{date_str}/{model_version_name}-recommendations"

    if total_batches == 0:
        report = {
            "n_users": 0,
            "top_k": batch_top_k,
            "output_path": output_path,
            "n_records": 0,
            "date": date_str,
            "dynamodb_loaded": dynamodb_table is not None,
        }
        return report

    next_batch_idx = start_batch
    pending_futures: dict[Future, int] = {}

    # ── Batching loop ─────────────────────────────────────────────────────────
    with get_dask_client(
        mode=get_client_mode_from_config(scheduler_address), scheduler_address=scheduler_address
    ) as dask_client:
        while next_batch_idx < total_batches or pending_futures:
            # Process batches in parallel, up to n_parallel_batches at a time
            while next_batch_idx < total_batches and len(pending_futures) < n_parallel_batches:
                batch_start = next_batch_idx * user_batch_size
                batch_ids = all_user_ids[batch_start : batch_start + user_batch_size]
                future = dask_client.submit(
                    predict_user_batch,
                    als_model,
                    batch_ids,
                    batch_top_k,
                    model_version_name,
                    pure=False,
                )
                pending_futures[future] = next_batch_idx
                next_batch_idx += 1

            # Wait for any batch to complete and process its results
            future: Future
            for future in as_completed(list(pending_futures.keys())):
                batch_idx = pending_futures.pop(future)
                batch_df = future.result()

                if isinstance(batch_df, DataFrame | pd.DataFrame):
                    shard_path = f"{output_path}/batch_{batch_idx:06d}.parquet"
                    batch_df.to_parquet(shard_path, index=False)
                    records_written += len(batch_df)

                    if dynamodb_table:
                        _load_to_dynamodb(
                            df=batch_df,
                            table_name=dynamodb_table,
                            partition_key_name=dynamodb_partition_key,
                            top_k=batch_top_k,
                        )

                    save_checkpoint(
                        epoch=batch_idx + 1,
                        primary=np.asarray([records_written], dtype=np.int64),
                        secondary=None,
                        base_path=run_checkpoint_path,
                    )
                    logger.info(
                        "Processed batch %d/%d (%d users, %d rows)",
                        batch_idx + 1,
                        total_batches,
                        len(batch_df[CFG_RECS_FIELD_NAMES.USER_ID.value].unique()),
                        len(batch_df),
                    )

    report = {
        "n_users": total_users,
        "top_k": batch_top_k,
        "output_path": output_path,
        "n_records": records_written,
        "date": date_str,
        "dynamodb_loaded": dynamodb_table is not None,
    }
    return report


def _load_to_dynamodb(
    df: pd.DataFrame | DataFrame,
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
    ttl_seconds = int(time.time()) + 48 * 3600  # 48-hour TTL
    count = 0

    # Group by userId
    grouped = df.sort_values(
        [
            CFG_RECS_FIELD_NAMES.USER_ID.value,
            CFG_RECS_FIELD_NAMES.REC_RANK.value,
        ]
    ).groupby(CFG_RECS_FIELD_NAMES.USER_ID.value)

    # Batch write (max 25 items per DynamoDB batch)
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

    logger.info(
        "Loaded %d user recommendation lists to DynamoDB table '%s'",
        count,
        table_name,
    )
