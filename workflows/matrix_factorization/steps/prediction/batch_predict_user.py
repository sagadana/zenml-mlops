"""
steps/serving/batch_predict_user.py

ZenML steps: get_total_users, predict_user_batch

Generates top-K recommendations for a contiguous slice of users, stores them
to S3 and optionally DynamoDB, and returns a batch summary dict.
Fan-out pattern: get_total_users returns the total user count and batch size;
predict_user_batch computes its own start/end slice from batch_idx, runs
inference, writes predictions, and returns a summary.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step
from zenml.client import Client
from zenml.enums import StepRuntime

from workflows.matrix_factorization.configs import (
    CFG_BATCH_USER_SUMMARY_OUTPUT,
    CFG_RECS_FIELD_NAMES,
)
from workflows.matrix_factorization.models.base_recommender import BaseRecommender, PredictionItem

logger = logging.getLogger(__name__)

KEY_ACCESS_KEY_ID = "access_key_id"
KEY_SECRET_ACCESS_KEY = "secret_access_key"


def _iter_recommendation_rows(
    batch_predictions: dict[str, list[PredictionItem]],
    model_name: str,
    model_version: str,
) -> Iterator[dict]:
    """Yield flat recommendation rows for one user batch."""
    model_id_prefix = model_name.replace("_", "-").lower()
    for user_id_str, recommendations in batch_predictions.items():
        uid = int(user_id_str)
        for rank_pos, rec in enumerate(recommendations):
            yield {
                CFG_RECS_FIELD_NAMES.RECORD_ID.value: f"{model_id_prefix}-{uid}",
                CFG_RECS_FIELD_NAMES.USER_ID.value: uid,
                CFG_RECS_FIELD_NAMES.REC_ITEM_ID.value: rec.item_id,
                CFG_RECS_FIELD_NAMES.REC_SCORE.value: rec.score,
                CFG_RECS_FIELD_NAMES.REC_RANK.value: rank_pos + 1,
                CFG_RECS_FIELD_NAMES.VERSION.value: model_version,
            }


def _resolve_s3_storage_options(
    path: str,
    client: Client,
    seaweedfs_s3_internal_endpoint: str | None,
    zenml_local_s3_secret_name: str | None,
) -> dict | None:
    """Build fsspec storage_options for S3 paths (SeaweedFS or AWS)."""
    if not path.startswith("s3://"):
        return None

    if not seaweedfs_s3_internal_endpoint:
        # For AWS S3 or default credentials chain, let pandas/s3fs resolve auth.
        return None

    if not zenml_local_s3_secret_name:
        raise ValueError(
            "SeaweedFS endpoint is set but zenml_local_s3_secret_name is missing. "
            "Provide a ZenML secret with access_key_id and secret_access_key."
        )

    secret = client.get_secret(zenml_local_s3_secret_name)
    access_key_id = secret.secret_values.get(KEY_ACCESS_KEY_ID)
    secret_access_key = secret.secret_values.get(KEY_SECRET_ACCESS_KEY)
    if not access_key_id or not secret_access_key:
        raise ValueError(
            f"ZenML secret '{zenml_local_s3_secret_name}' is missing keys "
            f"'{KEY_ACCESS_KEY_ID}' and/or '{KEY_SECRET_ACCESS_KEY}'."
        )

    return {
        "client_kwargs": {"endpoint_url": seaweedfs_s3_internal_endpoint},
        "key": access_key_id,
        "secret": secret_access_key,
    }


def _load_to_dynamodb(
    df: pd.DataFrame,
    table_name: str,
    partition_key_name: str,
    top_k: int,
    region_name: str | None = None,
) -> None:
    """Write user recommendation lists to DynamoDB."""
    import boto3

    dynamodb = boto3.resource("dynamodb", region_name=region_name)
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


@step(enable_cache=True, runtime=StepRuntime.INLINE)
def get_total_users(
    model: BaseRecommender,
    n_batches: int,
    min_user_batch_size: int,
) -> tuple[
    Annotated[int, "total_users"],
    Annotated[int, "batch_size"],
]:
    """
    Get the total number of users from the model and compute the effective batch size.

    Args:
        model: Loaded ALS recommender (passed from load_als_model step).
        n_batches: Number of fan-out batches.
        min_user_batch_size: Minimum users per batch; actual batch size is
            max(min_user_batch_size, ceil(total_users / n_batches)).

    Returns:
        total_users: Total number of users in the model.
        batch_size: Effective number of users per batch.
    """
    total_users = len(model.user_encoder.index)
    batch_size = max(min_user_batch_size, math.ceil(total_users / n_batches))
    return total_users, batch_size


@step(enable_cache=True, runtime=StepRuntime.ISOLATED)
def predict_user_batch(
    total_users: int,
    batch_size: int,
    batch_idx: int,
    model: BaseRecommender,
    model_name: str,
    model_version: str,
    batch_top_k: int,
    batch_output_path: str = "./predictions/batch",
    dynamodb_table: str | None = None,
    dynamodb_partition_key: str = CFG_RECS_FIELD_NAMES.RECORD_ID.value,
    dynamodb_region: str | None = None,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> Annotated[dict, CFG_BATCH_USER_SUMMARY_OUTPUT]:
    """
    Generate top-K recommendations for one batch of users, store them, and
    return a summary dict.

    Each fan-out step writes its own Parquet shard to S3 (or local disk) and
    optionally loads records into DynamoDB independently, enabling true
    parallelism without a storage bottleneck in the fan-in step.

    Args:
        total_users: Total number of users in the model (from get_total_users).
        batch_size: Effective users per batch (from get_total_users).
        batch_idx: Zero-based batch index (used to compute slice and name the output shard).
        model: Loaded ALS recommender (passed from load_als_model step).
        model_name: Name of the ALS model (passed from load_als_model step).
        model_version: Version of the ALS model (passed from load_als_model step).
        batch_top_k: Number of top recommendations to generate per user.
        batch_output_path: Base path (local or S3) for Parquet shards.
        dynamodb_table: DynamoDB table name. If set, loads recommendations there.
        dynamodb_partition_key: DynamoDB partition key attribute name.
        dynamodb_region: AWS region for DynamoDB client/resource.
        seaweedfs_s3_internal_endpoint: SeaweedFS internal S3 endpoint.
        zenml_local_s3_secret_name: ZenML secret with SeaweedFS credentials.

    Returns:
        Summary dict: batch_idx, n_users, n_records, shard_path, dynamodb_loaded.
    """
    client = Client()
    effective_model_name = model.name or model_name
    effective_model_version = model.version or model_version

    # --- Step 1: Inference — generate top-K recommendations for the user slice ---
    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, total_users)
    batch_ids = np.asarray(model.user_encoder.index[batch_start:batch_end].tolist(), dtype=np.int64)
    batch_predictions = model.batch_predict(batch_ids, top_k=batch_top_k)

    batch_df = pd.DataFrame.from_records(
        _iter_recommendation_rows(
            batch_predictions.predictions,
            model_name=effective_model_name,
            model_version=effective_model_version,
        )
    )

    # --- Step 2: S3 — write this batch's Parquet shard independently ---
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    batch_range_str = f"{batch_start:08d}-{batch_end:08d}"
    output_path = f"{batch_output_path}/{effective_model_name}/{date_str}/{effective_model_version}-recommendations"
    shard_path = f"{output_path}/batch_{batch_range_str}.parquet"

    storage_options = _resolve_s3_storage_options(
        path=output_path,
        client=client,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        zenml_local_s3_secret_name=zenml_local_s3_secret_name,
    )

    if storage_options is None:
        batch_df.to_parquet(shard_path, index=False)
    else:
        batch_df.to_parquet(shard_path, index=False, storage_options=storage_options)

    n_users = batch_df[CFG_RECS_FIELD_NAMES.USER_ID.value].nunique()

    # --- Step 3: DynamoDB — load recommendations (AWS only, skipped on local orchestrator) ---
    can_load_dynamodb = (
        dynamodb_table is not None and client.active_stack.orchestrator.flavor != "local"
    )

    if can_load_dynamodb:
        _load_to_dynamodb(
            df=batch_df,
            table_name=dynamodb_table or "",
            partition_key_name=dynamodb_partition_key,
            top_k=batch_top_k,
            region_name=dynamodb_region,
        )

    # --- Step 4: Summary — return metadata for fan-in aggregation ---
    logger.info(
        "Batch %d: %d users, %d rows → %s",
        batch_idx,
        n_users,
        len(batch_df),
        shard_path,
    )

    return {
        "batch_idx": batch_idx,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "n_users": n_users,
        "n_records": len(batch_df),
        "shard_path": shard_path,
        "dynamodb_loaded": can_load_dynamodb,
    }
