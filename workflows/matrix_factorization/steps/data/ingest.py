"""
steps/data_ingestion/ingest.py

ZenML step: ingest_data.

Queries a MovieLens ratings Hive table through Spark SQL and returns its rows as
a pandas DataFrame ZenML artifact.

Config parameters (from pipeline YAML):
    dataset_table: Hive table containing MovieLens ratings.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import ValidationError
from zenml import step
from zenml.client import Client
from zenml.enums import ModelStages

from helpers.s3_client import (
    get_s3_client,
    parse_s3_uri,
    resolve_zenml_s3_credentials,
    s3_get_object_text,
)
from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_DATASET_FIELD_TYPES,
    CFG_INFERENCE_LOGS_EXT,
    CFG_MODEL_NAME,
    CFG_RECS_FIELD_NAMES,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.models import PredictionLog

logger = logging.getLogger(__name__)

_SPARK_APP_NAME = f"{CFG_WORKFLOW_NAME}_ingest"

# --- Ingest Data Step --------------------------------------------------------------------

@step(enable_cache=True)
def ingest_data(
    dataset_table: str = "ml_ratings_1m",
    lookback_days: int = 30,
    spark_master_url: str = "spark://spark-master:7077",
    limit: int | None = None,
    sample_fraction: float | None = None,
) -> Annotated[pd.DataFrame, "raw_ratings"]:
    """
    Query MovieLens ratings from a Hive table through Spark SQL.

    Args:
        dataset_table: Hive table name, optionally qualified with one database,
            containing userId, movieId, rating, and timestamp columns.
        lookback_days: Number of days to return relative to the table's latest
            eventDate partition.
        spark_master_url: Spark cluster master URL used to execute the query.
        sample_fraction: Fraction of rows to randomly sample from the Hive table. Default is None (no sampling).
        limit: Maximum number of rows to ingest from the Hive table. Default is None (no limit).
    Returns:
        pandas DataFrame with columns: userId, movieId, rating, timestamp.
    """

    from helpers.spark_client import query_hive, validate_hive_identifier

    # --- Validate Hive table identifier ---
    validate_hive_identifier(dataset_table)

    # --- Prepare Hive SQL query for recent ratings ---
    quoted_table = ".".join(
        f"`{identifier}`" for identifier in dataset_table.split(".")
    )
    query = f"""
    WITH dataset_window AS (
        SELECT date_sub(MAX(eventDate), {lookback_days}) AS cutoff_date
        FROM {quoted_table}
    )
    SELECT userId, movieId, rating, CAST(timestamp AS BIGINT) AS timestamp
    FROM {quoted_table}
    """
    if sample_fraction is not None:
        query += f" TABLESAMPLE({sample_fraction * 100} PERCENT)"
    if limit is not None:
        query += f" LIMIT {limit}"
    
    query += " WHERE eventDate >= (SELECT cutoff_date FROM dataset_window)"

    # --- Execute Hive SQL query and return results as a pandas DataFrame ---
    df_pandas = query_hive(
        query=query,
        app_name=_SPARK_APP_NAME,
        spark_master_url=spark_master_url,
    )

    logger.info("Returning pandas DataFrame: %d rows", len(df_pandas))
    return df_pandas


def _validate_user_limits(
    max_users: int | None,
    max_user_items: int | None,
) -> None:
    """Validate optional prediction dataset limits."""
    for parameter_name, value in (
        ("max_users", max_users),
        ("max_user_items", max_user_items),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{parameter_name} must be at least 1 when provided")


def _limit_users_and_items(
    df: pd.DataFrame,
    user_column: str,
    max_users: int | None,
    max_user_items: int | None,
    item_order_column: str | None = None,
) -> pd.DataFrame:
    """Limit a DataFrame to its first users and items per user."""
    _validate_user_limits(max_users, max_user_items)

    limited = df
    if max_users is not None:
        selected_users = limited[user_column].drop_duplicates().head(max_users)
        limited = limited[limited[user_column].isin(selected_users)]

    if max_user_items is not None:
        if item_order_column is not None:
            item_rank = limited.groupby(user_column, sort=False)[
                item_order_column
            ].rank(method="first")
            limited = limited[item_rank <= max_user_items]
        else:
            limited = limited.groupby(user_column, sort=False).head(max_user_items)

    return limited.reset_index(drop=True)


# --- Ingest Logs Step --------------------------------------------------------------------

@step(enable_cache=False)
def ingest_prediction_logs(
    model_name: str = CFG_MODEL_NAME,
    model_stage: ModelStages = ModelStages.STAGING,
    logs_path: str = "s3://zenml-predictions/logs",
    lookback_days: int = 7,
    limit: int | None = None,
    sample_fraction: float | None = None,
    max_users: int | None = None,
    max_user_items: int | None = None,
    chunk_size: int = 1000,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> Annotated[pd.DataFrame, "inference_logs"]:
    """
    Load recent inference request logs into a DataFrame with the same schema as the training dataset.

    Expected log format (JSON lines written by any serving app):
        {
            "timestamp": "...", "user_id": int, "top_k": int,
            "latency_ms": float, "count": int,
            "predictions": [{"item_id": int, "score": float}, ...]
        }

    Args:
        logs_path: S3 prefix (or local dir) containing JSONL log files.
        lookback_days: Number of days of logs to load.
        limit: Maximum number of rows to ingest from the Hive table. Default is None (no limit).
        sample_fraction: Fraction of rows to randomly sample from the Hive table. Default is None (no sampling).
        max_users: Maximum number of users to retain, in source order. Default is None.
        max_user_items: Maximum prediction rows to retain per user. Default is None.
        chunk_size: Number of rows to materialize per DataFrame chunk.

    Returns:
        DataFrame with inference log records. Returns empty DataFrame if no logs.
    """

    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    records: Iterator[dict[str, object]]
    _validate_user_limits(max_users, max_user_items)

    client = Client()
    version = client.get_model_version(model_name, model_stage)
    model_version_name = str(version.name)

    access_key_id, secret_access_key = resolve_zenml_s3_credentials(
        zenml_local_s3_secret_name
    )

    if logs_path.startswith("s3://"):
        records = _load_s3_logs(
            logs_path,
            cutoff,
            limit=limit,
            max_users=max_users,
            max_user_items=max_user_items,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=access_key_id,
            seaweedfs_secret_access_key=secret_access_key,
            model_name=model_name,
            model_version=model_version_name,
        )
    else:
        records = _load_filesystem_logs(
            logs_path, cutoff, limit=limit, model_name=model_name, model_version=model_version_name
        )

    dtype_map: dict[str, np.dtype] = {
        CFG_DATASET_FIELD_NAMES.USER_ID.value: np.dtype(
            CFG_DATASET_FIELD_TYPES.USER_ID.value
        ),
        CFG_DATASET_FIELD_NAMES.ITEM_ID.value: np.dtype(
            CFG_DATASET_FIELD_TYPES.ITEM_ID.value
        ),
        CFG_DATASET_FIELD_NAMES.RATING.value: np.dtype(
            CFG_DATASET_FIELD_TYPES.RATING.value
        ),
        CFG_DATASET_FIELD_NAMES.TIMESTAMP.value: np.dtype(
            CFG_DATASET_FIELD_TYPES.TIMESTAMP.value
        ),
    }

    # Materialize records into DataFrame chunks to avoid memory issues with large logs
    buffer: list[dict[str, object]] = []
    chunks: list[pd.DataFrame] = []
    for record in records:
        buffer.append(record)
        if len(buffer) >= chunk_size:
            # Buffer size reached, materialize a DataFrame chunk and clear buffer
            chunks.append(_build_chunk_df(buffer, dtype_map))
            buffer.clear()

    # Materialize any remaining buffered records into a final chunk
    if buffer:
        chunks.append(_build_chunk_df(buffer, dtype_map))

    # Concatenate all chunks into a single DataFrame (or return empty DataFrame if no logs)
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in dtype_map.items()}
        )

    if df.empty:
        logger.warning(
            "No inference logs found at %s (lookback=%d days) for model (%s:%s)",
            logs_path,
            lookback_days,
            model_name,
            model_version_name,
        )
        return df

    df = _limit_users_and_items(
        df,
        user_column=CFG_DATASET_FIELD_NAMES.USER_ID.value,
        max_users=max_users,
        max_user_items=max_user_items,
    )

    if sample_fraction is not None:
        df = df.sample(frac=sample_fraction).reset_index(drop=True)

    logger.info(
        "Loaded %d inference log records from %s for model (%s:%s)",
        len(df),
        logs_path,
        model_name,
        model_version_name,
    )
        
    return df


def _build_chunk_df(
    records: list[dict[str, object]],
    dtype_map: dict[str, np.dtype],
) -> pd.DataFrame:
    """Create a typed DataFrame chunk from buffered flattened records."""

    chunk_df = pd.DataFrame.from_records(records)
    for col, dtype in dtype_map.items():
        if col in chunk_df.columns:
            chunk_df[col] = chunk_df[col].astype(dtype)
    return chunk_df


def _iter_prediction_rows(
    rec: PredictionLog, ts: datetime
) -> Iterator[dict[str, object]]:
    """Yield one flattened row per predicted item from a request log entry."""

    ts_unix = int(ts.timestamp())
    for pred in rec.predictions:
        yield {
            CFG_DATASET_FIELD_NAMES.USER_ID.value: rec.user_id,
            CFG_DATASET_FIELD_NAMES.ITEM_ID.value: pred.item_id,
            CFG_DATASET_FIELD_NAMES.RATING.value: pred.score,
            CFG_DATASET_FIELD_NAMES.TIMESTAMP.value: ts_unix,
        }


def _load_filesystem_logs(
    logs_path: str,
    cutoff: datetime,
    limit: int | None = None,
    model_name: str = CFG_MODEL_NAME,
    model_version: str = "unknown",
) -> Iterator[dict[str, object]]:
    """Yield flattened JSONL log rows from local filesystem directory."""

    import json
    count = 0

    log_dir = Path(logs_path)
    for log_file in sorted(log_dir.glob(f"*{CFG_INFERENCE_LOGS_EXT}")):
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = PredictionLog.model_validate_json(line.strip())
                    ts = datetime.fromisoformat(rec.timestamp)
                    if (
                        ts >= cutoff
                        and (not rec.model_name or rec.model_name == model_name)
                        and (
                            not rec.model_version or rec.model_version == model_version
                        )
                    ):
                        yield from _iter_prediction_rows(rec, ts)
                        count += 1
                        if limit is not None and count >= limit:
                            return
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass


# --- Ingest Batch Recommendations Step --------------------------------------------------


@step(enable_cache=False)
def ingest_batch_predictions(
    model_name: str = CFG_MODEL_NAME,
    model_stage: ModelStages = ModelStages.STAGING,
    batch_output_path: str = "s3://zenml-predictions/batch",
    lookback_days: int = 30,
    sample_fraction: float = 1.0,
    limit: int | None = None,
    max_users: int | None = None,
    max_user_items: int | None = None,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> Annotated[pd.DataFrame, "batch_recommendations"]:
    """
    Load recent batch recommendation Parquet shards for drift monitoring.

    Reads shards written by collect_batch_inference_report at:
        {batch_output_path}/{model_name}/{date}/{model_version}-recommendations/*.parquet

    The score column is renamed to ``rating`` so the DataFrame is compatible with
    the Evidently reference dataset (which uses ``userId`` and ``rating``).

    Args:
        model_name: Registered ZenML model name.
        model_stage: ZenML model stage to resolve the current version.
        batch_output_path: S3 prefix (or local dir) where batch shards live.
        lookback_days: How many past days to scan for shards.
        sample_fraction: Fraction of rows to randomly sample from the loaded data.
        limit: Maximum number of rows to load.
        max_users: Maximum number of users to retain, in source order. Default is None.
        max_user_items: Maximum recommendations to retain per user. Default is None.
        seaweedfs_s3_internal_endpoint: SeaweedFS internal S3 endpoint (local only).
        zenml_local_s3_secret_name: ZenML secret with SeaweedFS credentials.

    Returns:
        DataFrame with columns: userId, rating.  Raises ValueError if empty.
    """
    _validate_user_limits(max_users, max_user_items)

    client = Client()
    version = client.get_model_version(model_name, model_stage)
    model_version_name = str(version.name)

    access_key_id, secret_access_key = resolve_zenml_s3_credentials(
        zenml_local_s3_secret_name
    )

    if batch_output_path.startswith("s3://"):
        df: pd.DataFrame = pd.concat(
            [
                pd.DataFrame(),
                *_load_s3_batch_parquet(
                    batch_output_path=batch_output_path,
                    model_name=model_name,
                    model_version=model_version_name,
                    lookback_days=lookback_days,
                    seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
                    access_key_id=access_key_id,
                    secret_access_key=secret_access_key,
                    limit=limit,
                    max_users=max_users,
                    max_user_items=max_user_items,
                ),
            ],
            ignore_index=True,
        )
    else:
        df: pd.DataFrame = pd.concat(
            [
                pd.DataFrame(),
                *_load_filesystem_batch_parquet(
                    batch_output_path=batch_output_path,
                    model_name=model_name,
                    model_version=model_version_name,
                    lookback_days=lookback_days,
                    limit=limit,
                    max_users=max_users,
                    max_user_items=max_user_items,
                ),
            ],
            ignore_index=True,
        )

    if df is None or df.empty:
        raise ValueError(
            f"No batch recommendation shards found at '{batch_output_path}' "
            f"for model '{model_name}' (version={model_version_name}, "
            f"lookback={lookback_days} days). "
            "Run the batch inference pipeline first."
        )

    if df.empty:
        raise ValueError(
            f"No batch recommendation rows found at '{batch_output_path}' "
            f"for model '{model_name}' (version={model_version_name}, "
            f"lookback={lookback_days} days). "
            "Run the batch inference pipeline first."
        )

    # Rename columns to match Evidently reference schema
    df = df.rename(
        columns={
            CFG_RECS_FIELD_NAMES.USER_ID.value: CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_RECS_FIELD_NAMES.REC_ITEM_ID.value: CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            CFG_RECS_FIELD_NAMES.REC_SCORE.value: CFG_DATASET_FIELD_NAMES.RATING.value,
        }
    )

    df = _limit_users_and_items(
        df,
        user_column=CFG_DATASET_FIELD_NAMES.USER_ID.value,
        max_users=max_users,
        max_user_items=max_user_items,
        item_order_column=CFG_RECS_FIELD_NAMES.REC_RANK.value,
    )

    # Sample the DataFrame if a fraction less than 1.0 is specified.   
    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    logger.info(
        "Loaded %d batch recommendation rows from '%s' (%d date(s) scanned)",
        len(df),
        batch_output_path,
        lookback_days + 1,
    )
    return df


def _load_s3_batch_parquet(
    batch_output_path: str,
    model_name: str,
    model_version: str,
    lookback_days: int,
    seaweedfs_s3_internal_endpoint: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
    limit: int | None = None,
    max_users: int | None = None,
    max_user_items: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield bounded Parquet shards from recent S3 batch outputs."""
    s3 = get_s3_client(
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )
    storage_options: dict | None = None
    if seaweedfs_s3_internal_endpoint and access_key_id and secret_access_key:
        storage_options = {
            "client_kwargs": {"endpoint_url": seaweedfs_s3_internal_endpoint},
            "key": access_key_id,
            "secret": secret_access_key,
        }

    user_column = CFG_RECS_FIELD_NAMES.USER_ID.value
    recs_columns = [
        user_column,
        CFG_RECS_FIELD_NAMES.REC_ITEM_ID.value,
        CFG_RECS_FIELD_NAMES.REC_SCORE.value,
        CFG_RECS_FIELD_NAMES.REC_RANK.value,
    ]
    
    filters = None
    if max_user_items is not None:
        filters = [(CFG_RECS_FIELD_NAMES.REC_RANK.value, "<=", max_user_items)]

    count = 0

    paginator = s3.get_paginator("list_objects_v2")
    selected_users: list[int] = []
    today = datetime.now(UTC).date()
    for date_offset in range(lookback_days + 1):
        date_str = (today - timedelta(days=date_offset)).strftime("%Y-%m-%d")
        s3_prefix = (
            f"{batch_output_path}/{model_name}/{date_str}/"
            f"{model_version}-recommendations"
        )
        bucket, prefix = parse_s3_uri(s3_prefix)

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".parquet"):
                    continue
                shard_uri = f"s3://{bucket}/{obj['Key']}"
                if storage_options:
                    shard = pd.read_parquet(
                        shard_uri,
                        columns=recs_columns,
                        filters=filters,
                        storage_options=storage_options,
                    )
                else:
                    shard = pd.read_parquet(
                        shard_uri,
                        columns=recs_columns,
                        filters=filters,
                    )

                if max_users is not None:
                    selected_user_set = set(selected_users)
                    for user_id in shard[user_column].drop_duplicates():
                        if user_id in selected_user_set:
                            continue
                        if len(selected_users) >= max_users:
                            break
                        selected_users.append(int(user_id))
                        selected_user_set.add(user_id)
                    shard = shard[shard[user_column].isin(selected_user_set)]

                shard = _limit_users_and_items(
                    shard,
                    user_column=user_column,
                    max_users=None,
                    max_user_items=max_user_items,
                    item_order_column=CFG_RECS_FIELD_NAMES.REC_RANK.value,
                )

                count += len(shard)

                if not shard.empty:
                    yield shard

                if max_users is not None and len(selected_users) >= max_users:
                    return
                
                if limit is not None and count >= limit:
                    return


def _load_filesystem_batch_parquet(
    batch_output_path: str,
    model_name: str,
    model_version: str,
    lookback_days: int,
    limit: int | None = None,
    max_users: int | None = None,
    max_user_items: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield bounded Parquet shards from recent filesystem batch outputs."""
    user_column = CFG_RECS_FIELD_NAMES.USER_ID.value
    selected_users: list[int] = []
    today = datetime.now(UTC).date()

    count = 0

    for date_offset in range(lookback_days + 1):
        date_str = (today - timedelta(days=date_offset)).strftime("%Y-%m-%d")
        shard_dir = Path(
            f"{batch_output_path}/{model_name}/{date_str}/{model_version}-recommendations"
        )
        if not shard_dir.exists():
            continue

        for shard_path in sorted(shard_dir.glob("*.parquet")):
            shard = pd.read_parquet(shard_path)

            if max_users is not None:
                selected_user_set = set(selected_users)
                for user_id in shard[user_column].drop_duplicates():
                    if user_id in selected_user_set:
                        continue
                    if len(selected_users) >= max_users:
                        break
                    selected_users.append(int(user_id))
                    selected_user_set.add(user_id)
                shard = shard[shard[user_column].isin(selected_user_set)]

            shard = _limit_users_and_items(
                shard,
                user_column=user_column,
                max_users=max_users,
                max_user_items=max_user_items,
                item_order_column=CFG_RECS_FIELD_NAMES.REC_RANK.value,
            )

            count += len(shard)

            if not shard.empty:
                yield shard
            
            if max_users is not None and len(selected_users) >= max_users:
                return

            if limit is not None and count >= limit:
                return


def _load_s3_logs(
    s3_prefix: str,
    cutoff: datetime,
    limit: int | None = None,
    max_users: int | None = None,
    max_user_items: int | None = None,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
    model_name: str = CFG_MODEL_NAME,
    model_version: str = "unknown",
) -> Iterator[dict[str, object]]:
    """Yield flattened JSONL log rows from S3 prefix."""

    import json

    s3 = get_s3_client(
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    bucket, prefix = parse_s3_uri(s3_prefix)
    selected_users: set[int] = set()
    user_item_counts: dict[int, int] = {}

    count = 0

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3_get_object_text(s3, bucket=bucket, key=obj["Key"])
            for line in body.splitlines():
                try:
                    rec = PredictionLog.model_validate_json(line.strip())
                    ts = datetime.fromisoformat(rec.timestamp)
                    if (
                        ts >= cutoff
                        and (not rec.model_name or rec.model_name == model_name)
                        and (
                            not rec.model_version or rec.model_version == model_version
                        )
                    ):
                        count += 1
                        if rec.user_id not in selected_users:
                            if max_users is not None and len(selected_users) >= max_users:
                                continue
                            selected_users.add(rec.user_id)

                        remaining_items = None
                        if max_user_items is not None:
                            remaining_items = max_user_items - user_item_counts.get(rec.user_id, 0)
                            if remaining_items <= 0:
                                continue

                        for row in _iter_prediction_rows(rec, ts):
                            if remaining_items is not None and remaining_items <= 0:
                                break
                            yield row
                            user_item_counts[rec.user_id] = (
                                user_item_counts.get(rec.user_id, 0) + 1
                            )
                            if remaining_items is not None:
                                remaining_items -= 1

                        if limit is not None and count >= limit:
                            return

                        if (
                            max_users is not None
                            and max_user_items is not None
                            and len(selected_users) >= max_users
                            and all(
                                user_item_counts.get(user_id, 0) >= max_user_items
                                for user_id in selected_users
                            )
                        ):
                            return
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass
