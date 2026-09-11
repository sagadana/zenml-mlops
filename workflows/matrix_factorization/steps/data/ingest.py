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
import os
import re
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

_HIVE_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$"
)
_SPARK_IDENTITY = "root"
_JAVA_USER_NAME_OPTION = f"-Duser.name={_SPARK_IDENTITY}"

# --- Ingest Data Step --------------------------------------------------------------------


@step(enable_cache=True)
def ingest_data(
    dataset_table: str = "ml_ratings_1m",
    lookback_days: int = 30,
    make_recent: bool = False,
    spark_master_url: str = "spark://spark-master:7077",
) -> Annotated[pd.DataFrame, "raw_ratings"]:
    """
    Query MovieLens ratings from a Hive table through Spark SQL.

    Args:
        dataset_table: Hive table name, optionally qualified with one database,
            containing userId, movieId, rating, and timestamp columns.
        lookback_days: Number of recent days of ratings to return.
        make_recent: Shift static timestamps to the present before filtering. Enable
            only for local MovieLens fixtures; production tables should be current.
        spark_master_url: Spark cluster master URL used to execute the query.

    Returns:
        pandas DataFrame with columns: userId, movieId, rating, timestamp.
    """

    # --- Prepare Hive SQL query for recent ratings ---
    quoted_table = ".".join(
        f"`{identifier}`" for identifier in dataset_table.split(".")
    )
    timestamp_expression = (
        "timestamp + unix_timestamp(current_timestamp()) - max(timestamp) OVER ()"
        if make_recent
        else "timestamp"
    )
    cutoff_expression = (
        f"unix_timestamp(current_timestamp()) - {lookback_days * 86_400}"
    )
    query = f"""
    WITH normalized_ratings AS (
        SELECT
            userId,
            movieId,
            rating,
            CAST({timestamp_expression} AS BIGINT) AS timestamp
        FROM {quoted_table}
    )
    SELECT userId, movieId, rating, timestamp
    FROM normalized_ratings
    WHERE timestamp >= {cutoff_expression}
    """

    # --- Execute Hive SQL query and return results as a pandas DataFrame ---
    df_pandas = _query_hive(
        query=query,
        dataset_table=dataset_table,
        lookback_days=lookback_days,
        spark_master_url=spark_master_url,
    )

    logger.info("Returning pandas DataFrame: %d rows", len(df_pandas))
    return df_pandas


def _query_hive(
    query: str,
    dataset_table: str,
    lookback_days: int,
    spark_master_url: str,
) -> pd.DataFrame:
    """Read the canonical ratings fields from a configured Hive table."""
    if not _HIVE_IDENTIFIER_PATTERN.fullmatch(dataset_table):
        raise ValueError(
            "dataset_table must be an unquoted table name optionally qualified with one database."
        )
    if lookback_days < 0:
        raise ValueError("lookback_days must be greater than or equal to zero.")

    _ensure_spark_identity()

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(f"{CFG_WORKFLOW_NAME}_ingest")
        .master(spark_master_url)
        .config("spark.sql.catalogImplementation", "hive")
        .enableHiveSupport()
        .getOrCreate()
    )
    try:
        return spark.sql(query).toPandas()
    finally:
        spark.stop()


def _ensure_spark_identity() -> None:
    """Set a fallback Unix identity before Spark starts Hadoop login."""
    for variable_name in ("USER", "LOGNAME", "HADOOP_USER_NAME", "SPARK_USER"):
        os.environ.setdefault(variable_name, _SPARK_IDENTITY)
    for variable_name in ("JAVA_TOOL_OPTIONS", "HADOOP_OPTS"):
        current_value = os.environ.get(variable_name, "")
        if _JAVA_USER_NAME_OPTION not in current_value.split():
            os.environ[variable_name] = (
                f"{current_value} {_JAVA_USER_NAME_OPTION}".strip()
            )


# --- Ingest Logs Step --------------------------------------------------------------------


@step(enable_cache=False)
def ingest_logs(
    model_name: str = CFG_MODEL_NAME,
    model_stage: ModelStages = ModelStages.STAGING,
    logs_path: str = "s3://zenml-predictions/logs",
    lookback_days: int = 7,
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
        chunk_size: Number of rows to materialize per DataFrame chunk.

    Returns:
        DataFrame with inference log records. Returns empty DataFrame if no logs.
    """

    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    records: Iterator[dict[str, object]]

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
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=access_key_id,
            seaweedfs_secret_access_key=secret_access_key,
            model_name=model_name,
            model_version=model_version_name,
        )
    else:
        records = _load_filesystem_logs(
            logs_path, cutoff, model_name=model_name, model_version=model_version_name
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
    model_name: str = CFG_MODEL_NAME,
    model_version: str = "unknown",
) -> Iterator[dict[str, object]]:
    """Yield flattened JSONL log rows from local filesystem directory."""

    import json

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
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass


# --- Ingest Batch Recommendations Step --------------------------------------------------


@step(enable_cache=False)
def ingest_batch_recommendations(
    model_name: str = CFG_MODEL_NAME,
    model_stage: ModelStages = ModelStages.STAGING,
    batch_output_path: str = "s3://zenml-predictions/batch",
    lookback_days: int = 1,
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
        seaweedfs_s3_internal_endpoint: SeaweedFS internal S3 endpoint (local only).
        zenml_local_s3_secret_name: ZenML secret with SeaweedFS credentials.

    Returns:
        DataFrame with columns: userId, rating.  Raises ValueError if empty.
    """
    from datetime import UTC, datetime, timedelta

    client = Client()
    version = client.get_model_version(model_name, model_stage)
    model_version_name = str(version.name)

    access_key_id, secret_access_key = resolve_zenml_s3_credentials(
        zenml_local_s3_secret_name
    )

    today = datetime.now(UTC).date()
    date_strings = [
        (today - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(lookback_days + 1)
    ]

    dfs: list[pd.DataFrame] = []
    for date_str in date_strings:
        prefix = f"{batch_output_path}/{model_name}/{date_str}/{model_version_name}-recommendations"
        if prefix.startswith("s3://"):
            dfs.extend(
                _load_s3_batch_parquet(
                    prefix,
                    seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
                    access_key_id=access_key_id,
                    secret_access_key=secret_access_key,
                )
            )
        else:
            dfs.extend(_load_filesystem_batch_parquet(prefix))

    if not dfs:
        raise ValueError(
            f"No batch recommendation shards found at '{batch_output_path}' "
            f"for model '{model_name}' (version={model_version_name}, "
            f"lookback={lookback_days} days). "
            "Run the batch inference pipeline first."
        )

    df = pd.concat(dfs, ignore_index=True)

    # Rename columns to match Evidently reference schema
    df = df.rename(
        columns={
            CFG_RECS_FIELD_NAMES.USER_ID.value: CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_RECS_FIELD_NAMES.REC_ITEM_ID.value: CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            CFG_RECS_FIELD_NAMES.REC_SCORE.value: CFG_DATASET_FIELD_NAMES.RATING.value,
        }
    )

    logger.info(
        "Loaded %d batch recommendation rows from '%s' (%d date(s) scanned)",
        len(df),
        batch_output_path,
        len(date_strings),
    )
    return df


def _load_s3_batch_parquet(
    s3_prefix: str,
    seaweedfs_s3_internal_endpoint: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
) -> list[pd.DataFrame]:
    """Return a list of DataFrames read from Parquet shards under an S3 prefix."""
    s3 = get_s3_client(
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )
    bucket, prefix = parse_s3_uri(s3_prefix)

    storage_options: dict | None = None
    if seaweedfs_s3_internal_endpoint and access_key_id and secret_access_key:
        storage_options = {
            "client_kwargs": {"endpoint_url": seaweedfs_s3_internal_endpoint},
            "key": access_key_id,
            "secret": secret_access_key,
        }

    result: list[pd.DataFrame] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".parquet"):
                continue
            shard_uri = f"s3://{bucket}/{obj['Key']}"
            if storage_options:
                result.append(
                    pd.read_parquet(shard_uri, storage_options=storage_options)
                )
            else:
                result.append(pd.read_parquet(shard_uri))

    return result


def _load_filesystem_batch_parquet(path: str) -> list[pd.DataFrame]:
    """Return a list of DataFrames read from Parquet shards in a local directory."""
    result: list[pd.DataFrame] = []
    shard_dir = Path(path)
    if not shard_dir.exists():
        return result
    for shard in sorted(shard_dir.glob("*.parquet")):
        result.append(pd.read_parquet(shard))
    return result


def _load_s3_logs(
    s3_prefix: str,
    cutoff: datetime,
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
                        yield from _iter_prediction_rows(rec, ts)
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass
