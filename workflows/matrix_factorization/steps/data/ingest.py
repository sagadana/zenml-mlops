"""
steps/data_ingestion/ingest.py

ZenML step: ingest_data

Downloads MovieLens dataset (1M or 25M), parses ratings into a pandas DataFrame
and returns it as a ZenML artifact.

Config parameters (from pipeline YAML):
    dataset_size: "1m" | "25m"
"""

from __future__ import annotations

import logging
import os
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import ValidationError
from zenml import step
from zenml.client import Client

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
)
from workflows.matrix_factorization.models import PredictionLog

logger = logging.getLogger(__name__)

_MOVIELENS_URLS = {
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

_RATINGS_FILES = {
    "1m": "ml-1m/ratings.dat",
    "25m": "ml-25m/ratings.csv",
}

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB

# --- Ingest Data Step --------------------------------------------------------------------


@step(enable_cache=True)
def ingest_data(
    dataset_size: str = "1m",
    lookback_days: int = 30,
) -> Annotated[pd.DataFrame, "raw_ratings"]:
    """
    Download and ingest MovieLens ratings into a pandas DataFrame.

    NOTE: This can be adapted to ingest datasets from other sources (e.g., S3, Spark, BigQuery)

    Args:
        dataset_size: "1m" for MovieLens 1M (local dev) or "25m" for 25M (AWS).
        lookback_days: Number of recent days of ratings to return. Since the MovieLens
            dataset is static, timestamps are shifted to the present and the data is
            filtered to the last ``lookback_days``. In production this step would
            fetch recent ratings from a live data source directly.

    Returns:
        pandas DataFrame with columns: userId, movieId, rating, timestamp.
    """
    if dataset_size not in _MOVIELENS_URLS:
        raise ValueError(
            f"Unknown dataset_size: {dataset_size!r}. Choose from {list(_MOVIELENS_URLS)}"
        )

    # Cache raw downloads in ./data/ (gitignored)
    cache_dir = Path(os.environ.get("MOVIELENS_CACHE_DIR", "./data"))
    extract_dir = _download_movielens(dataset_size, cache_dir)
    df_pandas = _parse_ratings(extract_dir, dataset_size)

    # Shift timestamps to the present and filter to the last lookback_days,
    # simulating a live data source that returns only recent ratings.
    # In production, replace this with a query against your ratings database or API.
    df_pandas = _make_dataset_recent(df_pandas, lookback_days)

    logger.info("Returning pandas DataFrame: %d rows", len(df_pandas))
    return df_pandas


def _download_movielens(dataset_size: str, cache_dir: Path) -> Path:
    """Download and extract MovieLens zip if not already cached."""
    import ssl
    import urllib.request

    url = _MOVIELENS_URLS[dataset_size]
    zip_path = cache_dir / f"ml-{dataset_size}.zip"
    extract_dir = cache_dir / f"ml-{dataset_size}-extracted"

    if extract_dir.exists():
        logger.info("Using cached MovieLens %s at %s", dataset_size, extract_dir)
        return extract_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MovieLens %s from %s ...", dataset_size, url)

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    logger.warning(
        "SSL certificate verification disabled for download (self-signed cert detected)."
    )

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = min(downloaded / total_size * 100, 100)
                    logger.debug("  %.1f%% (%d / %d bytes)", pct, downloaded, total_size)

    logger.info("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    zip_path.unlink()  # remove zip to save space
    logger.info("Extracted to %s", extract_dir)
    return extract_dir


def _parse_ratings(extract_dir: Path, dataset_size: str) -> pd.DataFrame:
    """Parse ratings file into a pandas DataFrame with canonical column names."""
    ratings_rel = _RATINGS_FILES[dataset_size]
    ratings_path = extract_dir / ratings_rel

    dtypes = {
        CFG_DATASET_FIELD_NAMES.USER_ID.value: CFG_DATASET_FIELD_TYPES.USER_ID.value,
        CFG_DATASET_FIELD_NAMES.ITEM_ID.value: CFG_DATASET_FIELD_TYPES.ITEM_ID.value,
        CFG_DATASET_FIELD_NAMES.RATING.value: CFG_DATASET_FIELD_TYPES.RATING.value,
        CFG_DATASET_FIELD_NAMES.TIMESTAMP.value: CFG_DATASET_FIELD_TYPES.TIMESTAMP.value,
    }

    if dataset_size == "1m":
        # Format: UserID::MovieID::Rating::Timestamp
        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=list(dtypes.keys()),
            dtype=dict(dtypes.items()),
        )
    else:
        # Format: userId,movieId,rating,timestamp (CSV with header)
        df = pd.read_csv(
            ratings_path,
            dtype=dict(dtypes.items()),
        )

    logger.info(
        "Parsed %d ratings (%d users, %d items)",
        len(df),
        df[CFG_DATASET_FIELD_NAMES.USER_ID.value].nunique(),
        df[CFG_DATASET_FIELD_NAMES.ITEM_ID.value].nunique(),
    )
    return df


def _make_dataset_recent(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """
    Shift dataset timestamps to make it appear recent, for testing purposes.
    Then filter to only include ratings within the last `lookback_days`.
    """
    now = datetime.now(UTC)
    max_timestamp = df[CFG_DATASET_FIELD_NAMES.TIMESTAMP.value].max()
    shift_seconds = int((now - datetime.fromtimestamp(max_timestamp, UTC)).total_seconds())
    df[CFG_DATASET_FIELD_NAMES.TIMESTAMP.value] += shift_seconds

    cutoff_timestamp = int((now - timedelta(days=lookback_days)).timestamp())
    recent_df = df[df[CFG_DATASET_FIELD_NAMES.TIMESTAMP.value] >= cutoff_timestamp]

    logger.info(
        "Shifted timestamps by %d seconds. Filtered to %d recent ratings (last %d days).",
        shift_seconds,
        len(recent_df),
        lookback_days,
    )
    return recent_df.reset_index(drop=True)


# --- Ingest Logs Step --------------------------------------------------------------------


@step(enable_cache=False)
def ingest_logs(
    model_name: str = CFG_MODEL_NAME,
    model_stage: str = "staging",
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

    access_key_id, secret_access_key = resolve_zenml_s3_credentials(zenml_local_s3_secret_name)

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
        CFG_DATASET_FIELD_NAMES.USER_ID.value: np.dtype(CFG_DATASET_FIELD_TYPES.USER_ID.value),
        CFG_DATASET_FIELD_NAMES.ITEM_ID.value: np.dtype(CFG_DATASET_FIELD_TYPES.ITEM_ID.value),
        CFG_DATASET_FIELD_NAMES.RATING.value: np.dtype(CFG_DATASET_FIELD_TYPES.RATING.value),
        CFG_DATASET_FIELD_NAMES.TIMESTAMP.value: np.dtype(CFG_DATASET_FIELD_TYPES.TIMESTAMP.value),
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
        df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dtype_map.items()})

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


def _iter_prediction_rows(rec: PredictionLog, ts: datetime) -> Iterator[dict[str, object]]:
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
                        and (not rec.model_version or rec.model_version == model_version)
                    ):
                        yield from _iter_prediction_rows(rec, ts)
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass


# --- Ingest Batch Recommendations Step --------------------------------------------------


@step(enable_cache=False)
def ingest_batch_recommendations(
    model_name: str = CFG_MODEL_NAME,
    model_stage: str = "staging",
    batch_output_path: str = "s3://zenml-predictions/batch",
    lookback_days: int = 1,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> Annotated[pd.DataFrame, "batch_recommendations"]:
    """
    Load recent batch recommendation Parquet shards for drift monitoring.

    Reads shards written by collect_batch_recommendations at:
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

    access_key_id, secret_access_key = resolve_zenml_s3_credentials(zenml_local_s3_secret_name)

    today = datetime.now(UTC).date()
    date_strings = [
        (today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(lookback_days + 1)
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
                result.append(pd.read_parquet(shard_uri, storage_options=storage_options))
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
                        and (not rec.model_version or rec.model_version == model_version)
                    ):
                        yield from _iter_prediction_rows(rec, ts)
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass
