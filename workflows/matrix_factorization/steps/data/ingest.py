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
) -> Annotated[pd.DataFrame, "raw_ratings"]:
    """
    Download and ingest MovieLens ratings into a pandas DataFrame.

    NOTE: This can be adapted to ingest datasets from other sources (e.g., S3, Spark, BigQuery)

    Args:
        dataset_size: "1m" for MovieLens 1M (local dev) or "25m" for 25M (AWS).

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


# --- Ingest Logs Step --------------------------------------------------------------------


@step(enable_cache=False)
def ingest_logs(
    logs_path: str = "s3://zenml-predictions/logs",
    lookback_days: int = 7,
    chunk_size: int = 1000,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> Annotated[pd.DataFrame, "inference_logs"]:
    """
    Load recent inference request logs into a DataFrame.

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

    access_key_id, secret_access_key = resolve_zenml_s3_credentials(zenml_local_s3_secret_name)

    if logs_path.startswith("s3://"):
        records = _load_s3_logs(
            logs_path,
            cutoff,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=access_key_id,
            seaweedfs_secret_access_key=secret_access_key,
        )
    else:
        records = _load_filesystem_logs(logs_path, cutoff)

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
        logger.warning("No inference logs found at %s (lookback=%d days)", logs_path, lookback_days)
        return df

    logger.info("Loaded %d inference log records from %s", len(df), logs_path)
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


def _load_filesystem_logs(logs_path: str, cutoff: datetime) -> Iterator[dict[str, object]]:
    """Yield flattened JSONL log rows from local filesystem directory."""

    import json

    log_dir = Path(logs_path)
    for log_file in sorted(log_dir.glob(f"*{CFG_INFERENCE_LOGS_EXT}")):
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = PredictionLog.model_validate_json(line.strip())
                    ts = datetime.fromisoformat(rec.timestamp)
                    if ts >= cutoff:
                        yield from _iter_prediction_rows(rec, ts)
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass


def _load_s3_logs(
    s3_prefix: str,
    cutoff: datetime,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
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
                    if ts >= cutoff:
                        yield from _iter_prediction_rows(rec, ts)
                except (json.JSONDecodeError, ValueError, ValidationError):
                    pass
