"""
steps/monitoring/collect_logs.py

ZenML step: collect_inference_logs

Collects recent inference request logs from S3 or a local directory.
Generic across workflows — expects JSONL files written by any serving app.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def collect_inference_logs(
    logs_path: str = "s3://aips-zenml-predictions/logs",
    lookback_days: int = 7,
) -> Annotated[pd.DataFrame, "inference_logs"]:
    """
    Load recent inference request logs into a DataFrame.

    Expected log format (JSON lines written by any serving app):
        {"timestamp": "...", "user_id": int, "top_k": int,
         "latency_ms": float, "items_returned": int}

    Args:
        logs_path: S3 prefix (or local dir) containing JSONL log files.
        lookback_days: Number of days of logs to load.

    Returns:
        DataFrame with inference log records. Returns empty DataFrame if no logs.
    """
    import json
    from pathlib import Path

    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    records = []

    if logs_path.startswith("s3://"):
        records = _load_s3_logs(logs_path, cutoff)
    else:
        log_dir = Path(logs_path)
        if log_dir.exists():
            for log_file in sorted(log_dir.glob("*.jsonl")):
                with open(log_file) as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            ts = datetime.fromisoformat(rec.get("timestamp", ""))
                            if ts >= cutoff:
                                records.append(rec)
                        except (json.JSONDecodeError, ValueError):
                            pass

    if not records:
        logger.warning("No inference logs found at %s (lookback=%d days)", logs_path, lookback_days)
        return pd.DataFrame(
            columns=["timestamp", "user_id", "top_k", "latency_ms", "items_returned"]
        )

    df = pd.DataFrame(records)
    logger.info("Loaded %d inference log records from %s", len(df), logs_path)
    return df


def _load_s3_logs(s3_prefix: str, cutoff: datetime) -> list[dict]:
    """Load JSONL log records from S3 prefix."""
    import json

    import boto3

    s3 = boto3.client("s3")
    bucket = s3_prefix.replace("s3://", "").split("/")[0]
    prefix = "/".join(s3_prefix.replace("s3://", "").split("/")[1:])

    records = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode()
            for line in body.splitlines():
                try:
                    rec = json.loads(line.strip())
                    ts = datetime.fromisoformat(rec.get("timestamp", ""))
                    if ts >= cutoff:
                        records.append(rec)
                except (json.JSONDecodeError, ValueError):
                    pass
    return records
