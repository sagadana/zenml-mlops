"""
steps/monitoring/drift_detection.py

ZenML step: run_drift_detection

Compares recent inference traffic against a training reference dataset using
Evidently AI. Generic across workflows — callers supply pre-prepared
pandas DataFrames and specify which columns to monitor.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from zenml import step
from zenml.types import HTMLString, JSONString

from helpers.s3_client import (
    get_s3_client,
    parse_s3_uri,
    resolve_zenml_s3_credentials,
    s3_put_object_bytes,
)

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def run_drift_detection(
    reference_dataset: pd.DataFrame,
    current_dataset: pd.DataFrame,
    monitoring_output_path: str = "s3://zenml-predictions/monitoring",
    sample_seed: int = 42,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> tuple[
    Annotated[JSONString, "drift_report"],
    Annotated[HTMLString, "drift_report_html"],
]:
    """
    Run Evidently data drift detection comparing inference logs vs. a reference dataset.

    The step expects both DataFrames to share the same schema and runs
    ``DataDriftPreset`` over detected text and numerical columns.

    Args:
        reference_dataset: Training reference DataFrame (pandas, already sampled/filtered).
        current_dataset: Recent inference logs DataFrame (pandas, already sampled/filtered).
        monitoring_output_path: S3 path (or local path) for Evidently HTML/JSON output.
        sample_seed: Random seed for sampling reference data to match current size.

    Returns:
        drift_report dict with keys: dataset_drift (bool), n_drifted_features (int),
        drift_share (float), report_path (str).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    if current_dataset.empty:
        logger.warning("Empty inference logs — skipping drift detection")
        return JSONString(
            {
                "dataset_drift": False,
                "n_drifted_features": 0,
                "drift_share": 0.0,
                "report_path": "",
                "skipped": True,
            }
        ), HTMLString("<p>No inference logs to analyze.</p>")

    missing_columns = sorted(set(reference_dataset.columns) - set(current_dataset.columns))
    if missing_columns:
        raise ValueError(
            "Current dataset is missing baseline columns: " + ", ".join(missing_columns)
        )

    text_columns = reference_dataset.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numerical_columns = reference_dataset.select_dtypes(include=["number"]).columns.tolist()

    if not text_columns and not numerical_columns:
        raise ValueError("No text or numerical columns found for drift detection")

    drift_columns = [
        column
        for column in reference_dataset.columns
        if column in set(text_columns + numerical_columns)
    ]

    data_definition = DataDefinition(
        text_columns=text_columns,
        numerical_columns=numerical_columns,
    )

    current = current_dataset[reference_dataset.columns].copy()

    # Create Evidently Dataset objects for reference data
    # - Sample reference data to match current size for balanced comparison
    ref_sample = reference_dataset[drift_columns].sample(
        n=min(len(current) * 2, len(reference_dataset)), random_state=sample_seed
    )
    reference_data = Dataset.from_pandas(ref_sample, data_definition=data_definition)

    # Create Evidently Dataset objects for current data
    current_data = Dataset.from_pandas(current[drift_columns], data_definition=data_definition)

    # ------------------------------------
    # Run Evidently data drift detection
    # and generate HTML/JSON reports
    # ------------------------------------

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    my_eval = report.run(current_data=current_data, reference_data=reference_data)

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    html_path = f"{monitoring_output_path}/{date_str}/drift_report.html"
    json_path = f"{monitoring_output_path}/{date_str}/drift_report.json"

    report_html = my_eval.get_html_str(True)
    access_key_id, secret_access_key = resolve_zenml_s3_credentials(zenml_local_s3_secret_name)

    _write_text(
        html_path,
        report_html,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )
    report_dict = my_eval.dict()
    _write_text(
        json_path,
        json.dumps(report_dict, default=str),
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )

    drift_metrics = _extract_drift_summary(report_dict)
    drift_metrics["report_path"] = html_path

    logger.info(
        "Drift detection: dataset_drift=%s, n_drifted=%d, drift_share=%.4f",
        drift_metrics["dataset_drift"],
        drift_metrics["n_drifted_features"],
        drift_metrics["drift_share"],
    )
    return JSONString(drift_metrics), HTMLString(report_html)


def _extract_drift_summary(report_dict: dict) -> dict:
    """Extract key drift metrics from Evidently v0.7+ report dict.

    The modern Evidently API stores each metric as::

        {"id": ..., "metric_name": "DriftedColumnsCount(drift_share=0.5)",
         "config": ..., "value": {"count": 2.0, "share": 0.4}}

    ``DriftedColumnsCount.value.count`` is the number of drifted columns;
    ``value.share`` is the fraction; ``drift_share`` in the metric name is
    the dataset-level threshold above which the dataset is considered drifted.
    """
    import re

    n_drifted = 0
    drift_share = 0.0
    dataset_drift = False

    for metric in report_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        if "DriftedColumnsCount" in metric_name:
            value = metric.get("value", {})
            if isinstance(value, dict):
                n_drifted = int(value.get("count", 0))
                drift_share = float(value.get("share", 0.0))
                # Parse the dataset-level drift threshold from the metric name,
                # e.g. "DriftedColumnsCount(drift_share=0.5)"
                match = re.search(r"drift_share=([\d.]+)", metric_name)
                threshold = float(match.group(1)) if match else 0.5
                dataset_drift = drift_share > threshold
            break

    return {
        "dataset_drift": dataset_drift,
        "n_drifted_features": n_drifted,
        "drift_share": drift_share,
    }


def _write_text(
    path: str,
    content: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> None:
    """Write text content to local path or S3."""
    if path.startswith("s3://"):
        bucket, key = parse_s3_uri(path)
        s3 = get_s3_client(
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
        s3_put_object_bytes(s3, bucket=bucket, key=key, body=content.encode())
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
