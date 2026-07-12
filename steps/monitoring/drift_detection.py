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
from typing import Annotated

import pandas as pd
from zenml import step

from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_RECS_FIELD_NAMES,
)

logger = logging.getLogger(__name__)

# Data drift detection columns: map inference log (current) column names to baseline dataset (reference) column names
DATASET_TEXT_DRIFT_COLUMNS = {
    CFG_RECS_FIELD_NAMES.USER_ID.value: CFG_DATASET_FIELD_NAMES.USER_ID.value,
}
DATASET_INT_DRIFT_COLUMNS = {
    CFG_RECS_FIELD_NAMES.REC_SCORE.value: CFG_DATASET_FIELD_NAMES.RATING.value,
}
DATASET_DRIFT_COLUMN_MAP = {**DATASET_TEXT_DRIFT_COLUMNS, **DATASET_INT_DRIFT_COLUMNS}


@step(enable_cache=False)
def run_drift_detection(
    baseline_dataset: pd.DataFrame,
    inference_logs: pd.DataFrame,
    reference_id_column: CFG_DATASET_FIELD_NAMES = CFG_DATASET_FIELD_NAMES.USER_ID,
    current_id_column: CFG_RECS_FIELD_NAMES = CFG_RECS_FIELD_NAMES.USER_ID,
    monitoring_output_path: str = "s3://zenml-predictions/monitoring",
) -> Annotated[dict, "drift_report"]:
    """
    Run Evidently data drift detection comparing inference logs vs. a reference dataset.

    The step aligns column names between the two DataFrames using
    ``reference_id_column`` and ``current_id_column``, then runs
    ``DataDriftPreset`` over the specified ``DATASET_TEXT_DRIFT_COLUMNS``.

    Args:
        baseline_dataset: Training reference DataFrame (pandas, already sampled/filtered).
        inference_logs: Recent inference log DataFrame (from collect_inference_logs).
        reference_id_column: Column name for the entity ID in ``baseline_dataset``.
        current_id_column: Column name for the entity ID in ``inference_logs``.
        monitoring_output_path: S3 path (or local path) for Evidently HTML/JSON output.

    Returns:
        drift_report dict with keys: dataset_drift (bool), n_drifted_features (int),
        drift_share (float), report_path (str).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    if inference_logs.empty:
        logger.warning("Empty inference logs — skipping drift detection")
        return {
            "dataset_drift": False,
            "n_drifted_features": 0,
            "drift_share": 0.0,
            "report_path": "",
            "skipped": True,
        }

    data_definition = DataDefinition(
        text_columns=list(DATASET_TEXT_DRIFT_COLUMNS.values()),
        numerical_columns=list(DATASET_INT_DRIFT_COLUMNS.values()),
    )

    # Align current data to reference column names
    current = (
        inference_logs[[current_id_column]]
        .rename(columns={current_id_column: reference_id_column, **DATASET_DRIFT_COLUMN_MAP})
        .copy()
    )

    drift_columns = list(DATASET_DRIFT_COLUMN_MAP.values())

    # Create Evidently Dataset objects for reference data
    # - Sample reference data to match current size for balanced comparison
    ref_sample = baseline_dataset[drift_columns].sample(
        n=min(len(current) * 2, len(baseline_dataset)), random_state=42
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

    _write_text(html_path, my_eval.get_html_str(True))
    report_dict = my_eval.dict()
    _write_text(json_path, json.dumps(report_dict, default=str))

    drift_metrics = _extract_drift_summary(report_dict)
    drift_metrics["report_path"] = html_path

    logger.info(
        "Drift detection: dataset_drift=%s, n_drifted=%d, drift_share=%.4f",
        drift_metrics["dataset_drift"],
        drift_metrics["n_drifted_features"],
        drift_metrics["drift_share"],
    )
    return drift_metrics


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


def _write_text(path: str, content: str) -> None:
    """Write text content to local path or S3."""
    if path.startswith("s3://"):
        import boto3

        parts = path.replace("s3://", "").split("/", 1)
        boto3.client("s3").put_object(Bucket=parts[0], Key=parts[1], Body=content.encode())
    else:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
