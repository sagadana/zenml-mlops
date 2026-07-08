"""
steps/monitoring/drift_detection.py

ZenML step: run_drift_detection

Compares recent inference traffic (userId distribution, activity patterns)
against the training reference dataset using Evidently AI.
Generates an HTML report and a JSON summary.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Annotated

import dask.dataframe as dd
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def run_drift_detection(
    inference_logs: pd.DataFrame,
    raw_ratings: dd.DataFrame,  # type: ignore[arg-type]
    monitoring_output_path: str = "s3://aips-zenml-predictions/monitoring",
) -> Annotated[dict, "drift_report"]:
    """
    Run Evidently data drift detection comparing inference logs vs. training data.

    Args:
        inference_logs: Recent inference log DataFrame from collect_inference_logs.
        raw_ratings: Training reference DataFrame from ingest_data.
        monitoring_output_path: S3 path (or local path) for Evidently HTML/JSON output.

    Returns:
        drift_report dict with keys: dataset_drift (bool), n_drifted_features (int),
        drift_share (float), report_path (str).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    if inference_logs.empty:
        logger.warning("Empty inference logs — skipping drift detection")
        return {
            "dataset_drift": False,
            "n_drifted_features": 0,
            "drift_share": 0.0,
            "report_path": "",
            "skipped": True,
        }

    # Build reference dataset from training data (sample to match inference size)
    ref_pd = raw_ratings[["userId", "rating"]].compute()
    ref_sample = ref_pd.sample(n=min(len(inference_logs) * 2, len(ref_pd)), random_state=42)

    # Build current dataset from inference logs
    current = inference_logs[["user_id"]].rename(columns={"user_id": "userId"}).copy()
    if "latency_ms" in inference_logs.columns:
        current["latency_ms"] = inference_logs["latency_ms"]

    # Define schema: userId is a numerical feature
    data_definition = DataDefinition(
        numerical_columns=["userId"],
    )

    reference_dataset = Dataset.from_pandas(
        ref_sample[["userId"]],
        data_definition=data_definition,
    )
    current_dataset = Dataset.from_pandas(
        current[["userId"]],
        data_definition=data_definition,
    )

    # Run Evidently report with DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_dataset, current_dataset)

    # Write HTML report
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    html_path = f"{monitoring_output_path}/{date_str}/drift_report.html"
    json_path = f"{monitoring_output_path}/{date_str}/drift_report.json"

    _write_text(html_path, my_eval.get_html_str(True))
    report_dict = my_eval.dict()
    _write_text(json_path, json.dumps(report_dict, default=str))

    # Extract summary metrics from Evidently output
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
    """Extract key drift metrics from Evidently report JSON."""
    dataset_drift = False
    n_drifted = 0
    drift_share = 0.0

    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        if "dataset_drift" in result:
            dataset_drift = bool(result["dataset_drift"])
        if "number_of_drifted_columns" in result:
            n_drifted = int(result["number_of_drifted_columns"])
        if "share_of_drifted_columns" in result:
            drift_share = float(result["share_of_drifted_columns"])

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
