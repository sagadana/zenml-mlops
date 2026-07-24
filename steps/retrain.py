"""
steps/monitoring/retrain.py

ZenML step: trigger_retraining

Conditionally triggers a new training pipeline run.
Only executes when should_retrain=True from check_retrain.

Generic across workflows — the target pipeline is specified by its dotted
module path and function name so no workflow-specific imports are needed.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Annotated

from zenml import step
from zenml.client import Client

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def check_retrain(
    report_json: str,
    model_name: str = "",
    model_stage: str = "staging",
    drifted_column_share_threshold: float = 0.5,
    missing_values_share_threshold: float = 0.1,
    max_age_days: int = 30,
) -> Annotated[bool, "should_retrain"]:
    """
    Determine whether the model should be retrained.

    Retrain if ANY of these conditions are true:
      - DatasetDriftMetric.share_of_drifted_columns > drifted_column_share_threshold
      - DatasetMissingValuesMetric.current.share_of_missing_values > missing_values_share_threshold
      - Time since last successful training > max_age_days

    Args:
        report_json: JSON string from the data-drift Evidently report
            (DataQualityPreset + DataDriftPreset).
        model_name: Registered ZenML model name to check age for.
        model_stage: ZenML model stage to resolve the current version.
        drifted_column_share_threshold: Max allowed share of drifted columns before triggering (0.0–1.0).
        missing_values_share_threshold: Max allowed share of missing values in current data (0.0–1.0).
        max_age_days: Max days since last training before scheduled retrain.

    Returns:
        should_retrain: True if retraining is recommended.
    """
    if not model_name or not report_json:
        raise ValueError("model_name and report_json cannot be empty.")

    reasons: list[str] = []

    def _check_report(raw_json: str, source_label: str) -> None:
        report = json.loads(raw_json)
        metrics_by_name: dict = {
            entry["metric"]: entry["result"] for entry in report.get("metrics", [])
        }

        drift_result = metrics_by_name.get("DatasetDriftMetric", {})
        share_drifted = float(drift_result.get("share_of_drifted_columns", 0.0))
        if share_drifted > drifted_column_share_threshold:
            reasons.append(
                f"[{source_label}] Data drift: {share_drifted:.0%} of columns drifted "
                f"(threshold={drifted_column_share_threshold:.0%})"
            )

        missing_result = metrics_by_name.get("DatasetMissingValuesMetric", {})
        share_missing = float(missing_result.get("current", {}).get("share_of_missing_values", 0.0))
        if share_missing > missing_values_share_threshold:
            reasons.append(
                f"[{source_label}] Missing values: {share_missing:.0%} of current data is missing "
                f"(threshold={missing_values_share_threshold:.0%})"
            )

    _check_report(report_json, "data-drift")

    # Model age check (runs once regardless of number of reports)
    last_training_date = _get_last_training_date(model_name, model_stage)
    if last_training_date is None:
        reasons.append("No previous model found — initial training required")
    else:
        age_days = (datetime.now(UTC) - last_training_date).days
        if age_days > max_age_days:
            reasons.append(f"Model age: {age_days} days (max_age_days={max_age_days})")

    should_retrain = len(reasons) > 0
    if should_retrain:
        logger.info("Retrain triggered. Reasons:\n%s", "\n".join(f"  - {r}" for r in reasons))
    else:
        logger.info("No retrain needed. All drift checks passed, model is recent.")

    return should_retrain


def _get_last_training_date(model_name: str, model_stage: str = "staging") -> datetime | None:
    """Retrieve the creation date of the latest production model version."""
    try:
        client = Client()
        latest_version = client.get_model_version(model_name, model_stage)
        created_at = latest_version.created
        return created_at.replace(tzinfo=UTC) if created_at.tzinfo is None else created_at
    except Exception as exc:
        logger.warning("Could not retrieve last training date for '%s': %s", model_name, exc)
        return None
