"""
steps/monitoring/retrain.py

ZenML step: trigger_retraining

Conditionally triggers a new training pipeline run.
Only executes when should_retrain=True from check_retrain_trigger.

Generic across workflows — the target pipeline is specified by its dotted
module path and function name so no workflow-specific imports are needed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from zenml import step
from zenml.client import Client

from helpers.pipeline_trigger import trigger_pipeline_run

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def check_retrain_trigger(
    drift_report: dict,
    model_name: str = "",
    drift_threshold_n_features: int = 2,
    max_age_days: int = 30,
) -> Annotated[bool, "should_retrain"]:
    """
    Determine whether the model should be retrained.

    Retrain if ANY of these conditions are true:
      - drift_report.n_drifted_features > drift_threshold_n_features
      - Time since last successful training > max_age_days

    Args:
        drift_report: Output from run_drift_detection step.
        model_name: Registered ZenML model name to check age for.
        drift_threshold_n_features: Max allowed drifted features before triggering.
        max_age_days: Max days since last training before scheduled retrain.

    Returns:
        should_retrain: True if retraining is recommended.
    """

    if not model_name or len(drift_report) == 0:
        raise ValueError("model_name and drift_report cannot be empty.")

    reasons: list[str] = []

    # Check 1: Data drift
    n_drifted = int(drift_report.get("n_drifted_features", 0))
    if n_drifted > drift_threshold_n_features:
        reasons.append(
            f"Data drift: {n_drifted} features drifted (threshold={drift_threshold_n_features})"
        )

    # Check 2: Model age
    last_training_date = _get_last_training_date(model_name)
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
        logger.info("No retrain needed. Drift=%d features, model is recent.", n_drifted)

    return should_retrain


@step(enable_cache=False)
def trigger_retraining(
    pipeline_name: str = "",
    config_path: str = "",
    should_retrain: bool = False,
) -> Annotated[dict, "retrain_trigger_report"]:
    """
    Trigger a new training pipeline run if should_retrain is True.

    Args:
        pipeline_name: Name of the pipeline to trigger, e.g. ``"matrix_factorization_training"``.
        config_path: Path to pipeline config file.
        server_url: ZenML server URL for REST API calls.
        service_account_name: Optional service account name for API authentication.
        should_retrain: If False, this step is a no-op.
        api_key_name: Optional API key name for authentication. Default is "default".

    Returns:
        Report dict with trigger status and run ID (if triggered).
    """
    if not pipeline_name or not config_path:
        raise ValueError("pipeline_name, config_path cannot be empty.")

    if not should_retrain:
        logger.info("Retrain not needed — skipping.")
        return {"triggered": False, "run_id": None}

    logger.info("Triggering retraining pipeline: %s", pipeline_name)
    run_id = trigger_pipeline_run(
        pipeline_name=pipeline_name,
        config_path=config_path,
    )
    logger.info("Retraining pipeline started: run_id=%s", run_id)
    return {"triggered": True, "run_id": run_id}


def _get_last_training_date(model_name: str) -> datetime | None:
    """Retrieve the creation date of the latest production model version."""
    try:
        client = Client()
        latest_version = client.get_model_version(model_name, "production")
        created_at = latest_version.created
        return created_at.replace(tzinfo=UTC) if created_at.tzinfo is None else created_at
    except Exception as exc:
        logger.warning("Could not retrieve last training date for '%s': %s", model_name, exc)
        return None
