"""
steps/monitoring/trigger.py

ZenML step: check_retrain_trigger

Evaluates whether the model should be retrained based on:
  1. Data drift detected by Evidently (n_drifted_features > threshold)
  2. Time elapsed since last training > max_age_days
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from zenml import step
from zenml.client import Client

logger = logging.getLogger(__name__)

_MODEL_NAME = "als_movie_recommender"


@step(enable_cache=False)
def check_retrain_trigger(
    drift_report: dict,
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
        drift_threshold_n_features: Max allowed drifted features before triggering.
        max_age_days: Max days since last training before scheduled retrain.

    Returns:
        should_retrain: True if retraining is recommended.
    """
    reasons: list[str] = []

    # Check 1: Data drift
    n_drifted = int(drift_report.get("n_drifted_features", 0))
    if n_drifted > drift_threshold_n_features:
        reasons.append(
            f"Data drift: {n_drifted} features drifted (threshold={drift_threshold_n_features})"
        )

    # Check 2: Model age
    last_training_date = _get_last_training_date()
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


def _get_last_training_date() -> datetime | None:
    """Retrieve the creation date of the latest production model version."""
    try:
        client = Client()
        latest_version = client.get_model_version(_MODEL_NAME, "production")
        created_at = latest_version.created
        return created_at.replace(tzinfo=UTC) if created_at.tzinfo is None else created_at
    except Exception as exc:
        logger.warning("Could not retrieve last training date: %s", exc)
        return None
