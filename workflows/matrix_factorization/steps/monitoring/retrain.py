"""
steps/monitoring/retrain.py

ZenML step: trigger_retraining

Conditionally triggers a new training pipeline run.
Only executes when should_retrain=True from check_retrain_trigger.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import step

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def trigger_retraining(
    should_retrain: bool,
    config_path: str = "workflows/matrix_factorization/configs/aws.yaml",
) -> Annotated[dict, "retrain_trigger_report"]:
    """
    Trigger a new training pipeline run if should_retrain is True.

    In production, this fires a new ZenML pipeline run with cache disabled
    to ensure fresh data is processed and a new model version is created.

    Args:
        should_retrain: If False, this step is a no-op.
        config_path: Pipeline config to use for the retrain run.

    Returns:
        Report dict with trigger status and run ID (if triggered).
    """
    if not should_retrain:
        logger.info("Retrain not needed — skipping.")
        return {"triggered": False, "run_id": None}

    logger.info("Triggering retraining pipeline run...")
    try:
        from workflows.matrix_factorization.pipelines.training_pipeline import training_pipeline

        run = training_pipeline.with_options(
            config_path=config_path,
            enable_cache=False,
        )()

        if run is None:
            logger.warning("Retraining pipeline did not start successfully.")
            return {"triggered": False, "run_id": None, "error": "Pipeline run returned None"}

        run_id = str(run.id) if hasattr(run, "id") else "unknown"
        logger.info("Retraining pipeline started: run_id=%s", run_id)

        return {"triggered": True, "run_id": run_id}
    except Exception as exc:
        logger.error("Failed to trigger retraining: %s", exc)
        return {"triggered": False, "run_id": None, "error": str(exc)}
