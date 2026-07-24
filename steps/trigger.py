"""
steps/monitoring/retrain.py

ZenML step: trigger_retraining

Conditionally triggers a new training pipeline run.
Only executes when should_retrain=True from check_retrain.

Generic across workflows — the target pipeline is specified by its dotted
module path and function name so no workflow-specific imports are needed.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import step

from helpers.pipeline import trigger_pipeline_run

logger = logging.getLogger(__name__)


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
