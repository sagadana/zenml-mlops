"""
steps/serving/trigger.py

ZenML step: trigger_serving_pipeline

Triggers a deployment pipeline run after a successful training registration.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import step

from helpers.pipeline_trigger import trigger_pipeline_run
from workflows.matrix_factorization.configs import CFG_DEPLOYMENT_PIPELINE_NAME

logger = logging.getLogger(__name__)


def _run_serving_pipeline_async(run_name: str, configs: dict) -> None:
    """Launch the deployment pipeline in a dedicated background thread."""
    from workflows.matrix_factorization.pipelines.deployment_pipeline import (
        deployment_pipeline,
    )

    try:
        run = deployment_pipeline.with_options(run_name=run_name, **configs)()
        if not run:
            raise RuntimeError(f"Failed to trigger deployment pipeline run '{run_name}'")
        logger.info("Deployment pipeline started asynchronously: run_id=%s", run.id)
    except Exception:
        logger.exception("Asynchronous deployment pipeline launch failed for run_name=%s", run_name)


@step(enable_cache=False)
def trigger_serving_pipeline(
    pipeline_name: str = CFG_DEPLOYMENT_PIPELINE_NAME,
    config_path: str = "",
) -> Annotated[dict, "serving_trigger_report"]:
    """
    Trigger a deployment pipeline run.

    Args:
        pipeline_name: Name of the pipeline to trigger.
        config_path: Path to pipeline config file.

    Returns:
        Report dict with trigger status and run ID.
    """
    if not pipeline_name or not config_path:
        raise ValueError("pipeline_name, config_path cannot be empty.")

    logger.info("Triggering deployment pipeline: %s", pipeline_name)
    run_id = trigger_pipeline_run(
        pipeline_name=pipeline_name,
        config_path=config_path,
    )
    logger.info("Deployment pipeline started: run_id=%s", run_id)
    return {"triggered": True, "run_id": run_id}
