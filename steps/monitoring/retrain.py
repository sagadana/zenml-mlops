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
from typing import Annotated

from zenml import step
from zenml.cli import Pipeline

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def trigger_retraining(
    should_retrain: bool,
    pipeline_module: str,
    pipeline_function: str,
    config_path: str,
) -> Annotated[dict, "retrain_trigger_report"]:
    """
    Trigger a new training pipeline run if should_retrain is True.

    Dynamically imports the pipeline function from ``pipeline_module`` so this
    step remains workflow-agnostic.

    Args:
        should_retrain: If False, this step is a no-op.
        pipeline_module: Dotted module path to import the pipeline from,
            e.g. ``"workflows.matrix_factorization.pipelines.training_pipeline"``.
        pipeline_function: Name of the pipeline function in that module,
            e.g. ``"training_pipeline"``.
        config_path: Pipeline config file path passed to ``with_options``.

    Returns:
        Report dict with trigger status and run ID (if triggered).
    """
    if not should_retrain:
        logger.info("Retrain not needed — skipping.")
        return {"triggered": False, "run_id": None}

    logger.info("Triggering retraining pipeline: %s.%s", pipeline_module, pipeline_function)
    try:
        import importlib

        module = importlib.import_module(pipeline_module)
        training_pipeline: Pipeline = getattr(module, pipeline_function)

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
