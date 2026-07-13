"""
helpers/pipeline_trigger.py

Shared helper for triggering ZenML pipeline snapshots by pipeline name.
"""

from __future__ import annotations

import logging

from zenml.client import Client

logger = logging.getLogger(__name__)


def trigger_pipeline_run(
    pipeline_name: str,
    config_path: str,
) -> str:
    """
    Trigger a ZenML pipeline and return the started run ID.

    Args:
        pipeline_name: Name of the target pipeline.
        config_path: Pipeline config path (used by the SDK path only).
        server_url: ZenML server base URL (required for REST API fallback).
        service_account_name: Optional service account name (for logging context).
        api_key_name: Optional API key name for authentication. Default is "default".

    Returns:
        Started pipeline run ID, or ``"unknown"`` when unavailable.
    """

    if not pipeline_name or not config_path:
        raise ValueError("pipeline_name, config_path cannot be empty.")

    client = Client()
    active_stack = client.active_stack
    stack_id = active_stack.id

    logger.info(
        "Triggering pipeline '%s' on stack '%s' with config '%s'",
        pipeline_name,
        stack_id,
        config_path,
    )

    try:
        run = client.trigger_pipeline(
            stack_name_or_id=stack_id,
            pipeline_name_or_id=pipeline_name,
            config_path=config_path,
            synchronous=False,
        )
        return str(run.id) if hasattr(run, "id") else "unknown"
    except Exception as exc:
        raise RuntimeError(f"Failed to trigger pipeline '{pipeline_name}'") from exc
