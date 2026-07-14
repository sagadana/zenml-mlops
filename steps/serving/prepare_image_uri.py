"""
steps/serving/prepare_image_uri.py

ZenML step: prepare_serving_uris

Resolves the model version, constructs the full serving image URI, and
returns the model artifact URI — without performing any Docker operations.

Keeping URI preparation separate from the build step allows ZenML to cache
the (expensive) build step whenever the image URI (and therefore the model
version) has not changed.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import step
from zenml.client import Client

logger = logging.getLogger(__name__)


@step
def prepare_serving_uris(
    model_name: str = "",
    model_artifact_name: str = "",
    workflow_name: str = "",
    image_registry_uri: str = "",
    model_stage: str = "staging",
) -> tuple[Annotated[str, "serving_image_uri"], Annotated[str, "model_artifact_uri"]]:
    """
    Resolve the current model version and construct the serving image URI.

    Args:
        model_name: Registered ZenML model name.
        model_artifact_name: Name of the model artifact within the model version.
        workflow_name: Workflow directory name; used as the image repository path segment.
        image_registry_uri: Container registry base URI.
            Local example : "localhost:5001"
            AWS example   : "123456789.dkr.ecr.us-east-1.amazonaws.com"
        model_stage: ZenML model stage to look up (e.g. "staging", "production").

    Returns:
        serving_image_uri: Full image URI including registry host, repository and tag.
        model_artifact_uri: ZenML artifact store URI for the model; passed to the
            build step so Docker can embed the model in the image.
    """
    if not model_name or not model_artifact_name or not workflow_name or not image_registry_uri:
        raise ValueError(
            "model_name, model_artifact_name, workflow_name, and image_registry_uri cannot be empty."
        )

    client = Client()

    model_version = client.get_model_version(model_name, model_stage)
    version_str = str(model_version.model.latest_version_name).replace(" ", "-").lower()

    artifact = model_version.get_artifact(model_artifact_name)
    if artifact is None:
        raise ValueError(f"Model artifact '{model_artifact_name}' not found for {model_name}")

    # e.g. localhost:5001/matrix-factorization/1-0-0
    #   or 123456789.dkr.ecr.us-east-1.amazonaws.com/matrix-factorization/1-0-0
    serving_image_uri = (
        f"{image_registry_uri}/{workflow_name.replace('_', '-').lower()}:{version_str}"
    )
    model_artifact_uri = artifact.uri

    logger.info("Resolved serving image URI: %s", serving_image_uri)
    logger.info("Model artifact URI: %s", model_artifact_uri)

    return serving_image_uri, model_artifact_uri
