"""
steps/serving/model_artifacts.py

ZenML step: get_model_artifact_uri

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
def get_model_artifact_uri(
    model_name: str = "",
    model_artifact_name: str = "",
    model_artifact_filename="",
    model_stage: str = "staging",
) -> tuple[
    Annotated[str, "model_artifact_uri"],
    Annotated[str, "model_version"],
]:
    """
    Resolve the current model version and construct the serving image URI.

    Args:
        model_name: Registered ZenML model name.
        model_artifact_name: Name of the model artifact within the model version.
        model_artifact_filename: Filename of the model artifact (e.g. "model.pkl").
        model_stage: ZenML model stage to look up (e.g. "staging", "production").

    Returns:
        serving_image_uri: Full image URI including registry host, repository and tag.
        model_artifact_uri: ZenML artifact store URI for the model; passed to the
            build step so Docker can embed the model in the image.
    """
    if not model_name or not model_artifact_name or not model_artifact_filename:
        raise ValueError(
            "model_name, model_artifact_name, and model_artifact_filename cannot be empty."
        )

    client = Client()

    model_version = client.get_model_version(model_name, model_stage)
    model_version_name = str(model_version.name)

    artifact = model_version.get_model_artifact(model_artifact_name)
    if artifact is None:
        raise ValueError(
            f"Model artifact '{model_artifact_name}' not found for {model_name}, stage '{model_stage}', version '{model_version_name}'."
        )

    model_artifact_uri = f"{artifact.uri}/{model_artifact_filename}"

    logger.info("Model artifact URI: %s", model_artifact_uri)
    logger.info("Model version: %s", model_version_name)

    return model_artifact_uri, model_version_name
