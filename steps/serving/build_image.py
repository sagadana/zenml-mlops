"""
steps/serving/build_image.py

ZenML step: build_serving_image

Builds a Docker image from the pre-computed serving_image_uri and model
artifact URI, then pushes it to the container registry.

Call prepare_serving_uris first to resolve the URI; this step is
intentionally cacheable — if serving_image_uri has not changed (same model
version), ZenML will skip the build automatically.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Annotated

from zenml import step

logger = logging.getLogger(__name__)


@step
def build_serving_image(
    serving_image_uri: str = "",
    model_artifact_uri: str = "",
    workflow_name: str = "",
    model_name: str = "",
    service_name: str = "zenml-mlops-serving",
    serving_dockerfile_path: str = "docker/serving/Dockerfile",
) -> Annotated[str, "serving_image_uri"]:
    """
    Build and push the FastAPI serving Docker image to the container registry.

    This step is cacheable: ZenML will skip it when all inputs (especially
    serving_image_uri, which encodes the model version) are unchanged.

    Args:
        serving_image_uri: Full image URI produced by prepare_serving_uris.
            E.g. "localhost:5001/matrix-factorization:1.0.0"
                 "123456789.dkr.ecr.us-east-1.amazonaws.com/matrix-factorization:1.0.0"
        model_artifact_uri: ZenML artifact store URI for the model to embed in the image.
        workflow_name: Workflow directory name; determines the FastAPI app bundled into the image.
        model_name: Registered ZenML model name (passed as a build-arg).
        service_name: Service name passed as a Docker build-arg.
        serving_dockerfile_path: Path to the serving Dockerfile (relative to repo root).

    Returns:
        serving_image_uri: The full image URI that was built and pushed.
    """
    if not serving_image_uri or not model_artifact_uri or not workflow_name or not model_name:
        raise ValueError(
            "serving_image_uri, model_artifact_uri, workflow_name, and model_name cannot be empty."
        )

    docker_build_command = [
        "docker",
        "build",
        "-t",
        serving_image_uri,
        "-f",
        serving_dockerfile_path,
        "--build-arg",
        f"WORKFLOW={workflow_name}",
        "--build-arg",
        f"MODEL_URI={model_artifact_uri}",
        "--build-arg",
        f"MODEL_NAME={model_name}",
        "--build-arg",
        f"SERVICE={service_name}",
    ]

    docker_build_command.append(".")

    logger.info("Building serving image: %s", serving_image_uri)
    result = subprocess.run(docker_build_command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Docker build failed:\n%s", result.stderr)
        raise RuntimeError(f"Docker build failed: {result.stderr[:500]}")

    logger.info("Pushing serving image: %s", serving_image_uri)
    subprocess.run(["docker", "push", serving_image_uri], check=True)
    logger.info("Serving image pushed: %s", serving_image_uri)

    return serving_image_uri
