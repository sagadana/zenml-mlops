"""
steps/serving/build_image.py

ZenML step: build_serving_image

Builds a Docker image from the pre-computed serving_image_uri and model
artifact URI, then pushes it to the container registry.

Call get_model_artifact_uri first to resolve the URI; this step is
intentionally cacheable — if serving_image_uri has not changed (same model
version), ZenML will skip the build automatically.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from zenml import step

logger = logging.getLogger(__name__)


def _prepare_docker_env() -> tuple[dict[str, str], str]:
    """Prepare a writable Docker config directory for subprocess calls."""
    docker_env = os.environ.copy()

    home_dir = docker_env.get("HOME", "")
    if not home_dir or home_dir == "/":
        # Some orchestrators run steps with HOME=/ which leads to '/.docker'.
        # Use a known-writable base directory instead.
        home_dir = tempfile.gettempdir()
        docker_env["HOME"] = home_dir

    docker_config_dir = docker_env.get("DOCKER_CONFIG", "")
    if docker_config_dir:
        docker_config_path = Path(docker_config_dir)
    else:
        docker_config_path = Path(home_dir) / ".docker"

    try:
        docker_config_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_path = Path(tempfile.gettempdir()) / ".docker"
        fallback_path.mkdir(parents=True, exist_ok=True)
        docker_config_path = fallback_path

    docker_config_dir = str(docker_config_path)
    docker_env["DOCKER_CONFIG"] = docker_config_dir

    return docker_env, docker_config_dir


@step
def build_serving_image(
    model_artifact_uri: str = "",
    model_version: str = "",
    workflow_name: str = "",
    model_name: str = "",
    image_registry_uri: str = "",
    service_name: str = "zenml-mlops-serving",
    serving_dockerfile_path: str = "docker/serving/Dockerfile",
) -> Annotated[str, "serving_image_uri"]:
    """
    Build and push the FastAPI serving Docker image to the container registry.

    This step is cacheable: ZenML will skip it when all inputs (especially
    serving_image_uri, which encodes the model version) are unchanged.

    Args:
        image_tag: Image tag (repository + version) to build and push.
        model_artifact_uri: ZenML artifact store URI for the model to embed in the image.
        workflow_name: Workflow directory name; determines the FastAPI app bundled into the image.
        model_name: Registered ZenML model name (passed as a build-arg).
        image_registry_uri: Container registry base URI.
            Local example : "localhost:5001"
            AWS example   : "123456789.dkr.ecr.us-east-1.amazonaws.com"
        service_name: Service name passed as a Docker build-arg.
        serving_dockerfile_path: Path to the serving Dockerfile (relative to repo root).

    Returns:
        serving_image_uri: The full image URI that was built and pushed.
    """
    if (
        not model_artifact_uri
        or not model_version
        or not workflow_name
        or not model_name
        or not image_registry_uri
    ):
        raise ValueError(
            "model_artifact_uri, model_version, workflow_name, model_name, and image_registry_uri cannot be empty."
        )

    image_name = f"{workflow_name}/{model_name}".replace("_", "-").lower()
    image_version = model_version.replace(" ", "-").lower()
    image_tag = f"{image_name}:{image_version}"

    # e.g. localhost:5001/matrix-factorization/als-mf-model:1
    #   or 123456789.dkr.ecr.us-east-1.amazonaws.com/matrix-factorization/als-mf-model:1
    serving_image_uri = f"{image_registry_uri}/{image_tag}"

    docker_build_command = [
        "docker",
        "buildx",
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

    docker_env, docker_config_dir = _prepare_docker_env()
    logger.info("Using Docker config directory: %s", docker_config_dir)

    logger.info("Building serving image: %s", serving_image_uri)
    result = subprocess.run(
        docker_build_command,
        capture_output=True,
        text=True,
        env=docker_env,
    )
    if result.returncode != 0:
        logger.error("Docker build failed:\n%s", result.stderr)
        raise RuntimeError(
            "Docker build failed. Ensure the step container can access a Docker daemon "
            "(for local_docker, mount /var/run/docker.sock via orchestrator run_args). "
            f"Error: {result.stderr[:500]}"
        )

    logger.info("Pushing serving image: %s", serving_image_uri)
    subprocess.run(
        ["docker", "push", serving_image_uri],
        check=True,
        env=docker_env,
    )

    logger.info("Serving image built and pushed: %s", serving_image_uri)

    return serving_image_uri
