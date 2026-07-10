"""
steps/serving/build_image.py

ZenML step: build_serving_image

Builds a Docker image containing the FastAPI serving app and the current
production model, then pushes it to ECR.

Reusable across workflows — pass model_name and model_artifact_name as
step parameters (via YAML config or pipeline call).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Annotated

from zenml import step
from zenml.client import Client

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def build_serving_image(
    model_name: str,
    model_artifact_name: str,
    workflow_name: str,
    ecr_uri: str | None = None,
    model_stage: str = "staging",
    service_name: str = "zenml-mlops-serving",
    serving_dockerfile_path: str = "docker/serving/Dockerfile",
) -> Annotated[str, "serving_image_uri"]:
    """
    Build and push the FastAPI serving Docker image to ECR.

    Args:
        model_name: Registered ZenML model name to embed in the image.
        model_artifact_name: Name of the model artifact to retrieve from the model version.
        ecr_uri: ECR base URI (e.g. "123456789.dkr.ecr.us-east-1.amazonaws.com").
            If empty, uses local tag only (for local dev).
        model_stage: ZenML model stage to embed in image tag.
        service_name: Service name used when constructing the ECR image path.
        workflow_name: Workflow directory name; determines the FastAPI app bundled into the image.
        serving_dockerfile_path: Path to the serving Dockerfile (relative to repo root).

    Returns:
        Full image URI (ECR URI or local tag).
    """
    client = Client()
    image_name = f"{model_name.replace('_', '-').lower()}-serving"

    model_version = client.get_model_version(model_name, model_stage)
    version_str = str(model_version.model.latest_version_name).replace(" ", "-").lower()

    artifact = model_version.get_artifact(model_artifact_name)
    if artifact is None:
        raise ValueError(f"Model artifact '{model_artifact_name}' not found for {model_name}")

    local_tag = f"{image_name}:{version_str}"

    # Construct the full image URI (ECR or local):
    # E.g. "123456789.dkr.ecr.us-east-1.amazonaws.com/recs-wf-serving/recommender:1.0.0"
    image_uri = (
        f"{ecr_uri}/{workflow_name.replace('_', '-').lower()}/{version_str}"
        if ecr_uri
        else local_tag
    )

    model_uri = artifact.uri

    logger.info("Building serving image: %s", local_tag)
    result = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            local_tag,
            "-f",
            serving_dockerfile_path,
            "--build-arg",
            f"WORKFLOW={workflow_name}",
            "--build-arg",
            f"MODEL_URI={model_uri}",
            "--build-arg",
            f"MODEL_NAME={model_name}",
            "--build-arg",
            f"SERVICE={service_name}",
            ".",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Docker build failed:\n%s", result.stderr)
        raise RuntimeError(f"Docker build failed: {result.stderr[:500]}")

    if ecr_uri:
        logger.info("Tagging and pushing to ECR: %s", image_uri)
        subprocess.run(["docker", "tag", local_tag, image_uri], check=True)
        subprocess.run(["docker", "push", image_uri], check=True)
        logger.info("Serving image pushed to ECR: %s", image_uri)

    return image_uri
