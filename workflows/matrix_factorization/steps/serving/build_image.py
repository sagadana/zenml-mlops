"""
steps/serving/build_image.py

ZenML step: build_serving_image

Builds a Docker image containing the FastAPI serving app and the current
production ALSRecommender model, then pushes it to ECR.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Annotated

from zenml import step
from zenml.client import Client

logger = logging.getLogger(__name__)

_MODEL_NAME = "als_movie_recommender"


@step(enable_cache=False)
def build_serving_image(
    ecr_uri: str = "",
    model_stage: str = "staging",
) -> Annotated[str, "serving_image_uri"]:
    """
    Build and push the FastAPI serving Docker image to ECR.

    Args:
        ecr_uri: ECR base URI (e.g. "123456789.dkr.ecr.us-east-1.amazonaws.com").
            If empty, uses local tag only (for local dev).
        model_stage: ZenML model stage to embed in image tag.

    Returns:
        Full image URI (ECR URI or local tag).
    """
    client = Client()
    model_version = client.get_model_version(_MODEL_NAME, model_stage)
    version_str = str(model_version.version).replace(" ", "-").lower()

    local_tag = f"aips-zenml-serving:{version_str}"
    image_uri = f"{ecr_uri}/aips-zenml/als-serving:{version_str}" if ecr_uri else local_tag

    logger.info("Building serving image: %s", local_tag)
    result = subprocess.run(
        ["docker", "build", "-t", local_tag, "-f", "workflows/matrix_factorization/serving/Dockerfile", "."],
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
        logger.info("Pushed: %s", image_uri)

    return image_uri
