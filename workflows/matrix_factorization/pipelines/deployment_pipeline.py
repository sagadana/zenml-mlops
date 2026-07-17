"""
pipelines/matrix_factorization/deployment_pipeline.py

Real-time serving deployment pipeline.

Flow:
  get_model_artifact_uri → build_serving_image → deploy_endpoint

Run:
  python run.py run --workflow matrix_factorization --pipeline deployment_pipeline --config workflows/matrix_factorization/configs/local/deployment_pipeline.yaml
  python run.py run --workflow matrix_factorization --pipeline deployment_pipeline --config workflows/matrix_factorization/configs/aws/deployment_pipeline.yaml --stack aws_stack
"""

import logging

from zenml import pipeline

from steps.serving.build_image import build_serving_image
from steps.serving.deploy_model import deploy_endpoint
from steps.serving.model_artifacts import get_model_artifact_uri
from workflows.matrix_factorization.configs import (
    CFG_DEPLOYMENT_PIPELINE_NAME,
    CFG_DEPLOYMENT_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_DEPLOYMENT_PIPELINE_SNAPSHOT_NAME,
    CFG_MODEL_ARTIFACT_NAME,
    CFG_MODEL_NAME,
    CFG_WORKFLOW_NAME,
)

logger = logging.getLogger(__name__)


@pipeline(name=CFG_DEPLOYMENT_PIPELINE_NAME)
def deployment_pipeline() -> None:
    """Build and deploy a real-time endpoint for the ALS recommender."""
    model_artifact_uri, model_version = get_model_artifact_uri(
        model_name=CFG_MODEL_NAME,
        model_artifact_name=CFG_MODEL_ARTIFACT_NAME,
    )
    built_image_uri = build_serving_image(
        model_artifact_uri=model_artifact_uri,
        model_version=model_version,
        workflow_name=CFG_WORKFLOW_NAME,
        model_name=CFG_MODEL_NAME,
    )
    endpoint_url = deploy_endpoint(serving_image_uri=built_image_uri)
    logger.info("Real-time endpoint deployed at: %s", endpoint_url)


deployment_pipeline.create_snapshot(
    name=CFG_DEPLOYMENT_PIPELINE_SNAPSHOT_NAME,
    description=CFG_DEPLOYMENT_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "deployment"],
    replace=True,
)
