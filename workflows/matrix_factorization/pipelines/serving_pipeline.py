"""
pipelines/matrix_factorization/serving_pipeline.py

Model serving pipeline: batch pre-computation + real-time endpoint deployment.

Two parallel sub-flows:
  1. Batch: generate_batch_recommendations → S3 + DynamoDB
  2. Real-time: build_serving_image → deploy_endpoint

Run:
  python run.py run --workflow matrix_factorization --pipeline serving_pipeline --config workflows/matrix_factorization/configs/local/serving_pipeline.yaml
  python run.py run --workflow matrix_factorization --pipeline serving_pipeline --config workflows/matrix_factorization/configs/aws/serving_pipeline.yaml --stack aws_stack
"""

import logging

from zenml import pipeline

from steps.serving.build_image import build_serving_image
from steps.serving.deploy import deploy_endpoint
from workflows.matrix_factorization.steps.serving.batch_predict import (
    generate_batch_recommendations,
)

logger = logging.getLogger(__name__)


@pipeline(name="matrix_factorization_serving", enable_cache=False)
def serving_pipeline() -> None:
    """
    Deploy the trained model for both batch and real-time serving.

    Batch flow:
      - Loads production model from ZenML MCP
      - Generates top-K recs for all users in parallel
      - Writes to S3 and optionally to DynamoDB (TTL 48h)

    Real-time flow:
      - Builds a Docker image with FastAPI + model
      - Pushes to ECR (if ecr_uri set)
      - Deploys to SageMaker or local Docker

    Step-specific parameters are configured in step blocks of the
    pipeline run config YAML.
    """
    # Sub-flow 1: Batch recommendations (independent of real-time flow)
    batch_report = generate_batch_recommendations()
    logger.info("Batch job report: %s", batch_report)

    # Sub-flow 2: Real-time endpoint
    serving_image_uri = build_serving_image()

    endpoint_url = deploy_endpoint(
        serving_image_uri=serving_image_uri,
    )
    logger.info("Real-time endpoint deployed at: %s", endpoint_url)
