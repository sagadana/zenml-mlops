"""
pipelines/matrix_factorization/serving_pipeline.py

Model serving pipeline: batch pre-computation + real-time endpoint deployment.

Two parallel sub-flows:
  1. Batch: generate_batch_recommendations → S3 + DynamoDB
  2. Real-time: build_serving_image → deploy_endpoint

Run:
  python run.py --pipeline serving --config workflows/matrix_factorization/configs/local.yaml
  python run.py --pipeline serving --config workflows/matrix_factorization/configs/aws.yaml --stack aws_stack
"""

import logging

from zenml import pipeline

from workflows.matrix_factorization.steps.serving.batch_predict import (
    generate_batch_recommendations,
)
from workflows.matrix_factorization.steps.serving.build_image import build_serving_image
from workflows.matrix_factorization.steps.serving.deploy import deploy_endpoint

logger = logging.getLogger(__name__)


@pipeline(name="matrix_factorization_serving", enable_cache=False)
def serving_pipeline(
    # Batch serving
    batch_top_k: int = 50,
    batch_output_path: str = "s3://aips-zenml-predictions/batch",
    dynamodb_table: str = "",
    model_stage: str = "staging",
    # Real-time serving
    ecr_uri: str = "",
    endpoint_name: str = "als-movie-recommender",
    instance_type: str = "ml.t2.medium",
    deploy_mode: str = "local",
) -> None:
    """
    Deploy the trained model for both batch and real-time serving.

    Batch flow:
      - Loads production model from ZenML MCP
      - Generates top-K recs for all users via Dask
      - Writes to S3 and optionally to DynamoDB (TTL 48h)

    Real-time flow:
      - Builds a Docker image with FastAPI + model
      - Pushes to ECR (if ecr_uri set)
      - Deploys to SageMaker or local Docker

    Args:
        batch_top_k: Recommendations per user for batch job.
        batch_output_path: S3 or local output path.
        dynamodb_table: DynamoDB table name ("" to skip).
        model_stage: Model stage to load ("staging" or "production").
        ecr_uri: ECR base URI (leave empty for local-only build).
        endpoint_name: Name for SageMaker endpoint or Docker container.
        instance_type: SageMaker instance type.
        deploy_mode: "local" or "sagemaker".
    """
    # Sub-flow 1: Batch recommendations (independent of real-time flow)
    batch_report = generate_batch_recommendations(
        batch_top_k=batch_top_k,
        batch_output_path=batch_output_path,
        dynamodb_table=dynamodb_table if dynamodb_table else None,
        model_stage=model_stage,
    )
    logger.info("Batch job report: %s", batch_report)

    # Sub-flow 2: Real-time endpoint
    serving_image_uri = build_serving_image(
        ecr_uri=ecr_uri,
        model_stage=model_stage,
    )

    endpoint_url = deploy_endpoint(
        serving_image_uri=serving_image_uri,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        deploy_mode=deploy_mode,
    )
    logger.info("Real-time endpoint deployed at: %s", endpoint_url)
