"""
steps/serving/deploy.py

ZenML step: deploy_endpoint

Deploys the serving Docker image to SageMaker (AWS) or runs it locally.
Registers the endpoint URL in ZenML model metadata.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import Model, log_metadata, step

logger = logging.getLogger(__name__)

_MODEL_NAME = "als_movie_recommender"


@step(enable_cache=False, model=Model(name=_MODEL_NAME))
def deploy_endpoint(
    serving_image_uri: str,
    endpoint_name: str = "als-movie-recommender",
    instance_type: str = "ml.t2.medium",
    deploy_mode: str = "local",
) -> Annotated[str, "endpoint_url"]:
    """
    Deploy the recommendation serving endpoint.

    Args:
        serving_image_uri: Docker image URI to deploy.
        endpoint_name: SageMaker endpoint name (used in AWS mode).
        instance_type: SageMaker instance type.
        deploy_mode: "local" runs via docker-compose; "sagemaker" deploys to AWS.

    Returns:
        Endpoint URL string.
    """
    if deploy_mode == "local":
        endpoint_url = _deploy_local(serving_image_uri, endpoint_name)
    elif deploy_mode == "sagemaker":
        endpoint_url = _deploy_sagemaker(serving_image_uri, endpoint_name, instance_type)
    else:
        raise ValueError(f"Unknown deploy_mode: {deploy_mode!r}. Choose 'local' or 'sagemaker'.")

    # Log endpoint URL to ZenML model metadata for discovery
    log_metadata(
        metadata={"endpoint_url": endpoint_url, "deploy_mode": deploy_mode},
        infer_model=True,
    )
    logger.info("Endpoint deployed: %s", endpoint_url)
    return endpoint_url


def _deploy_local(image_uri: str, name: str) -> str:
    """Run the serving container locally via Docker."""
    import subprocess

    # Stop any existing container with the same name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", "8080:8080", image_uri],
        check=True,
    )
    return "http://localhost:8080"


def _deploy_sagemaker(image_uri: str, endpoint_name: str, instance_type: str) -> str:
    """Deploy to SageMaker endpoint with blue/green traffic shifting."""
    import boto3

    sm = boto3.client("sagemaker")

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    # Create SageMaker model
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri, "Mode": "SingleModel"},
        ExecutionRoleArn=_get_execution_role_arn(),
    )

    # Create endpoint config
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # Create or update endpoint (blue/green: update if exists)
    existing = sm.list_endpoints(NameContains=endpoint_name, StatusEquals="InService")
    if existing["Endpoints"]:
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        logger.info("Updating SageMaker endpoint: %s", endpoint_name)
    else:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        logger.info("Creating SageMaker endpoint: %s", endpoint_name)

    # Derive URL (SageMaker endpoints use the runtime API, not a public HTTP URL)
    import boto3.session

    region = boto3.session.Session().region_name
    return f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"


def _get_execution_role_arn() -> str:
    import boto3

    iam = boto3.client("iam")
    return iam.get_role(RoleName="zenml-execution-role")["Role"]["Arn"]
