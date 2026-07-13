"""
steps/serving/deploy.py

ZenML step: deploy_endpoint

Deploys the serving Docker image to SageMaker (AWS) or runs it locally.
Registers the endpoint URL in ZenML model metadata.

Reusable across workflows — pass model_name as a step parameter
(via YAML config or pipeline call).
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from zenml import log_metadata, step
from zenml.client import Client

from workflows.matrix_factorization.configs import CFG_MODEL_NAME

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def deploy_endpoint(
    deploy_mode: Literal["local", "sagemaker"] = "local",
    serving_image_uri: str = "",
    model_stage: str = "staging",
    execution_role_name: str = "zenml-execution-role",
    endpoint_name: str = "als-movie-recommender",
    instance_type: str = "ml.t2.medium",
    local_port: int = 8000,
) -> Annotated[str, "endpoint_url"]:
    """
    Deploy the recommendation serving endpoint.

    Args:
        deploy_mode: "local" runs via docker run; "sagemaker" deploys to AWS.
        serving_image_uri: Docker image URI to deploy.
        model_stage: ZenML model stage to look up for metadata logging.
        endpoint_name: Endpoint name.
        execution_role_name: IAM role name for SageMaker deployment (used in AWS mode) - For sagemaker deployment only.
        instance_type: SageMaker instance type - For sagemaker deployment only.
        local_port: Host port mapped when running locally - For local deployment only.

    Returns:
        Endpoint URL string.
    """
    if not serving_image_uri:
        raise ValueError("serving_image_uri and model_name cannot be empty.")

    if deploy_mode == "local":
        endpoint_url = _deploy_local(serving_image_uri, endpoint_name, local_port)
    elif deploy_mode == "sagemaker":
        endpoint_url = _deploy_sagemaker(
            serving_image_uri, endpoint_name, instance_type, execution_role_name
        )
    else:
        raise ValueError(f"Unknown deploy_mode: {deploy_mode!r}. Choose 'local' or 'sagemaker'.")

    # Attach endpoint URL to the model version in ZenML
    client = Client()
    model_version = client.get_model_version(CFG_MODEL_NAME, model_stage)
    log_metadata(
        metadata={
            "endpoint_url": endpoint_url,
            "deploy_mode": deploy_mode,
        },
        model_version_id=model_version.id,
    )

    logger.info("Endpoint deployed: %s", endpoint_url)
    return endpoint_url


def _deploy_local(image_uri: str, name: str, local_port: int) -> str:
    """Run the serving container locally via Docker."""
    import subprocess

    # TODO: Update to use ZenML pipeline deployment
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", "8080:8080", image_uri],
        check=True,
    )
    return f"http://localhost:{local_port}"


def _deploy_sagemaker(
    image_uri: str, endpoint_name: str, instance_type: str, execution_role_name
) -> str:
    """Deploy to SageMaker endpoint with blue/green traffic shifting."""
    import boto3

    sm = boto3.client("sagemaker")

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri, "Mode": "SingleModel"},
        ExecutionRoleArn=_get_execution_role_arn(execution_role_name),
    )

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

    existing = sm.list_endpoints(NameContains=endpoint_name, StatusEquals="InService")
    if existing["Endpoints"]:
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        logger.info("Updating SageMaker endpoint: %s", endpoint_name)
    else:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        logger.info("Creating SageMaker endpoint: %s", endpoint_name)

    import boto3.session

    region = boto3.session.Session().region_name
    return f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"


def _get_execution_role_arn(execution_role_name: str) -> str:
    import boto3

    iam = boto3.client("iam")
    return iam.get_role(RoleName=execution_role_name)["Role"]["Arn"]
