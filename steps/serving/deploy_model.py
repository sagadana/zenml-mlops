"""
steps/serving/deploy_model.py

ZenML step: deploy_endpoint

Deploys the serving Docker image to SageMaker (AWS) or runs it locally.
Registers the endpoint URL in ZenML model metadata.

Reusable across workflows — pass model_name as a step parameter
(via YAML config or pipeline call).

Using these steps requires the following directory structure in your ZenML repository:
    workflows/
        <workflow_name>/
            configs/
                <config_files>.py
            models/
                <model_files>
            serving/
                app.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Literal

from zenml import log_metadata, step
from zenml.client import Client
from zenml.enums import ModelStages

from workflows.matrix_factorization.configs import (
    CFG_DEPLOYMENT_ENDPOINT_URL_OUTPUT,
    CFG_MODEL_NAME,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from types_boto3_sagemaker.literals import ProductionVariantInstanceTypeType
    from types_boto3_sagemaker.type_defs import InstancePoolTypeDef
else:
    type ProductionVariantInstanceTypeType = str
    type InstancePoolTypeDef = dict


@step(enable_cache=False)
def deploy_endpoint(
    serving_image_uri: str = "",
    model_stage: ModelStages = ModelStages.STAGING,
    deploy_mode: Literal["local", "sagemaker"] = "local",
    execution_role_name: str = "zenml-execution-role",
    endpoint_name: str = "als-movie-recommender",
    instance_type: ProductionVariantInstanceTypeType = "ml.t2.medium",
    instance_type_pool: list[ProductionVariantInstanceTypeType] | None = None,
    local_port: int = 8000,
    container_env: dict[str, str] | None = None,
) -> Annotated[str, CFG_DEPLOYMENT_ENDPOINT_URL_OUTPUT]:
    """
    Deploy the recommendation serving endpoint.

    Args:
        deploy_mode: "local" runs via docker run; "sagemaker" deploys to AWS.
        serving_image_uri: Docker image URI to deploy.
        model_stage: ZenML model stage to look up for metadata logging.
        endpoint_name: Endpoint name.
        execution_role_name: IAM role name for SageMaker deployment (used in AWS mode) - For sagemaker deployment only.
        instance_type: SageMaker instance type - For sagemaker deployment only.
        instance_type_pool: List of SageMaker instance types to try in order - For sagemaker deployment only.
        local_port: Host port mapped when running locally - For local deployment only.
        container_env: Environment variables (name -> value) to inject into the serving container.
    Returns:
        Endpoint URL string.
    """
    if instance_type_pool is None:
        instance_type_pool = []

    if container_env is None:
        container_env = {}

    if not serving_image_uri:
        raise ValueError("serving_image_uri and model_name cannot be empty.")

    if deploy_mode == "local":
        endpoint_url = _deploy_local(serving_image_uri, endpoint_name, local_port, container_env)
    elif deploy_mode == "sagemaker":
        endpoint_url = _deploy_sagemaker(
            serving_image_uri,
            endpoint_name,
            execution_role_name,
            container_env,
            instance_type,
            instance_type_pool,
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


def _deploy_local(image_uri: str, name: str, local_port: int, container_env: dict[str, str]) -> str:
    """Run the serving container locally via Docker."""
    import subprocess

    env_args: list[str] = []
    for key, value in container_env.items():
        if value:
            env_args.extend(["-e", f"{key}={value}"])

    subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    max_port_attempts = 20
    last_stderr = ""

    for offset in range(max_port_attempts):
        candidate_port = local_port + offset
        run_result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-v",
                "/var/run/docker.sock:/var/run/docker.sock",
                "-p",
                f"{candidate_port}:8080",
                *env_args,
                image_uri,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        if run_result.returncode == 0:
            if candidate_port != local_port:
                logger.warning(
                    "Requested local port %d was unavailable; deployed endpoint '%s' on port %d.",
                    local_port,
                    name,
                    candidate_port,
                )
            return f"http://localhost:{candidate_port}"

        last_stderr = run_result.stderr.strip() if run_result.stderr else "unknown docker error"
        if "port is already allocated" not in last_stderr.lower():
            raise RuntimeError(
                f"Local deploy failed for endpoint '{name}' on port {candidate_port}: {last_stderr}"
            )

        # Docker may leave a stopped container with the same name when start fails.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    raise RuntimeError(
        f"Local deploy failed for endpoint '{name}': no free host port in range "
        f"{local_port}-{local_port + max_port_attempts - 1}. Last error: {last_stderr}"
    )


def _deploy_sagemaker(
    image_uri: str,
    endpoint_name: str,
    execution_role_name: str,
    container_env: dict[str, str],
    instance_type: ProductionVariantInstanceTypeType,
    instance_type_pool: list[ProductionVariantInstanceTypeType] | None = None,
) -> str:
    """Deploy to SageMaker endpoint with blue/green traffic shifting."""
    import boto3

    if instance_type_pool is None:
        instance_type_pool = []

    sm = boto3.client("sagemaker")

    container_environment = {key: value for key, value in container_env.items() if value}

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "Mode": "SingleModel",
            "Environment": container_environment,
        },
        ExecutionRoleArn=_get_execution_role_arn(execution_role_name),
    )

    instances = list(set([instance_type] + instance_type_pool))
    pool: list[InstancePoolTypeDef] = []
    for i, it in enumerate(instances):
        pool.append(
            {
                "InstanceType": it,  # type: ignore
                "Priority": len(instances) - i,  # higher priority for the first instance type
            }
        )

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InstancePools": pool,
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
