#!/usr/bin/env bash
# infra/aws/setup_stacks.sh
#
# Idempotent ZenML stack setup script for AWS.
# Run this once after deploying the ZenML server via `zenml deploy`.
#
# Prerequisites:
#   - AWS CLI configured with credentials that can manage IAM/S3/ECR resources
#   - ZenML CLI connected to remote server: zenml connect --url https://<server>
#   - Environment variables set (see below)
#
# Required environment variables:
#   AWS_ACCOUNT_ID      — 12-digit AWS account ID
#   AWS_REGION          — e.g. us-east-1
#
# Optional environment variables:
#   ZENML_ARTIFACT_BUCKET     — default: zenml-artifacts
#   ZENML_CHECKPOINT_BUCKET   — default: zenml-checkpoints
#   ZENML_DATA_BUCKET         — default: zenml-data
#   ZENML_PREDICTIONS_BUCKET  — default: zenml-predictions
#   ZENML_ECR_REPOSITORY      — default: zenml
#   ZENML_AWS_CONNECTOR_NAME  — default: aws_connector
#
# Usage:
#   export AWS_ACCOUNT_ID=123456789012
#   export AWS_REGION=us-east-1
#   bash infra/aws/setup_stacks.sh

set -euo pipefail

: "${AWS_ACCOUNT_ID:?ERROR: AWS_ACCOUNT_ID is not set}"
: "${AWS_REGION:?ERROR: AWS_REGION is not set}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ROLE_NAME="${ZENML_EXEC_ROLE_NAME:-zenml-execution-role}"
ROLE_POLICY_NAME="${ZENML_EXEC_ROLE_POLICY_NAME:-zenml-execution-policy}"

ZENML_EXECUTION_ROLE_ARN="${ZENML_EXECUTION_ROLE_ARN:-arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}}"
export ZENML_EXECUTION_ROLE_ARN

DEFAULT_ARTIFACT_BUCKET="zenml-artifacts"
DEFAULT_CHECKPOINT_BUCKET="zenml-checkpoints"
DEFAULT_DATA_BUCKET="zenml-data"
DEFAULT_PREDICTIONS_BUCKET="zenml-predictions"
DEFAULT_ECR_REPOSITORY="zenml"

ZENML_ARTIFACT_BUCKET="${ZENML_ARTIFACT_BUCKET:-${DEFAULT_ARTIFACT_BUCKET}}"
ZENML_CHECKPOINT_BUCKET="${ZENML_CHECKPOINT_BUCKET:-${DEFAULT_CHECKPOINT_BUCKET}}"
ZENML_DATA_BUCKET="${ZENML_DATA_BUCKET:-${DEFAULT_DATA_BUCKET}}"
ZENML_PREDICTIONS_BUCKET="${ZENML_PREDICTIONS_BUCKET:-${DEFAULT_PREDICTIONS_BUCKET}}"
ZENML_ECR_REPOSITORY="${ZENML_ECR_REPOSITORY:-${DEFAULT_ECR_REPOSITORY}}"
ZENML_AWS_CONNECTOR_NAME="${ZENML_AWS_CONNECTOR_NAME:-aws_connector}"

ASSUME_ROLE_POLICY_DOCUMENT="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)"

IAM_POLICY_DOCUMENT="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ArtifactAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation",
        "s3:HeadObject",
        "s3:HeadBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${ZENML_ARTIFACT_BUCKET}",
        "arn:aws:s3:::${ZENML_ARTIFACT_BUCKET}/*",
        "arn:aws:s3:::${ZENML_CHECKPOINT_BUCKET}",
        "arn:aws:s3:::${ZENML_CHECKPOINT_BUCKET}/*",
        "arn:aws:s3:::${ZENML_DATA_BUCKET}",
        "arn:aws:s3:::${ZENML_DATA_BUCKET}/*",
        "arn:aws:s3:::${ZENML_PREDICTIONS_BUCKET}",
        "arn:aws:s3:::${ZENML_PREDICTIONS_BUCKET}/*"
      ]
    },
    {
      "Sid": "ECRAccess",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:DescribeRepositories",
        "ecr:CreateRepository",
        "ecr:ListImages"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SageMakerAccess",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreatePipeline",
        "sagemaker:UpdatePipeline",
        "sagemaker:DeletePipeline",
        "sagemaker:StartPipelineExecution",
        "sagemaker:StopPipelineExecution",
        "sagemaker:DescribePipeline",
        "sagemaker:DescribePipelineExecution",
        "sagemaker:ListPipelineExecutions",
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:DescribeProcessingJob",
        "sagemaker:StopProcessingJob",
        "sagemaker:CreateModel",
        "sagemaker:CreateEndpoint",
        "sagemaker:CreateEndpointConfig",
        "sagemaker:UpdateEndpoint",
        "sagemaker:DeleteEndpoint",
        "sagemaker:DescribeEndpoint",
        "sagemaker:ListEndpoints",
        "sagemaker:AddTags"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogsAccess",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
        "logs:GetLogEvents",
        "logs:FilterLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    },
    {
      "Sid": "DynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:BatchWriteItem",
        "dynamodb:BatchGetItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:DescribeTable",
        "dynamodb:CreateTable"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/movie-recommendations"
    },
    {
      "Sid": "IAMPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::*:role/${ROLE_NAME}",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "sagemaker.amazonaws.com"
        }
      }
    }
  ]
}
EOF
)"

# -------------------------
# Create AWS resources 
# -------------------------

## IAM execution role
echo ""
echo "==> Creating IAM execution role (idempotent)..."
if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Role ${ROLE_NAME} already exists, skipping create"
else
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${ASSUME_ROLE_POLICY_DOCUMENT}" \
    >/dev/null
fi

aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "${ROLE_POLICY_NAME}" \
  --policy-document "${IAM_POLICY_DOCUMENT}" \
  >/dev/null

ZENML_EXECUTION_ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)"
export ZENML_EXECUTION_ROLE_ARN
echo "  ✓ IAM role ready: ${ZENML_EXECUTION_ROLE_ARN}"

## S3 buckets
echo "==> Creating S3 buckets (idempotent)..."
for bucket in "$ZENML_ARTIFACT_BUCKET" "$ZENML_CHECKPOINT_BUCKET" "$ZENML_DATA_BUCKET" "$ZENML_PREDICTIONS_BUCKET"; do
  aws s3api create-bucket \
    --bucket "$bucket" \
    --region "$AWS_REGION" \
    $([ "$AWS_REGION" != "us-east-1" ] && echo "--create-bucket-configuration LocationConstraint=$AWS_REGION") \
    2>/dev/null || echo "  Bucket $bucket already exists, skipping"
  # Block all public access
  aws s3api put-public-access-block \
    --bucket "$bucket" \
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
done
echo "  ✓ S3 buckets ready"

## ECR repository
echo "==> Creating ECR repository (idempotent)..."
aws ecr describe-repositories --repository-names "${ZENML_ECR_REPOSITORY}" --region "$AWS_REGION" 2>/dev/null || \
  aws ecr create-repository \
    --repository-name "${ZENML_ECR_REPOSITORY}" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true
echo "  ✓ ECR repository ready"


# --------------------------------------
# Register ZenML AWS service connector
# --------------------------------------

echo ""
echo "==> Registering AWS service connector..."

zenml service-connector describe "${ZENML_AWS_CONNECTOR_NAME}" 2>/dev/null || \
  zenml service-connector register "${ZENML_AWS_CONNECTOR_NAME}" \
    --type aws \
    --auth-method iam-role \
    --role_arn="${ZENML_EXECUTION_ROLE_ARN}" \
    --region="${AWS_REGION}"
echo "  ✓ Service connector ready: ${ZENML_AWS_CONNECTOR_NAME}"

# --------------------------------------
# Register ZenML stack components
# --------------------------------------

echo ""
echo "==> Registering ZenML stack components..."

## Artifact store
zenml artifact-store describe s3_store 2>/dev/null || \
  zenml artifact-store register s3_store \
    --flavor=s3 \
    --path="s3://${ZENML_ARTIFACT_BUCKET}/"
echo "  ✓ Artifact store: s3_store"

## Container registry
zenml container-registry describe ecr_registry 2>/dev/null || \
  zenml container-registry register ecr_registry \
    --flavor=aws \
    --uri="${ECR_URI}"
echo "  ✓ Container registry: ecr_registry"

## Orchestrator (SageMaker Pipelines)
zenml orchestrator describe sagemaker_orch 2>/dev/null || \
  zenml orchestrator register sagemaker_orch \
    --flavor=sagemaker \
    --region="${AWS_REGION}" \
    --execution_role="${ZENML_EXECUTION_ROLE_ARN}"
echo "  ✓ Orchestrator: sagemaker_orch"

## MLflow experiment tracker (requires MLFLOW_TRACKING_URI env var)
: "${MLFLOW_TRACKING_URI:=http://localhost:5000}"
zenml experiment-tracker describe mlflow_tracker 2>/dev/null || \
  zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="${MLFLOW_TRACKING_URI}" \
    --tracking_username="${MLFLOW_TRACKING_USERNAME:-}" \
    --tracking_password="${MLFLOW_TRACKING_PASSWORD:-}"
echo "  ✓ Experiment tracker: mlflow_tracker (uri=${MLFLOW_TRACKING_URI})"

## Evidently data validator
zenml data-validator describe evidently_data_validator 2>/dev/null || \
  zenml data-validator register evidently_data_validator --flavor=evidently
echo "  ✓ Data validator: evidently_data_validator"


# --------------------------------------
# Register ZenML AWS stack
# --------------------------------------

echo ""
echo "==> Assembling AWS ZenML stack..."

zenml stack describe aws_stack 2>/dev/null || \
  zenml stack register aws_stack \
    --orchestrator=sagemaker_orch \
    --artifact-store=s3_store \
    --container-registry=ecr_registry \
    --experiment-tracker=mlflow_tracker \
    --data-validator=evidently_data_validator \
    --set
echo "  ✓ Stack: aws_stack"

# ----------------------------------------------------------------
# Authenticate ZenML stack components with AWS service connector
# ----------------------------------------------------------------

echo ""
echo "==> Connecting AWS stack components to service connector..."

zenml artifact-store connect s3_store \
  --connector "${ZENML_AWS_CONNECTOR_NAME}" \
  --resource-id "s3://${ZENML_ARTIFACT_BUCKET}" \
  >/dev/null
echo "  ✓ Authenticated artifact store (s3_store)"

zenml container-registry connect ecr_registry \
  --connector "${ZENML_AWS_CONNECTOR_NAME}" \
  --resource-id "${ECR_URI}" \
  >/dev/null
echo "  ✓ Authenticated container registry (ecr_registry)"

zenml orchestrator connect sagemaker_orch \
  --connector "${ZENML_AWS_CONNECTOR_NAME}" \
  >/dev/null
echo "  ✓ Authenticated orchestrator (sagemaker_orch)"

echo ""
echo "🎉 AWS Stack Setup complete"
echo ""
