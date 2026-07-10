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
#   ZENML_BATCH_DDB_TABLE_NAME       — default: zenml-batch-predictions
#   ZENML_BATCH_DDB_PARTITION_KEY_NAME — default: id
#   ZENML_EXEC_ROLE_NAME      — default: zenml-execution-role
#   ZENML_EXEC_ROLE_POLICY_NAME — default: zenml-execution-policy
#   ZENML_SCHEDULER_ROLE_NAME — default: zenml-scheduler-role
#   ZENML_SCHEDULER_ROLE_POLICY_NAME — default: zenml-scheduler-policy
#   ZENML_SAGEMAKER_STEP_OPERATOR_NAME — default: sagemaker_step_operator
#   ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE — default: ml.m5.xlarge
#   ZENML_SAGEMAKER_EXPERIMENT_NAME — optional SageMaker experiment for step operator jobs
#   ZENML_AWS_STACK_NAME      — default: aws_stack
#   ZENML_AWS_ORCHESTRATOR_NAME — default: sagemaker_orchestrator
#   ZENML_AWS_ARTIFACT_STORE_NAME — default: s3_store
#   ZENML_AWS_CONTAINER_REGISTRY_NAME — default: ecr_registry
#   ZENML_AWS_EXPERIMENT_TRACKER_NAME — default: mlflow_tracker
#   ZENML_AWS_DATA_VALIDATOR_NAME — default: evidently_data_validator
#
# Usage:
#   export AWS_ACCOUNT_ID=123456789012
#   export AWS_REGION=us-east-1
#   bash infra/aws/setup_stacks.sh

set -euo pipefail

: "${AWS_ACCOUNT_ID:?ERROR: AWS_ACCOUNT_ID is not set}"
: "${AWS_REGION:?ERROR: AWS_REGION is not set}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

EXEC_ROLE_NAME="${ZENML_EXEC_ROLE_NAME:-zenml-execution-role}"
EXEC_ROLE_POLICY_NAME="${ZENML_EXEC_ROLE_POLICY_NAME:-zenml-execution-policy}"
SCHEDULER_ROLE_NAME="${ZENML_SCHEDULER_ROLE_NAME:-zenml-scheduler-role}"
SCHEDULER_ROLE_POLICY_NAME="${ZENML_SCHEDULER_ROLE_POLICY_NAME:-zenml-scheduler-policy}"

ZENML_EXECUTION_ROLE_ARN="${ZENML_EXECUTION_ROLE_ARN:-arn:aws:iam::${AWS_ACCOUNT_ID}:role/${EXEC_ROLE_NAME}}"
export ZENML_EXECUTION_ROLE_ARN
ZENML_SCHEDULER_ROLE_ARN="${ZENML_SCHEDULER_ROLE_ARN:-arn:aws:iam::${AWS_ACCOUNT_ID}:role/${SCHEDULER_ROLE_NAME}}"
export ZENML_SCHEDULER_ROLE_ARN

DEFAULT_ARTIFACT_BUCKET="zenml-artifacts"
DEFAULT_CHECKPOINT_BUCKET="zenml-checkpoints"
DEFAULT_DATA_BUCKET="zenml-data"
DEFAULT_PREDICTIONS_BUCKET="zenml-predictions"
DEFAULT_ECR_REPOSITORY="zenml"
DEFAULT_BATCH_DDB_TABLE_NAME="zenml-batch-predictions"
DEFAULT_BATCH_DDB_PARTITION_KEY_NAME="id"
DEFAULT_SAGEMAKER_STEP_OPERATOR_NAME="sagemaker_step_operator"
DEFAULT_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE="ml.m5.xlarge"
DEFAULT_AWS_STACK_NAME="aws_stack"
DEFAULT_AWS_ORCHESTRATOR_NAME="sagemaker_orchestrator"
DEFAULT_AWS_ARTIFACT_STORE_NAME="s3_store"
DEFAULT_AWS_CONTAINER_REGISTRY_NAME="ecr_registry"
DEFAULT_AWS_EXPERIMENT_TRACKER_NAME="mlflow_tracker"
DEFAULT_AWS_DATA_VALIDATOR_NAME="evidently_data_validator"

ZENML_ARTIFACT_BUCKET="${ZENML_ARTIFACT_BUCKET:-${DEFAULT_ARTIFACT_BUCKET}}"
ZENML_CHECKPOINT_BUCKET="${ZENML_CHECKPOINT_BUCKET:-${DEFAULT_CHECKPOINT_BUCKET}}"
ZENML_DATA_BUCKET="${ZENML_DATA_BUCKET:-${DEFAULT_DATA_BUCKET}}"
ZENML_PREDICTIONS_BUCKET="${ZENML_PREDICTIONS_BUCKET:-${DEFAULT_PREDICTIONS_BUCKET}}"
ZENML_ECR_REPOSITORY="${ZENML_ECR_REPOSITORY:-${DEFAULT_ECR_REPOSITORY}}"
ZENML_AWS_CONNECTOR_NAME="${ZENML_AWS_CONNECTOR_NAME:-aws_connector}"
ZENML_BATCH_DDB_TABLE_NAME="${ZENML_BATCH_DDB_TABLE_NAME:-${DEFAULT_BATCH_DDB_TABLE_NAME}}"
ZENML_BATCH_DDB_PARTITION_KEY_NAME="${ZENML_BATCH_DDB_PARTITION_KEY_NAME:-${DEFAULT_BATCH_DDB_PARTITION_KEY_NAME}}"
ZENML_BATCH_DDB_TABLE_ARN="arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${ZENML_BATCH_DDB_TABLE_NAME}"
ZENML_SAGEMAKER_STEP_OPERATOR_NAME="${ZENML_SAGEMAKER_STEP_OPERATOR_NAME:-${DEFAULT_SAGEMAKER_STEP_OPERATOR_NAME}}"
ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE="${ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE:-${DEFAULT_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE}}"
ZENML_SAGEMAKER_EXPERIMENT_NAME="${ZENML_SAGEMAKER_EXPERIMENT_NAME:-}"
ZENML_AWS_STACK_NAME="${ZENML_AWS_STACK_NAME:-${DEFAULT_AWS_STACK_NAME}}"
ZENML_AWS_ORCHESTRATOR_NAME="${ZENML_AWS_ORCHESTRATOR_NAME:-${DEFAULT_AWS_ORCHESTRATOR_NAME}}"
ZENML_AWS_ARTIFACT_STORE_NAME="${ZENML_AWS_ARTIFACT_STORE_NAME:-${DEFAULT_AWS_ARTIFACT_STORE_NAME}}"
ZENML_AWS_CONTAINER_REGISTRY_NAME="${ZENML_AWS_CONTAINER_REGISTRY_NAME:-${DEFAULT_AWS_CONTAINER_REGISTRY_NAME}}"
ZENML_AWS_EXPERIMENT_TRACKER_NAME="${ZENML_AWS_EXPERIMENT_TRACKER_NAME:-${DEFAULT_AWS_EXPERIMENT_TRACKER_NAME}}"
ZENML_AWS_DATA_VALIDATOR_NAME="${ZENML_AWS_DATA_VALIDATOR_NAME:-${DEFAULT_AWS_DATA_VALIDATOR_NAME}}"

EXEC_ASSUME_ROLE_POLICY_DOCUMENT="$(cat <<EOF
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

EXEC_ROLE_POLICY_DOCUMENT="$(cat <<EOF
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
        "sagemaker:CreateExperiment",
        "sagemaker:DescribeExperiment",
        "sagemaker:CreateTrial",
        "sagemaker:DescribeTrial",
        "sagemaker:CreateTrialComponent",
        "sagemaker:DescribeTrialComponent",
        "sagemaker:AssociateTrialComponent",
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
        "dynamodb:DescribeTimeToLive"
      ],
      "Resource": "${ZENML_BATCH_DDB_TABLE_ARN}"
    },
    {
      "Sid": "EventBridgeSchedulerAccess",
      "Effect": "Allow",
      "Action": [
        "scheduler:ListSchedules",
        "scheduler:GetSchedule",
        "scheduler:CreateSchedule",
        "scheduler:UpdateSchedule",
        "scheduler:DeleteSchedule"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SchedulerPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "${ZENML_SCHEDULER_ROLE_ARN}",
      "Condition": {
        "StringLike": {
          "iam:PassedToService": "scheduler.amazonaws.com"
        }
      }
    },
    {
      "Sid": "IAMPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::*:role/${EXEC_ROLE_NAME}",
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

## IAM scheduler role
SCHEDULER_ASSUME_ROLE_POLICY_DOCUMENT="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "${ZENML_EXECUTION_ROLE_ARN}",
        "Service": [
          "scheduler.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)"

SCHEDULER_POLICY_DOCUMENT="$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerSchedulerExecution",
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
      "Sid": "EventBridgeSchedulerAccess",
      "Effect": "Allow",
      "Action": [
        "scheduler:ListSchedules",
        "scheduler:GetSchedule",
        "scheduler:CreateSchedule",
        "scheduler:UpdateSchedule",
        "scheduler:DeleteSchedule"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassSageMakerExecutionRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "${ZENML_EXECUTION_ROLE_ARN}",
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
if aws iam get-role --role-name "${EXEC_ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Role ${EXEC_ROLE_NAME} already exists, skipping create"
else
  aws iam create-role \
    --role-name "${EXEC_ROLE_NAME}" \
    --assume-role-policy-document "${EXEC_ASSUME_ROLE_POLICY_DOCUMENT}" \
    >/dev/null
fi

aws iam put-role-policy \
  --role-name "${EXEC_ROLE_NAME}" \
  --policy-name "${EXEC_ROLE_POLICY_NAME}" \
  --policy-document "${EXEC_ROLE_POLICY_DOCUMENT}" \
  >/dev/null

ZENML_EXECUTION_ROLE_ARN="$(aws iam get-role --role-name "${EXEC_ROLE_NAME}" --query 'Role.Arn' --output text)"
export ZENML_EXECUTION_ROLE_ARN
echo "  ✓ IAM role ready: ${ZENML_EXECUTION_ROLE_ARN}"

echo ""
echo "==> Creating IAM scheduler role (idempotent)..."
if aws iam get-role --role-name "${SCHEDULER_ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Role ${SCHEDULER_ROLE_NAME} already exists, skipping create"
else
  aws iam create-role \
    --role-name "${SCHEDULER_ROLE_NAME}" \
    --assume-role-policy-document "${SCHEDULER_ASSUME_ROLE_POLICY_DOCUMENT}" \
    >/dev/null
fi

aws iam update-assume-role-policy \
  --role-name "${SCHEDULER_ROLE_NAME}" \
  --policy-document "${SCHEDULER_ASSUME_ROLE_POLICY_DOCUMENT}" \
  >/dev/null

aws iam put-role-policy \
  --role-name "${SCHEDULER_ROLE_NAME}" \
  --policy-name "${SCHEDULER_ROLE_POLICY_NAME}" \
  --policy-document "${SCHEDULER_POLICY_DOCUMENT}" \
  >/dev/null

ZENML_SCHEDULER_ROLE_ARN="$(aws iam get-role --role-name "${SCHEDULER_ROLE_NAME}" --query 'Role.Arn' --output text)"
export ZENML_SCHEDULER_ROLE_ARN
echo "  ✓ Scheduler role ready: ${ZENML_SCHEDULER_ROLE_ARN}"

## S3 buckets
echo ""
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
echo ""
echo "==> Creating ECR repository (idempotent)..."
aws ecr describe-repositories --repository-names "${ZENML_ECR_REPOSITORY}" --region "$AWS_REGION" 2>/dev/null || \
  aws ecr create-repository \
    --repository-name "${ZENML_ECR_REPOSITORY}" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true
echo "  ✓ ECR repository ready"

## DynamoDB table
echo ""
echo "==> Creating DynamoDB table (idempotent)..."
if aws dynamodb describe-table --table-name "${ZENML_BATCH_DDB_TABLE_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "  Table ${ZENML_BATCH_DDB_TABLE_NAME} already exists, skipping create"
else
  aws dynamodb create-table \
    --table-name "${ZENML_BATCH_DDB_TABLE_NAME}" \
    --attribute-definitions "AttributeName=${ZENML_BATCH_DDB_PARTITION_KEY_NAME},AttributeType=S" \
    --key-schema "AttributeName=${ZENML_BATCH_DDB_PARTITION_KEY_NAME},KeyType=HASH" \
    --billing-mode PAY_PER_REQUEST \
    --region "${AWS_REGION}" \
    >/dev/null
  aws dynamodb wait table-exists --table-name "${ZENML_BATCH_DDB_TABLE_NAME}" --region "${AWS_REGION}"
fi

ttl_status="$(aws dynamodb describe-time-to-live \
  --table-name "${ZENML_BATCH_DDB_TABLE_NAME}" \
  --region "${AWS_REGION}" \
  --query "TimeToLiveDescription.TimeToLiveStatus" \
  --output text 2>/dev/null || echo "DISABLED")"
if [ "${ttl_status}" = "ENABLED" ] || [ "${ttl_status}" = "ENABLING" ]; then
  echo "  TTL already ${ttl_status} on ${ZENML_BATCH_DDB_TABLE_NAME}"
else
  aws dynamodb update-time-to-live \
    --table-name "${ZENML_BATCH_DDB_TABLE_NAME}" \
    --time-to-live-specification "Enabled=true,AttributeName=updated_at" \
    --region "${AWS_REGION}" \
    >/dev/null
  echo "  Enabled TTL on attribute updated_at"
fi
echo "  ✓ DynamoDB table ready: ${ZENML_BATCH_DDB_TABLE_NAME} (PK=${ZENML_BATCH_DDB_PARTITION_KEY_NAME})"


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

## Orchestrator (SageMaker Pipelines)
if zenml orchestrator describe "${ZENML_AWS_ORCHESTRATOR_NAME}" >/dev/null 2>&1; then
  zenml orchestrator update "${ZENML_AWS_ORCHESTRATOR_NAME}" \
    --region="${AWS_REGION}" \
    --execution_role="${ZENML_EXECUTION_ROLE_ARN}" \
    --scheduler_role="${ZENML_SCHEDULER_ROLE_ARN}"
else
  zenml orchestrator register "${ZENML_AWS_ORCHESTRATOR_NAME}" \
    --flavor=sagemaker \
    --region="${AWS_REGION}" \
    --execution_role="${ZENML_EXECUTION_ROLE_ARN}" \
    --scheduler_role="${ZENML_SCHEDULER_ROLE_ARN}"
fi
echo "  ✓ Orchestrator: ${ZENML_AWS_ORCHESTRATOR_NAME} (scheduler_role=${ZENML_SCHEDULER_ROLE_ARN})"

## Artifact store
zenml artifact-store describe "${ZENML_AWS_ARTIFACT_STORE_NAME}" 2>/dev/null || \
  zenml artifact-store register "${ZENML_AWS_ARTIFACT_STORE_NAME}" \
    --flavor=s3 \
    --path="s3://${ZENML_ARTIFACT_BUCKET}/"
echo "  ✓ Artifact store: ${ZENML_AWS_ARTIFACT_STORE_NAME}"

## Container registry
zenml container-registry describe "${ZENML_AWS_CONTAINER_REGISTRY_NAME}" 2>/dev/null || \
  zenml container-registry register "${ZENML_AWS_CONTAINER_REGISTRY_NAME}" \
    --flavor=aws \
    --uri="${ECR_URI}"
echo "  ✓ Container registry: ${ZENML_AWS_CONTAINER_REGISTRY_NAME}"

## MLflow experiment tracker (requires MLFLOW_TRACKING_URI env var)
: "${MLFLOW_TRACKING_URI:=http://localhost:5000}"
zenml experiment-tracker describe "${ZENML_AWS_EXPERIMENT_TRACKER_NAME}" 2>/dev/null || \
  zenml experiment-tracker register "${ZENML_AWS_EXPERIMENT_TRACKER_NAME}" \
    --flavor=mlflow \
    --tracking_uri="${MLFLOW_TRACKING_URI}" \
    --tracking_username="${MLFLOW_TRACKING_USERNAME:-}" \
    --tracking_password="${MLFLOW_TRACKING_PASSWORD:-}"
echo "  ✓ Experiment tracker: ${ZENML_AWS_EXPERIMENT_TRACKER_NAME} (uri=${MLFLOW_TRACKING_URI})"

## Evidently data validator
zenml data-validator describe "${ZENML_AWS_DATA_VALIDATOR_NAME}" 2>/dev/null || \
  zenml data-validator register "${ZENML_AWS_DATA_VALIDATOR_NAME}" --flavor=evidently
echo "  ✓ Data validator: ${ZENML_AWS_DATA_VALIDATOR_NAME}"

## SageMaker step operator
if zenml step-operator describe "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}" >/dev/null 2>&1; then
  if [ -n "${ZENML_SAGEMAKER_EXPERIMENT_NAME}" ]; then
    zenml step-operator update "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}" \
      --role="${ZENML_EXECUTION_ROLE_ARN}" \
      --instance_type="${ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE}" \
      --experiment_name="${ZENML_SAGEMAKER_EXPERIMENT_NAME}"
  else
    zenml step-operator update "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}" \
      --role="${ZENML_EXECUTION_ROLE_ARN}" \
      --instance_type="${ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE}"
  fi
else
  if [ -n "${ZENML_SAGEMAKER_EXPERIMENT_NAME}" ]; then
    zenml step-operator register "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}" \
      --flavor=sagemaker \
      --role="${ZENML_EXECUTION_ROLE_ARN}" \
      --instance_type="${ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE}" \
      --experiment_name="${ZENML_SAGEMAKER_EXPERIMENT_NAME}"
  else
    zenml step-operator register "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}" \
      --flavor=sagemaker \
      --role="${ZENML_EXECUTION_ROLE_ARN}" \
      --instance_type="${ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE}"
  fi
fi
echo "  ✓ Step operator: ${ZENML_SAGEMAKER_STEP_OPERATOR_NAME} (instance_type=${ZENML_SAGEMAKER_STEP_OPERATOR_INSTANCE_TYPE})"


# --------------------------------------
# Register ZenML AWS stack
# --------------------------------------

echo ""
echo "==> Assembling AWS ZenML stack..."
if zenml stack describe "${ZENML_AWS_STACK_NAME}" >/dev/null 2>&1; then
  zenml stack update "${ZENML_AWS_STACK_NAME}" \
    -o "${ZENML_AWS_ORCHESTRATOR_NAME}" \
    -a "${ZENML_AWS_ARTIFACT_STORE_NAME}" \
    -c "${ZENML_AWS_CONTAINER_REGISTRY_NAME}" \
    -e "${ZENML_AWS_EXPERIMENT_TRACKER_NAME}" \
    -dv "${ZENML_AWS_DATA_VALIDATOR_NAME}" \
    -s "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}"
else
  zenml stack register "${ZENML_AWS_STACK_NAME}" \
    -o "${ZENML_AWS_ORCHESTRATOR_NAME}" \
    -a "${ZENML_AWS_ARTIFACT_STORE_NAME}" \
    -c "${ZENML_AWS_CONTAINER_REGISTRY_NAME}" \
    -e "${ZENML_AWS_EXPERIMENT_TRACKER_NAME}" \
    -dv "${ZENML_AWS_DATA_VALIDATOR_NAME}" \
    -s "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}" \
    --set
fi

echo "  ✓ Stack: ${ZENML_AWS_STACK_NAME}"

# ----------------------------------------------------------------
# Authenticate ZenML stack components with AWS service connector
# ----------------------------------------------------------------

echo ""
echo "==> Connecting AWS stack components to service connector..."

zenml artifact-store connect "${ZENML_AWS_ARTIFACT_STORE_NAME}" \
  --connector "${ZENML_AWS_CONNECTOR_NAME}" \
  --resource-id "s3://${ZENML_ARTIFACT_BUCKET}" \
  >/dev/null
echo "  ✓ Authenticated artifact store (${ZENML_AWS_ARTIFACT_STORE_NAME})"

zenml container-registry connect "${ZENML_AWS_CONTAINER_REGISTRY_NAME}" \
  --connector "${ZENML_AWS_CONNECTOR_NAME}" \
  --resource-id "${ECR_URI}" \
  >/dev/null
echo "  ✓ Authenticated container registry (${ZENML_AWS_CONTAINER_REGISTRY_NAME})"

zenml orchestrator connect "${ZENML_AWS_ORCHESTRATOR_NAME}" \
  --connector "${ZENML_AWS_CONNECTOR_NAME}" \
  >/dev/null
echo "  ✓ Authenticated orchestrator (${ZENML_AWS_ORCHESTRATOR_NAME})"

zenml step-operator connect "${ZENML_SAGEMAKER_STEP_OPERATOR_NAME}" \
  --connector "${ZENML_AWS_CONNECTOR_NAME}" \
  >/dev/null
echo "  ✓ Authenticated step operator (${ZENML_SAGEMAKER_STEP_OPERATOR_NAME})"

echo ""
echo "🎉 AWS Stack Setup complete"
echo ""
