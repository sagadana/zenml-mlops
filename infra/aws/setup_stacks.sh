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
# Usage:
#   export AWS_ACCOUNT_ID=123456789012
#   export AWS_REGION=us-east-1
#   bash infra/aws/setup_stacks.sh

set -euo pipefail

: "${AWS_ACCOUNT_ID:?ERROR: AWS_ACCOUNT_ID is not set}"
: "${AWS_REGION:?ERROR: AWS_REGION is not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IAM_POLICY_FILE="${SCRIPT_DIR}/iam_policy.json"
ROLE_NAME="${ZENML_EXEC_ROLE_NAME:-aips-recs-zenml-execution-role}"
ROLE_POLICY_NAME="${ZENML_EXEC_ROLE_POLICY_NAME:-aips-recs-zenml-execution-policy}"

if [ ! -f "${IAM_POLICY_FILE}" ]; then
  echo "ERROR: IAM policy file not found at ${IAM_POLICY_FILE}"
  exit 1
fi

ZENML_EXECUTION_ROLE_ARN="${ZENML_EXECUTION_ROLE_ARN:-arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}}"
export ZENML_EXECUTION_ROLE_ARN

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ARTIFACT_BUCKET="aips-zenml-artifacts"
CHECKPOINT_BUCKET="aips-zenml-checkpoints"
DATA_BUCKET="aips-zenml-data"
PREDICTIONS_BUCKET="aips-zenml-predictions"

echo "==> Installing ZenML integrations..."
zenml integration install aws s3 mlflow sagemaker evidently -y

# -------------------------
# Create AWS resources 
# -------------------------

echo "==> Creating IAM execution role (idempotent)..."
if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "  Role ${ROLE_NAME} already exists, skipping create"
else
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    >/dev/null
fi

aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "${ROLE_POLICY_NAME}" \
  --policy-document "file://${IAM_POLICY_FILE}" \
  >/dev/null

ZENML_EXECUTION_ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query 'Role.Arn' --output text)"
export ZENML_EXECUTION_ROLE_ARN
echo "  ✓ IAM role ready: ${ZENML_EXECUTION_ROLE_ARN}"

echo "==> Creating S3 buckets (idempotent)..."
for bucket in "$ARTIFACT_BUCKET" "$CHECKPOINT_BUCKET" "$DATA_BUCKET" "$PREDICTIONS_BUCKET"; do
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

echo "==> Creating ECR repository (idempotent)..."
aws ecr describe-repositories --repository-names aips-zenml --region "$AWS_REGION" 2>/dev/null || \
  aws ecr create-repository \
    --repository-name aips-zenml \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true
echo "  ✓ ECR repository ready"

# --------------------------------------
# Register ZenML AWS service connector
# --------------------------------------

echo "==> Registering AWS service connector..."

zenml service-connector describe aws_connector 2>/dev/null || \
  zenml service-connector register aws_connector \
    --type aws \
    --auth-method iam-role \
    --role_arn="${ZENML_EXECUTION_ROLE_ARN}" \
    --region="${AWS_REGION}"
echo "  ✓ Service connector ready"

# --------------------------------------
# Register ZenML stack components
# --------------------------------------

echo "==> Registering ZenML stack components..."

# Artifact store
zenml artifact-store describe s3_store 2>/dev/null || \
  zenml artifact-store register s3_store \
    --flavor=s3 \
    --path="s3://${ARTIFACT_BUCKET}/" \
    --connector aws_connector
echo "  ✓ Artifact store: s3_store"

# Container registry
zenml container-registry describe ecr_registry 2>/dev/null || \
  zenml container-registry register ecr_registry \
    --flavor=aws \
    --uri="${ECR_URI}" \
    --connector aws_connector
echo "  ✓ Container registry: ecr_registry"

# Orchestrator (SageMaker Pipelines)
zenml orchestrator describe sagemaker_orch 2>/dev/null || \
  zenml orchestrator register sagemaker_orch \
    --flavor=sagemaker \
    --region="${AWS_REGION}" \
    --execution_role="${ZENML_EXECUTION_ROLE_ARN}"
echo "  ✓ Orchestrator: sagemaker_orch"

# MLflow experiment tracker (requires MLFLOW_TRACKING_URI env var)
: "${MLFLOW_TRACKING_URI:=http://localhost:5000}"
zenml experiment-tracker describe mlflow_tracker 2>/dev/null || \
  zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="${MLFLOW_TRACKING_URI}" \
    --tracking_username="${MLFLOW_TRACKING_USERNAME:-}" \
    --tracking_password="${MLFLOW_TRACKING_PASSWORD:-}"
echo "  ✓ Experiment tracker: mlflow_tracker (uri=${MLFLOW_TRACKING_URI})"

# Evidently data validator
zenml data-validator describe evidently_data_validator 2>/dev/null || \
  zenml data-validator register evidently_data_validator --flavor=evidently
echo "  ✓ Data validator: evidently_data_validator"


# --------------------------------------
# Register ZenML AWS stack
# --------------------------------------

echo "==> Assembling AWS ZenML stack..."

zenml stack describe aws_stack 2>/dev/null || \
  zenml stack register aws_stack \
    --orchestrator=sagemaker_orch \
    --artifact-store=s3_store \
    --container-registry=ecr_registry \
    --experiment-tracker=mlflow_tracker \
    --data-validator=evidently_data_validator
echo "  ✓ Stack: aws_stack"

echo ""
echo "=== AWS Stack Setup complete ==="
echo ""
