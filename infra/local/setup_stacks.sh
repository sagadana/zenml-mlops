#!/usr/bin/env bash
# infra/local/setup_stacks.sh
#
# Idempotent ZenML stack setup script for local development.
#
# Usage:
#   bash infra/local/setup_stacks.sh

set -euo pipefail

ARTIFACT_STORE_PATH_RAW="${ZENML_ARTIFACT_STORE_PATH:-s3://${ZENML_ARTIFACT_BUCKET:-aips-recs-zenml-artifacts}/}"
ARTIFACT_STORE_HOST_ENDPOINT_URL_RAW="${ZENML_ARTIFACT_STORE_ENDPOINT_URL_HOST:-${SEAWEEDFS_S3_ENDPOINT_URL:-http://localtest.me:8333}}"
ARTIFACT_STORE_DOCKER_ENDPOINT_URL_RAW="${ZENML_ARTIFACT_STORE_ENDPOINT_URL_DOCKER:-${ZENML_ARTIFACT_STORE_ENDPOINT_URL:-http://host.docker.internal:${SEAWEEDFS_S3_PORT:-8333}}}"
S3_ACCESS_KEY_ID_RAW="${SEAWEEDFS_ACCESS_KEY_ID:-admin}"
S3_SECRET_ACCESS_KEY_RAW="${SEAWEEDFS_SECRET_ACCESS_KEY:-secret}"

DEFAULT_LOCAL_STACK_NAME="local_stack"
DEFAULT_LOCAL_DOCKER_STACK_NAME="local_docker_stack"
DEFAULT_LOCAL_ORCHESTRATOR_NAME="local_orchestrator"
DEFAULT_LOCAL_ARTIFACT_STORE_NAME="local_s3_store"
DEFAULT_LOCAL_DOCKER_ARTIFACT_STORE_NAME="local_s3_store_docker"
DEFAULT_LOCAL_S3_SECRET_NAME="local_s3_auth_secret"
DEFAULT_LOCAL_EXPERIMENT_TRACKER_NAME="mlflow_tracker"
DEFAULT_LOCAL_MODEL_REGISTRY_NAME="mlflow_model_registry"
DEFAULT_LOCAL_DATA_VALIDATOR_NAME="evidently_data_validator"

ZENML_LOCAL_STACK_NAME="${ZENML_LOCAL_STACK_NAME:-${DEFAULT_LOCAL_STACK_NAME}}"
ZENML_LOCAL_DOCKER_STACK_NAME="${ZENML_LOCAL_DOCKER_STACK_NAME:-${DEFAULT_LOCAL_DOCKER_STACK_NAME}}"
ZENML_LOCAL_ORCHESTRATOR_NAME="${ZENML_LOCAL_ORCHESTRATOR_NAME:-${DEFAULT_LOCAL_ORCHESTRATOR_NAME}}"
ZENML_LOCAL_ARTIFACT_STORE_NAME="${ZENML_LOCAL_ARTIFACT_STORE_NAME:-${DEFAULT_LOCAL_ARTIFACT_STORE_NAME}}"
ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME="${ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME:-${DEFAULT_LOCAL_DOCKER_ARTIFACT_STORE_NAME}}"
ZENML_LOCAL_S3_SECRET_NAME="${ZENML_LOCAL_S3_SECRET_NAME:-${DEFAULT_LOCAL_S3_SECRET_NAME}}"
ZENML_LOCAL_EXPERIMENT_TRACKER_NAME="${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME:-${DEFAULT_LOCAL_EXPERIMENT_TRACKER_NAME}}"
ZENML_LOCAL_MODEL_REGISTRY_NAME="${ZENML_LOCAL_MODEL_REGISTRY_NAME:-${DEFAULT_LOCAL_MODEL_REGISTRY_NAME}}"
ZENML_LOCAL_DATA_VALIDATOR_NAME="${ZENML_LOCAL_DATA_VALIDATOR_NAME:-${DEFAULT_LOCAL_DATA_VALIDATOR_NAME}}"

strip_wrapping_quotes() {
  local value="$1"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  printf '%s' "$value"
}

ARTIFACT_STORE_PATH="$(strip_wrapping_quotes "${ARTIFACT_STORE_PATH_RAW}")"
ARTIFACT_STORE_HOST_ENDPOINT_URL="$(strip_wrapping_quotes "${ARTIFACT_STORE_HOST_ENDPOINT_URL_RAW}")"
ARTIFACT_STORE_DOCKER_ENDPOINT_URL="$(strip_wrapping_quotes "${ARTIFACT_STORE_DOCKER_ENDPOINT_URL_RAW}")"
ARTIFACT_STORE_HOST_CLIENT_KWARGS="$(python3 -c 'import json,sys; print(json.dumps({"endpoint_url": sys.argv[1]}))' "${ARTIFACT_STORE_HOST_ENDPOINT_URL}")"
ARTIFACT_STORE_DOCKER_CLIENT_KWARGS="$(python3 -c 'import json,sys; print(json.dumps({"endpoint_url": sys.argv[1]}))' "${ARTIFACT_STORE_DOCKER_ENDPOINT_URL}")"
S3_ACCESS_KEY_ID="$(strip_wrapping_quotes "${S3_ACCESS_KEY_ID_RAW}")"
S3_SECRET_ACCESS_KEY="$(strip_wrapping_quotes "${S3_SECRET_ACCESS_KEY_RAW}")"
S3_AUTH_SECRET_VALUES="$(python3 -c 'import json,sys; print(json.dumps({"access_key_id": sys.argv[1], "secret_access_key": sys.argv[2]}))' "${S3_ACCESS_KEY_ID}" "${S3_SECRET_ACCESS_KEY}")"

# --------------------------------------
# Register ZenML stack components
# --------------------------------------

# Local Docker orchestrator
echo ""
echo "==> Registering local orchestrator..."

if zenml orchestrator describe "${ZENML_LOCAL_ORCHESTRATOR_NAME}" >/dev/null 2>&1; then
  true # do nothing, orchestrator already exists
else
  zenml orchestrator register "${ZENML_LOCAL_ORCHESTRATOR_NAME}" --flavor=local
fi
echo "  ✓ Orchestrator: ${ZENML_LOCAL_ORCHESTRATOR_NAME}"

# Artifact store
echo ""
echo "==> Registering local S3 artifact stores (SeaweedFS)..."

if zenml secret get "${ZENML_LOCAL_S3_SECRET_NAME}" >/dev/null 2>&1; then
  zenml secret update "${ZENML_LOCAL_S3_SECRET_NAME}" \
    --values="${S3_AUTH_SECRET_VALUES}"
else
  zenml secret create "${ZENML_LOCAL_S3_SECRET_NAME}" \
    --values="${S3_AUTH_SECRET_VALUES}"
fi
echo "  ✓ S3 auth secret: ${ZENML_LOCAL_S3_SECRET_NAME}"

if zenml artifact-store describe "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" >/dev/null 2>&1; then
  zenml artifact-store update "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    --path="${ARTIFACT_STORE_PATH}" \
    --client_kwargs="${ARTIFACT_STORE_HOST_CLIENT_KWARGS}" \
    --authentication_secret="${ZENML_LOCAL_S3_SECRET_NAME}"
else
  zenml artifact-store register "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    --flavor=s3 \
    --path="${ARTIFACT_STORE_PATH}" \
    --client_kwargs="${ARTIFACT_STORE_HOST_CLIENT_KWARGS}" \
    --authentication_secret="${ZENML_LOCAL_S3_SECRET_NAME}"
fi
echo "  ✓ Artifact store (host): ${ZENML_LOCAL_ARTIFACT_STORE_NAME} (endpoint=${ARTIFACT_STORE_HOST_ENDPOINT_URL})"

if zenml artifact-store describe "${ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME}" >/dev/null 2>&1; then
  zenml artifact-store update "${ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME}" \
    --path="${ARTIFACT_STORE_PATH}" \
    --client_kwargs="${ARTIFACT_STORE_DOCKER_CLIENT_KWARGS}" \
    --authentication_secret="${ZENML_LOCAL_S3_SECRET_NAME}"
else
  zenml artifact-store register "${ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME}" \
    --flavor=s3 \
    --path="${ARTIFACT_STORE_PATH}" \
    --client_kwargs="${ARTIFACT_STORE_DOCKER_CLIENT_KWARGS}" \
    --authentication_secret="${ZENML_LOCAL_S3_SECRET_NAME}"
fi
echo "  ✓ Artifact store (docker): ${ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME} (endpoint=${ARTIFACT_STORE_DOCKER_ENDPOINT_URL})"


# MLflow experiment tracker (requires MLFLOW_TRACKING_URI env var)
echo ""
echo "==> Registering local MLflow experiment tracker..."

: "${MLFLOW_TRACKING_URI:=http://localhost:5000}"
MLFLOW_TRACKING_URI="$(strip_wrapping_quotes "${MLFLOW_TRACKING_URI}")"
MLFLOW_TRACKING_USERNAME="$(strip_wrapping_quotes "${MLFLOW_TRACKING_USERNAME:-}")"
MLFLOW_TRACKING_PASSWORD="$(strip_wrapping_quotes "${MLFLOW_TRACKING_PASSWORD:-}")"

if zenml experiment-tracker describe "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" >/dev/null 2>&1; then
  zenml experiment-tracker update "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    --tracking_uri="${MLFLOW_TRACKING_URI}" \
    --tracking_username="${MLFLOW_TRACKING_USERNAME}" \
    --tracking_password="${MLFLOW_TRACKING_PASSWORD}"
else
  zenml experiment-tracker register "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    --flavor=mlflow \
    --tracking_uri="${MLFLOW_TRACKING_URI}" \
    --tracking_username="${MLFLOW_TRACKING_USERNAME}" \
    --tracking_password="${MLFLOW_TRACKING_PASSWORD}"
fi
echo "  ✓ Experiment tracker: ${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME} (uri=${MLFLOW_TRACKING_URI})"


# MLflow model registry (requires MLflow experiment tracker in the stack)
echo ""
echo "==> Registering local MLflow model registry..."

zenml model-registry describe "${ZENML_LOCAL_MODEL_REGISTRY_NAME}" 2>/dev/null || \
  zenml model-registry register "${ZENML_LOCAL_MODEL_REGISTRY_NAME}" --flavor=mlflow
echo "  ✓ Model registry: ${ZENML_LOCAL_MODEL_REGISTRY_NAME}"


# Evidently data validator
echo ""
echo "==> Registering local Evidently data validator..."

zenml data-validator describe "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" 2>/dev/null || \
  zenml data-validator register "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" --flavor=evidently
echo "  ✓ Data validator: ${ZENML_LOCAL_DATA_VALIDATOR_NAME}"


# --------------------------------------
# Register ZenML local stack
# --------------------------------------
echo ""
echo "==> Registering local ZenML stacks..."

if zenml stack describe "${ZENML_LOCAL_STACK_NAME}" >/dev/null 2>&1; then
  zenml stack update "${ZENML_LOCAL_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    -e "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    -r "${ZENML_LOCAL_MODEL_REGISTRY_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}"
  zenml stack set "${ZENML_LOCAL_STACK_NAME}"
else
  zenml stack register "${ZENML_LOCAL_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    -e "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    -r "${ZENML_LOCAL_MODEL_REGISTRY_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" \
    --set
fi
echo "  ✓ Stack (host): ${ZENML_LOCAL_STACK_NAME}"

if zenml stack describe "${ZENML_LOCAL_DOCKER_STACK_NAME}" >/dev/null 2>&1; then
  zenml stack update "${ZENML_LOCAL_DOCKER_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME}" \
    -e "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    -r "${ZENML_LOCAL_MODEL_REGISTRY_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}"
else
  zenml stack register "${ZENML_LOCAL_DOCKER_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_DOCKER_ARTIFACT_STORE_NAME}" \
    -e "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    -r "${ZENML_LOCAL_MODEL_REGISTRY_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}"
fi
echo "  ✓ Stack (docker): ${ZENML_LOCAL_DOCKER_STACK_NAME}"

echo ""
echo "🎉 Local Stack Setup Complete"
echo ""
