#!/usr/bin/env bash
# infra/local/setup_stacks.sh
#
# Idempotent ZenML stack setup script for local development.
#
# Usage:
#   bash infra/local/setup_stacks.sh

set -euo pipefail

ARTIFACT_STORE_PATH_RAW="${ZENML_ARTIFACT_STORE_PATH:-./data/artifact_store}"

DEFAULT_LOCAL_STACK_NAME="local_stack"
DEFAULT_LOCAL_ORCHESTRATOR_NAME="local_orchestrator"
DEFAULT_LOCAL_ARTIFACT_STORE_NAME="project_store"
DEFAULT_LOCAL_EXPERIMENT_TRACKER_NAME="mlflow_tracker"
DEFAULT_LOCAL_DATA_VALIDATOR_NAME="evidently_data_validator"

ZENML_LOCAL_STACK_NAME="${ZENML_LOCAL_STACK_NAME:-${DEFAULT_LOCAL_STACK_NAME}}"
ZENML_LOCAL_ORCHESTRATOR_NAME="${ZENML_LOCAL_ORCHESTRATOR_NAME:-${DEFAULT_LOCAL_ORCHESTRATOR_NAME}}"
ZENML_LOCAL_ARTIFACT_STORE_NAME="${ZENML_LOCAL_ARTIFACT_STORE_NAME:-${DEFAULT_LOCAL_ARTIFACT_STORE_NAME}}"
ZENML_LOCAL_EXPERIMENT_TRACKER_NAME="${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME:-${DEFAULT_LOCAL_EXPERIMENT_TRACKER_NAME}}"
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

ARTIFACT_STORE_PATH_RAW="$(strip_wrapping_quotes "${ARTIFACT_STORE_PATH_RAW}")"
ARTIFACT_STORE_PATH="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${ARTIFACT_STORE_PATH_RAW}")"

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
echo "==> Registering custom artifact store..."
if zenml artifact-store describe "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" >/dev/null 2>&1; then
  zenml artifact-store update "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" --path="$ARTIFACT_STORE_PATH"
else
  zenml artifact-store register "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    --flavor=local \
    --path="$ARTIFACT_STORE_PATH"
fi
echo "  ✓ Artifact store: ${ZENML_LOCAL_ARTIFACT_STORE_NAME}"


# MLflow experiment tracker (requires MLFLOW_TRACKING_URI env var)
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


# Evidently data validator
zenml data-validator describe "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" 2>/dev/null || \
  zenml data-validator register "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" --flavor=evidently
echo "  ✓ Data validator: ${ZENML_LOCAL_DATA_VALIDATOR_NAME}"


# --------------------------------------
# Register ZenML local stack
# --------------------------------------
echo ""
echo "==> Registering local ZenML stack..."

if zenml stack describe "${ZENML_LOCAL_STACK_NAME}" >/dev/null 2>&1; then
  zenml stack update "${ZENML_LOCAL_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    -e "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}"
  zenml stack set "${ZENML_LOCAL_STACK_NAME}"
else
  zenml stack register "${ZENML_LOCAL_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    -e "${ZENML_LOCAL_EXPERIMENT_TRACKER_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" \
    --set
fi
echo "  ✓ Stack: ${ZENML_LOCAL_STACK_NAME}"

echo ""
echo "🎉 Local Stack Setup Complete"
echo ""
