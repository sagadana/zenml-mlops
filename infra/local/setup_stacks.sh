#!/usr/bin/env bash
# infra/local/setup_stacks.sh
#
# Idempotent ZenML stack setup script for local development.
#
# Usage:
#   bash infra/local/setup_stacks.sh

set -euo pipefail

ARTIFACT_STORE_PATH_RAW="${ZENML_ARTIFACT_STORE_PATH:-./data/artifact_store}"

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
if zenml orchestrator describe local_docker_orchestrator >/dev/null 2>&1; then
  true
else
  zenml orchestrator register local_docker_orchestrator --flavor=local_docker
fi
echo "  ✓ Orchestrator: local_docker_orchestrator"

# Artifact store
echo ""
echo "==> Registering custom artifact store..."
if zenml artifact-store describe project_store >/dev/null 2>&1; then
  zenml artifact-store update project_store --path="$ARTIFACT_STORE_PATH"
else
  zenml artifact-store register project_store \
    --flavor=local \
    --path="$ARTIFACT_STORE_PATH"
fi
echo "  ✓ Artifact store: project_store"


# MLflow experiment tracker (requires MLFLOW_TRACKING_URI env var)
: "${MLFLOW_TRACKING_URI:=http://localhost:5000}"
MLFLOW_TRACKING_URI="$(strip_wrapping_quotes "${MLFLOW_TRACKING_URI}")"
MLFLOW_TRACKING_USERNAME="$(strip_wrapping_quotes "${MLFLOW_TRACKING_USERNAME:-}")"
MLFLOW_TRACKING_PASSWORD="$(strip_wrapping_quotes "${MLFLOW_TRACKING_PASSWORD:-}")"

if zenml experiment-tracker describe mlflow_tracker >/dev/null 2>&1; then
  zenml experiment-tracker update mlflow_tracker \
    --tracking_uri="${MLFLOW_TRACKING_URI}" \
    --tracking_username="${MLFLOW_TRACKING_USERNAME}" \
    --tracking_password="${MLFLOW_TRACKING_PASSWORD}"
else
  zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="${MLFLOW_TRACKING_URI}" \
    --tracking_username="${MLFLOW_TRACKING_USERNAME}" \
    --tracking_password="${MLFLOW_TRACKING_PASSWORD}"
fi
echo "  ✓ Experiment tracker: mlflow_tracker (uri=${MLFLOW_TRACKING_URI})"


# Evidently data validator
zenml data-validator describe evidently_data_validator 2>/dev/null || \
  zenml data-validator register evidently_data_validator --flavor=evidently
echo "  ✓ Data validator: evidently_data_validator"


# --------------------------------------
# Register ZenML local stack
# --------------------------------------
echo ""
echo "==> Registering local ZenML stack..."

if zenml stack describe local_stack >/dev/null 2>&1; then
  zenml stack update local_stack \
    -o local_docker_orchestrator \
    -a project_store \
    -e mlflow_tracker \
    -dv evidently_data_validator
  zenml stack set local_stack
else
  zenml stack register local_stack \
    -o local_docker_orchestrator \
    -a project_store \
    -e mlflow_tracker \
    -dv evidently_data_validator \
    --set
fi
echo "  ✓ Stack: local_stack"

echo ""
echo "🎉 Local Stack Setup Complete"
echo ""
