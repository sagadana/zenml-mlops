#!/usr/bin/env bash
# infra/local/setup_stacks.sh
#
# Idempotent ZenML stack setup script for local development.
#
# Usage:
#   bash infra/local/setup_stacks.sh

set -euo pipefail

ARTIFACT_STORE_PATH="${ZENML_ARTIFACT_STORE_PATH:-./data/artifact_store}"

# --------------------------------------
# Register ZenML stack components
# --------------------------------------

# Artifact store
echo ""
echo "==> Registering custom artifact store..."
zenml artifact-store describe project_store 2>/dev/null || \
  zenml artifact-store register project_store \
    --flavor=local \
    --path="$ARTIFACT_STORE_PATH"
echo "  ✓ Artifact store: project_store"


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
echo ""
echo "==> Registering local ZenML stack..."

zenml stack describe local_stack 2>/dev/null || \
  zenml stack register local_stack \
    -o default \
    -a project_store \
    -e mlflow_tracker \
    -dv evidently_data_validator \
    --set
echo "  ✓ Stack: local_stack"

echo ""
echo "🎉 Local Stack Setup Complete"
echo ""
