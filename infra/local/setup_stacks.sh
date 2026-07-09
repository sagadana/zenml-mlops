#!/usr/bin/env bash
# infra/local/setup_stacks.sh
#
# Idempotent ZenML stack setup script for local development.
#
# Usage:
#   bash infra/local/setup_stacks.sh

set -euo pipefail

ARTIFACT_STORE_PATH="./data/artifact_store"


# --------------------------------------
# Register ZenML stack components
# --------------------------------------

# Artifact store
echo "==> Registering custom artifact store..."
zenml artifact-store describe project_store 2>/dev/null || \
  zenml artifact-store register project_store \
    --flavor=local \
    --path="$ARTIFACT_STORE_PATH" --set
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


# --------------------------------------
# Register ZenML AWS stack
# --------------------------------------

echo "==> Registering local ZenML stack..."

zenml stack describe local_stack 2>/dev/null || \
  zenml stack register local_stack \
    --orchestrator=default \
    --artifact-store=project_store \
    --experiment-tracker=mlflow_tracker \
    --set 
echo "  ✓ Stack: local_stack"

echo ""
echo "=== Local setup complete ==="
echo "Available stacks:"
zenml stack list
echo ""
echo "To switch stack:"
echo "  zenml stack set local_stack"
