#!/usr/bin/env bash
# infra/setup_service_account.sh
#
# Shared helper to create a ZenML service account.
# Called from Makefile target: make zenml-service-account
#
# Required environment variables:
#   ZENML_SERVICE_ACCOUNT_NAME — name for the service account (e.g. "zenml-automation")
#
# Output:
#   Creates a service account in ZenML; API keys are fetched on-demand via Client.get_api_key()

set -euo pipefail

: "${ZENML_SERVICE_ACCOUNT_NAME:?ERROR: ZENML_SERVICE_ACCOUNT_NAME is not set}"

echo ""
echo "==> Setting up service account for API access..."

# Check if service account already exists
if zenml service-account describe "${ZENML_SERVICE_ACCOUNT_NAME}" >/dev/null 2>&1; then
  echo "  ✓ Service account '${ZENML_SERVICE_ACCOUNT_NAME}' already exists"
else
  echo "  → Creating new service account '${ZENML_SERVICE_ACCOUNT_NAME}'..."
  zenml service-account create "${ZENML_SERVICE_ACCOUNT_NAME}"
  echo "  ✓ Service account '${ZENML_SERVICE_ACCOUNT_NAME}' created"
fi

echo ""
echo "  ℹ To use this service account in pipeline triggering:"
echo "    1. Set ZENML_SERVICE_ACCOUNT_NAME='${ZENML_SERVICE_ACCOUNT_NAME}' in .env"
echo "    2. Add 'service_account_name: ${ZENML_SERVICE_ACCOUNT_NAME}' to pipeline config"
echo "    3. Pipeline trigger steps will fetch API keys on-demand"
echo ""

