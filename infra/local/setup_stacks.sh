#!/usr/bin/env bash
# infra/local/setup_stacks.sh
#
# Idempotent ZenML stack setup script for local development.
#
# Usage:
#   bash infra/local/setup_stacks.sh

set -euo pipefail

# This script configures local execution to run pipeline steps in Docker
# (local_docker orchestrator) while still talking to the local ZenML server
# and SeaweedFS services running on the host machine.

INFRA_DIR="$(cd "$(dirname "$0")" && pwd)/.."

ARTIFACT_STORE_PATH_RAW="${ZENML_ARTIFACT_STORE_PATH:-s3://${ZENML_ARTIFACT_BUCKET:-zenml-artifacts}/}"
ARTIFACT_STORE_ENDPOINT_URL_RAW="${ZENML_ARTIFACT_STORE_ENDPOINT_URL:-${ZENML_ARTIFACT_STORE_ENDPOINT_URL_DOCKER:-http://host.docker.internal:${SEAWEEDFS_S3_PORT:-8333}}}"
ZENML_SERVER_INTERNAL_URI_RAW="${ZENML_SERVER_INTERNAL_URI:-http://host.docker.internal:${ZENML_SERVER_PORT:-8237}}"

# Optional override used when host.docker.internal is not resolvable on host OS.
ZENML_HOST_IP_RAW="${ZENML_HOST_IP:-}"
SEAWEEDFS_S3_PORT_RAW="${SEAWEEDFS_S3_PORT:-8333}"
S3_ACCESS_KEY_ID_RAW="${SEAWEEDFS_ACCESS_KEY_ID:-admin}"
S3_SECRET_ACCESS_KEY_RAW="${SEAWEEDFS_SECRET_ACCESS_KEY:-secret}"

DEFAULT_LOCAL_STACK_NAME="local_docker_stack"
DEFAULT_LOCAL_ORCHESTRATOR_NAME="local_docker_orchestrator"
DEFAULT_LOCAL_ARTIFACT_STORE_NAME="local_s3_store_docker"
DEFAULT_LOCAL_S3_SECRET_NAME="local_s3_auth_secret"
DEFAULT_LOCAL_DATA_VALIDATOR_NAME="evidently_data_validator"
DEFAULT_LOCAL_CONTAINER_REGISTRY_NAME="local_container_registry"
DEFAULT_LOCAL_IMAGE_BUILDER_NAME="local_image_builder"

ZENML_LOCAL_STACK_NAME="${ZENML_LOCAL_STACK_NAME:-${DEFAULT_LOCAL_STACK_NAME}}"
ZENML_LOCAL_ORCHESTRATOR_NAME="${ZENML_LOCAL_ORCHESTRATOR_NAME:-${DEFAULT_LOCAL_ORCHESTRATOR_NAME}}"
ZENML_LOCAL_ARTIFACT_STORE_NAME="${ZENML_LOCAL_ARTIFACT_STORE_NAME:-${DEFAULT_LOCAL_ARTIFACT_STORE_NAME}}"
ZENML_LOCAL_S3_SECRET_NAME="${ZENML_LOCAL_S3_SECRET_NAME:-${DEFAULT_LOCAL_S3_SECRET_NAME}}"
ZENML_LOCAL_DATA_VALIDATOR_NAME="${ZENML_LOCAL_DATA_VALIDATOR_NAME:-${DEFAULT_LOCAL_DATA_VALIDATOR_NAME}}"
ZENML_LOCAL_CONTAINER_REGISTRY_NAME="${ZENML_LOCAL_CONTAINER_REGISTRY_NAME:-${DEFAULT_LOCAL_CONTAINER_REGISTRY_NAME}}"
ZENML_LOCAL_IMAGE_BUILDER_NAME="${ZENML_LOCAL_IMAGE_BUILDER_NAME:-${DEFAULT_LOCAL_IMAGE_BUILDER_NAME}}"
LOCAL_REGISTRY_PORT_RAW="${LOCAL_REGISTRY_PORT:-5001}"
LOCAL_DOCKER_RUN_ARGS_RAW="${LOCAL_DOCKER_RUN_ARGS:-}"

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
ARTIFACT_STORE_ENDPOINT_URL="$(strip_wrapping_quotes "${ARTIFACT_STORE_ENDPOINT_URL_RAW}")"
ZENML_SERVER_INTERNAL_URI="$(strip_wrapping_quotes "${ZENML_SERVER_INTERNAL_URI_RAW}")"
ZENML_HOST_IP="$(strip_wrapping_quotes "${ZENML_HOST_IP_RAW}")"
SEAWEEDFS_S3_PORT="$(strip_wrapping_quotes "${SEAWEEDFS_S3_PORT_RAW}")"
LOCAL_REGISTRY_PORT="$(strip_wrapping_quotes "${LOCAL_REGISTRY_PORT_RAW}")"
LOCAL_DOCKER_RUN_ARGS="$(strip_wrapping_quotes "${LOCAL_DOCKER_RUN_ARGS_RAW}")"

if [[ -z "${LOCAL_DOCKER_RUN_ARGS}" ]]; then
  LOCAL_DOCKER_RUN_ARGS='{"volumes": {"/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}}}'
fi

resolve_host_ip() {
  # Resolve the host LAN IP from the active default route without hardcoding
  # interface names (works better across Wi-Fi/Ethernet changes).
  python3 -c 'import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()'
}

host_resolves() {
  # Some host environments (especially Linux) cannot resolve
  # host.docker.internal. Detect this so we can fall back to a LAN IP URL.
  python3 - "$1" <<'PY'
import socket
import sys
host = sys.argv[1]
try:
    socket.getaddrinfo(host, None)
except OSError:
    sys.exit(1)
sys.exit(0)
PY
}

if [[ -z "${ZENML_HOST_IP}" ]]; then
  ZENML_HOST_IP="$(resolve_host_ip || true)"
fi

if [[ -n "${ZENML_HOST_IP}" ]]; then
  AUTO_ENDPOINT_URL="http://${ZENML_HOST_IP}:${SEAWEEDFS_S3_PORT}"
else
  AUTO_ENDPOINT_URL=""
fi

# If the configured endpoint references host.docker.internal but the host itself
# cannot resolve it, use a host LAN IP endpoint that both host and containers
# can reach.
if [[ "${ARTIFACT_STORE_ENDPOINT_URL}" == *"host.docker.internal"* ]]; then
  if ! host_resolves "host.docker.internal" && [[ -n "${AUTO_ENDPOINT_URL}" ]]; then
    echo "  ⚠ host.docker.internal is not resolvable on this host; using ${AUTO_ENDPOINT_URL} for artifact store endpoint"
    ARTIFACT_STORE_ENDPOINT_URL="${AUTO_ENDPOINT_URL}"
  fi
fi

if [[ -z "${ARTIFACT_STORE_ENDPOINT_URL}" ]]; then
  if [[ -n "${AUTO_ENDPOINT_URL}" ]]; then
    ARTIFACT_STORE_ENDPOINT_URL="${AUTO_ENDPOINT_URL}"
  else
    ARTIFACT_STORE_ENDPOINT_URL="http://host.docker.internal:${SEAWEEDFS_S3_PORT}"
  fi
fi

ARTIFACT_STORE_CLIENT_KWARGS="$(python3 -c 'import json,sys; print(json.dumps({"endpoint_url": sys.argv[1]}))' "${ARTIFACT_STORE_ENDPOINT_URL}")"
S3_ACCESS_KEY_ID="$(strip_wrapping_quotes "${S3_ACCESS_KEY_ID_RAW}")"
S3_SECRET_ACCESS_KEY="$(strip_wrapping_quotes "${S3_SECRET_ACCESS_KEY_RAW}")"
S3_AUTH_SECRET_VALUES="$(python3 -c 'import json,sys; print(json.dumps({"access_key_id": sys.argv[1], "secret_access_key": sys.argv[2]}))' "${S3_ACCESS_KEY_ID}" "${S3_SECRET_ACCESS_KEY}")"

# --------------------------------------
# Register ZenML stack components
# --------------------------------------

# Local Docker orchestrator
# We explicitly set the in-container ZenML server URL so step containers do not
# try to reach localhost:8237 inside themselves.
echo ""
echo "==> Registering local Docker orchestrator..."

if zenml orchestrator describe "${ZENML_LOCAL_ORCHESTRATOR_NAME}" >/dev/null 2>&1; then
  zenml orchestrator update "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    --env "ZENML_STORE_URL=${ZENML_SERVER_INTERNAL_URI}" \
    --env "ZENML_STORE_VERIFY_SSL=False" \
    --run_args="${LOCAL_DOCKER_RUN_ARGS}"
else
  zenml orchestrator register "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    --flavor=local_docker \
    --env "ZENML_STORE_URL=${ZENML_SERVER_INTERNAL_URI}" \
    --env "ZENML_STORE_VERIFY_SSL=False" \
    --run_args="${LOCAL_DOCKER_RUN_ARGS}"
fi
echo "  ✓ Orchestrator: ${ZENML_LOCAL_ORCHESTRATOR_NAME}"

# Artifact store
# We attach an auth secret to keep credentials out of component config and match
# ZenML's S3-compatible secret-based pattern (MinIO/SeaweedFS style).
echo ""
echo "==> Registering local S3 artifact store (SeaweedFS)..."

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
    --client_kwargs="${ARTIFACT_STORE_CLIENT_KWARGS}" \
    --authentication_secret="${ZENML_LOCAL_S3_SECRET_NAME}"
else
  zenml artifact-store register "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    --flavor=s3 \
    --path="${ARTIFACT_STORE_PATH}" \
    --client_kwargs="${ARTIFACT_STORE_CLIENT_KWARGS}" \
    --authentication_secret="${ZENML_LOCAL_S3_SECRET_NAME}"
fi
echo "  ✓ Artifact store: ${ZENML_LOCAL_ARTIFACT_STORE_NAME} (endpoint=${ARTIFACT_STORE_ENDPOINT_URL})"


# Evidently data validator
echo ""
echo "==> Registering local Evidently data validator..."

zenml data-validator describe "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" 2>/dev/null || \
  zenml data-validator register "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" --flavor=evidently
echo "  ✓ Data validator: ${ZENML_LOCAL_DATA_VALIDATOR_NAME}"


# Local container registry (Docker registry:2 running via docker-compose)
# Uses the default flavor which accepts any URI, including localhost registries.
echo ""
echo "==> Registering local container registry..."

LOCAL_REGISTRY_URI="localhost:${LOCAL_REGISTRY_PORT}"

if zenml container-registry describe "${ZENML_LOCAL_CONTAINER_REGISTRY_NAME}" >/dev/null 2>&1; then
  zenml container-registry update "${ZENML_LOCAL_CONTAINER_REGISTRY_NAME}" \
    --uri="${LOCAL_REGISTRY_URI}"
else
  zenml container-registry register "${ZENML_LOCAL_CONTAINER_REGISTRY_NAME}" \
    --flavor=default \
    --uri="${LOCAL_REGISTRY_URI}"
fi
echo "  ✓ Container registry: ${ZENML_LOCAL_CONTAINER_REGISTRY_NAME} (uri=${LOCAL_REGISTRY_URI})"


# Local image builder configured to run Docker builds as subprocess calls.
# This enables Docker CLI/BuildKit behavior required by some build workflows.
echo ""
echo "==> Registering local image builder..."

if zenml image-builder describe "${ZENML_LOCAL_IMAGE_BUILDER_NAME}" >/dev/null 2>&1; then
  zenml image-builder update "${ZENML_LOCAL_IMAGE_BUILDER_NAME}" \
    --use_subprocess_call=true
else
  zenml image-builder register "${ZENML_LOCAL_IMAGE_BUILDER_NAME}" \
    --flavor=local \
    --use_subprocess_call=true
fi
echo "  ✓ Image builder: ${ZENML_LOCAL_IMAGE_BUILDER_NAME} (use_subprocess_call=true)"


# --------------------------------------
# Register ZenML local stack
# --------------------------------------
echo ""
echo "==> Registering local ZenML stack..."

if zenml stack describe "${ZENML_LOCAL_STACK_NAME}" >/dev/null 2>&1; then
  zenml stack update "${ZENML_LOCAL_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    -c "${ZENML_LOCAL_CONTAINER_REGISTRY_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" \
    -i "${ZENML_LOCAL_IMAGE_BUILDER_NAME}"
  zenml stack set "${ZENML_LOCAL_STACK_NAME}"
else
  zenml stack register "${ZENML_LOCAL_STACK_NAME}" \
    -o "${ZENML_LOCAL_ORCHESTRATOR_NAME}" \
    -a "${ZENML_LOCAL_ARTIFACT_STORE_NAME}" \
    -c "${ZENML_LOCAL_CONTAINER_REGISTRY_NAME}" \
    -dv "${ZENML_LOCAL_DATA_VALIDATOR_NAME}" \
    -i "${ZENML_LOCAL_IMAGE_BUILDER_NAME}" \
    --set
fi
echo "  ✓ Stack: ${ZENML_LOCAL_STACK_NAME}"

# ---------------------------
# Additional external setups 
# ---------------------------

# Set up service account if ZENML_SERVICE_ACCOUNT_NAME is provided
if [[ -n "${ZENML_SERVICE_ACCOUNT_NAME:-}" ]]; then
  source "${INFRA_DIR}/setup_service_account.sh"
fi

# Register GitHub code repository if GITHUB_TOKEN is provided
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  source "${INFRA_DIR}/setup_code_repo.sh"
fi

echo ""
echo "🎉 Local Stack Setup Complete"
echo ""
