#!/usr/bin/env sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:=postgresql://mlflow:mlflow@mlflow-db:5432/mlflow}"
: "${MLFLOW_DEFAULT_ARTIFACT_ROOT:=/mlflow/artifacts}"
: "${MLFLOW_HOST:=0.0.0.0}"
: "${MLFLOW_PORT:=5000}"

mkdir -p "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --serve-artifacts
