#!/usr/bin/env sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:=mysql+pymysql//ops:ops@ops-db:3306/mlflow}"
: "${MLFLOW_DEFAULT_ARTIFACT_ROOT:=s3://zenml-artifacts/mlflow}"
: "${MLFLOW_HOST:=0.0.0.0}"
: "${MLFLOW_PORT:=5000}"

case "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" in
  s3://*) ;;
  *) mkdir -p "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" ;;
esac

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --serve-artifacts
