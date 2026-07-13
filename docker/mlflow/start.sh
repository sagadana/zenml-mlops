#!/usr/bin/env sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:=mysql+pymysql//ops:ops@ops-db:3306/mlflow}"
: "${MLFLOW_REGISTRY_STORE_URI:=mysql+pymysql//ops:ops@ops-db:3306/mlflow}"
: "${MLFLOW_ARTIFACT_URI:=s3://zenml-artifacts/mlflow}"
: "${MLFLOW_HOST:=0.0.0.0}"
: "${MLFLOW_PORT:=5000}"

case "${MLFLOW_ARTIFACT_URI}" in
  s3://*) ;;
  *) mkdir -p "${MLFLOW_ARTIFACT_URI}" ;;
esac

exec mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --registry-store-uri "${MLFLOW_REGISTRY_STORE_URI}" \
  --artifacts-destination "${MLFLOW_ARTIFACT_URI}" \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --serve-artifacts
