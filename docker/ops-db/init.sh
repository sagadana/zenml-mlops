#!/bin/bash
# docker/ops-db/init.sh
#
# Runs once on first container boot (postgres docker-entrypoint-initdb.d).
# Creates dedicated databases for MLflow and ZenML under the shared ops user.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE ${MLFLOW_DB_NAME:-mlflow};
    CREATE DATABASE ${ZENML_DB_NAME:-zenml};
EOSQL
