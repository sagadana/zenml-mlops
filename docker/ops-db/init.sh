#!/bin/bash
# docker/ops-db/init.sh
#
# Runs once on first container boot (mysql docker-entrypoint-initdb.d).
# Creates dedicated databases for ZenML and Optuna under the shared ops user.
set -e

mysql -u root -p"${MYSQL_ROOT_PASSWORD}" <<-EOSQL
    CREATE DATABASE IF NOT EXISTS \`${ZENML_DB_NAME:-zenml}\`;
    CREATE DATABASE IF NOT EXISTS \`${OPTUNA_DB_NAME:-optuna}\`;
    GRANT ALL PRIVILEGES ON \`${ZENML_DB_NAME:-zenml}\`.* TO '${MYSQL_USER:-ops}'@'%';
    GRANT ALL PRIVILEGES ON \`${OPTUNA_DB_NAME:-optuna}\`.* TO '${MYSQL_USER:-ops}'@'%';
    FLUSH PRIVILEGES;
EOSQL
