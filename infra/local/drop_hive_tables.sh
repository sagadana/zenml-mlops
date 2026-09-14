#!/usr/bin/env bash
# Idempotently drop local Hive tables.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SPARK_SQL_TIMEOUT_SECONDS="${SPARK_SQL_TIMEOUT_SECONDS:-20}"

cd "${REPO_ROOT}"

run_spark_sql() {
  docker compose exec -T spark-master timeout "${SPARK_SQL_TIMEOUT_SECONDS}" /opt/spark/bin/spark-sql \
    --master spark://spark-master:7077 \
    -e "$1"
}

drop_table() {
  local table_name="$1"
  local table_ddl

  if ! run_spark_sql "SHOW TABLES IN default LIKE '${table_name}'" | grep -q "${table_name}"; then
    echo "-> Hive table ${table_name} does not exist; skipping."
    return
  fi

  run_spark_sql "DROP TABLE IF EXISTS default.${table_name}"
  echo "-> Dropped Hive table ${table_name}."
}

for table_name in ml_ratings_1m ml_ratings_10m ml_ratings_25m; do
  drop_table "${table_name}"
done