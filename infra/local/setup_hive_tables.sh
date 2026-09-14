#!/usr/bin/env bash
# Idempotently download datasets and register Spark SQL external tables.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MOVIELENS_S3_URI="${MOVIELENS_S3_URI:-s3a://${ZENML_DATA_BUCKET:-zenml-data}/movielens}"
SEAWEEDFS_S3_INTERNAL_ENDPOINT="${SEAWEEDFS_S3_INTERNAL_ENDPOINT:-http://seaweedfs:8333}"
LOCAL_DOCKER_NETWORK="${LOCAL_DOCKER_NETWORK:-zenml-local}"
SPARK_SQL_TIMEOUT_SECONDS="${SPARK_SQL_TIMEOUT_SECONDS:-20}"
# CTAS table creation reads the full CSV and writes partitioned Parquet, so it needs much more time than metadata-only queries.
SPARK_SQL_CTAS_TIMEOUT_SECONDS="${SPARK_SQL_CTAS_TIMEOUT_SECONDS:-1800}"
SPARK_SQL_READY_ATTEMPTS="${SPARK_SQL_READY_ATTEMPTS:-6}"
SPARK_SQL_READY_INTERVAL_SECONDS="${SPARK_SQL_READY_INTERVAL_SECONDS:-2}"

cd "${REPO_ROOT}"

# Run a Spark SQL command inside the Spark master container.
run_spark_sql() {
  local timeout_seconds="${2:-${SPARK_SQL_TIMEOUT_SECONDS}}"
  # fs.s3a.impl.disable.cache avoids reusing a closed cached S3AFileSystem
  # instance when a CSV read is immediately followed by a write to the same bucket.
  # spark.executor.memory defaults to 1g regardless of SPARK_WORKER_MEMORY, which
  # OOM-kills executors (exit 137) writing the larger partitioned MovieLens tables.
  # --total-executor-cores 1 caps this to a single concurrent task so it doesn't
  # have to share that memory with a sibling task in the same executor JVM.
  docker compose exec -T spark-master timeout "${timeout_seconds}" /opt/spark/bin/spark-sql \
    --master spark://spark-master:7077 \
    --conf spark.hadoop.fs.s3a.impl.disable.cache=true \
    --conf spark.executor.memory=3g \
    --total-executor-cores 1 \
    -e "$1"
}

# Wait for the Spark SQL service to be available.
wait_for_spark() {
  local attempt
  for attempt in $(seq 1 "${SPARK_SQL_READY_ATTEMPTS}"); do
    if run_spark_sql "SHOW DATABASES" >/dev/null 2>&1; then
      return
    fi
    sleep "${SPARK_SQL_READY_INTERVAL_SECONDS}"
  done

  echo "-> Spark SQL could not connect to Hive Metastore after ${SPARK_SQL_READY_ATTEMPTS} attempts." >&2
  echo "-> Check Hive with: docker compose ps hive-metastore && docker logs --tail=100 hive-metastore" >&2
  exit 1
}

# Check if a Spark SQL table exists.
table_exists() {
  local table_name="$1"
  run_spark_sql "SHOW TABLES IN default LIKE '${table_name}'" | grep -q "${table_name}"
}

# Download an archive from a URL, with retries and handling for expired certificates.
download_archive() {
  local dataset_url="$1"
  local archive_path="$2"
  local archive_dir
  local archive_name
  archive_dir="$(dirname "${archive_path}")"
  archive_name="$(basename "${archive_path}")"

  if command -v aria2c >/dev/null 2>&1; then
    if aria2c \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --continue=true \
      --dir "${archive_dir}" \
      --max-connection-per-server=8 \
      --max-tries=5 \
      --min-split-size=1M \
      --out "${archive_name}" \
      --retry-wait=5 \
      --split=8 \
      "${dataset_url}"; then
      return
    fi

    echo "-> Verified aria2c download failed; retrying due to the expired GroupLens certificate..." >&2
    aria2c \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --check-certificate=false \
      --continue=true \
      --dir "${archive_dir}" \
      --max-connection-per-server=8 \
      --max-tries=5 \
      --min-split-size=1M \
      --out "${archive_name}" \
      --retry-wait=5 \
      --split=8 \
      "${dataset_url}"
    return
  fi

  if ! curl --fail --location --retry 3 --continue-at - --output "${archive_path}" "${dataset_url}"; then
    echo "-> Verified download failed; retrying due to the expired GroupLens certificate..." >&2
    curl --fail --insecure --location --retry 3 --continue-at - --output "${archive_path}" "${dataset_url}"
  fi
}

# Run a command against the SeaweedFS S3-compatible storage.
run_seaweedfs_s3() {
  docker run --rm \
    --network "${LOCAL_DOCKER_NETWORK}" \
    --env "AWS_ACCESS_KEY_ID=${SEAWEEDFS_ACCESS_KEY_ID:-admin}" \
    --env "AWS_SECRET_ACCESS_KEY=${SEAWEEDFS_SECRET_ACCESS_KEY:-secret}" \
    --env "AWS_DEFAULT_REGION=${AWS_REGION:-us-east-1}" \
    --volume "${DATA_DIR}:/data:ro" \
    amazon/aws-cli:2.24.22 \
    --endpoint-url "${SEAWEEDFS_S3_INTERNAL_ENDPOINT}" "$@"
}

# -------------------------------------------------
# MovieLens dataset setup (1M & 10M ratings)
# -------------------------------------------------

MOVIELENS_DATA_DIR="${MOVIELENS_DATA_DIR:-${DATA_DIR}/movielens}"

# Idempotently download the MovieLens dataset if it doesn't already exist.
download_movielens_dataset() {
  local dataset_url="$1"
  local ratings_file="$2"
  local archive_path="${MOVIELENS_DATA_DIR}/$(basename "${dataset_url}")"

  if [[ -f "${MOVIELENS_DATA_DIR}/${ratings_file}" ]]; then
    return
  fi

  mkdir -p "${MOVIELENS_DATA_DIR}"
  echo "Downloading $(basename "${dataset_url}")..."
  download_archive "${dataset_url}" "${archive_path}"
  unzip -q "${archive_path}" -d "${MOVIELENS_DATA_DIR}"
  rm -f "${archive_path}"
}

# Upload the MovieLens dataset to the S3-compatible storage if it hasn't been uploaded already.
upload_movielens_dataset() {
  local ratings_file="$1"
  local s3_key="movielens/${ratings_file}"
  local s3_uri="s3://${ZENML_DATA_BUCKET:-zenml-data}/${s3_key}"

  if run_seaweedfs_s3 s3api head-object \
    --bucket "${ZENML_DATA_BUCKET:-zenml-data}" \
    --key "${s3_key}" >/dev/null 2>&1; then
    return
  fi

  run_seaweedfs_s3 s3 cp "/data/movielens/${ratings_file}" "${s3_uri}"
  echo "-> Uploaded ${ratings_file} to ${s3_uri}."
}

# Create a Spark SQL external table for the MovieLens dataset if it doesn't already exist.
create_movielens_table() {
  local table_name="$1"
  local dataset_url="$2"
  local ratings_file="$3"
  local csv_options="$4"

  if table_exists "${table_name}"; then
    echo "-> Hive table ${table_name} already exists; skipping."
    return
  fi

  download_movielens_dataset "${dataset_url}" "${ratings_file}"
  upload_movielens_dataset "${ratings_file}"
  run_spark_sql "
    CREATE TEMPORARY VIEW ${table_name}_staging (
      userId INT,
      movieId INT,
      rating DOUBLE,
      timestamp BIGINT
    ) USING csv
    OPTIONS (path '${MOVIELENS_S3_URI}/${ratings_file}', ${csv_options});
    CREATE TABLE IF NOT EXISTS ${table_name}
    USING parquet
    PARTITIONED BY (eventDate)
    LOCATION '${MOVIELENS_S3_URI}/partitioned/${table_name}'
    AS SELECT
      userId,
      movieId,
      rating,
      timestamp,
      CAST(from_unixtime(timestamp) AS DATE) AS eventDate
    FROM ${table_name}_staging
    CLUSTER BY eventDate" "${SPARK_SQL_CTAS_TIMEOUT_SECONDS}"
  echo "-> Created Hive table ${table_name} (partitioned by eventDate)."
}

wait_for_spark

# Create MovieLens Hive tables.
create_movielens_table \
  "ml_ratings_1m" \
  "https://files.grouplens.org/datasets/movielens/ml-1m.zip" \
  "ml-1m/ratings.dat" \
  "sep '::', header 'false'"
create_movielens_table \
  "ml_ratings_10m" \
  "https://files.grouplens.org/datasets/movielens/ml-10m.zip" \
  "ml-10M100K/ratings.dat" \
  "sep '::', header 'false'"
