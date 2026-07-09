#!/usr/bin/env sh
set -eu

: "${EVIDENTLY_HOST:=0.0.0.0}"
: "${EVIDENTLY_PORT:=8000}"
: "${EVIDENTLY_WORKSPACE:=/app/workspace}"

mkdir -p "${EVIDENTLY_WORKSPACE}"

exec evidently ui \
  --host "${EVIDENTLY_HOST}" \
  --port "${EVIDENTLY_PORT}" \
  --workspace "${EVIDENTLY_WORKSPACE}"
