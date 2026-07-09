#!/usr/bin/env sh
set -eu

: "${DASK_SCHEDULER_PORT:=8786}"
: "${DASK_DASHBOARD_PORT:=8787}"

exec dask scheduler \
  --host 0.0.0.0 \
  --port "${DASK_SCHEDULER_PORT}" \
  --dashboard-address "0.0.0.0:${DASK_DASHBOARD_PORT}"
