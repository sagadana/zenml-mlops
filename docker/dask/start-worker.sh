#!/usr/bin/env sh
set -eu

: "${DASK_SCHEDULER_ADDRESS:=tcp://dask-scheduler:8786}"

exec dask worker "${DASK_SCHEDULER_ADDRESS}"
