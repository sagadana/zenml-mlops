"""
helpers/dask_cluster.py

Factory for creating Dask clients for local and remote (e.g. AWS ECS/SageMaker) environments.

Usage:
    from helpers.dask_cluster import get_dask_client

    with get_dask_client(mode="local", n_workers=4) as client:
        futures = [client.submit(fn, arg) for arg in args]
        results = client.gather(futures)

Environment variables (used when mode="remote"):
    DASK_SCHEDULER_ADDRESS — e.g. "tcp://dask-scheduler:8786"
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

logger = logging.getLogger(__name__)


@contextmanager
def get_dask_client(
    mode: Literal["local", "remote"] = "local",
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    scheduler_address: str | None = None,
) -> Generator:
    """
    Context manager that yields a Dask client and cleans up on exit.

    Args:
        mode: "local" creates a LocalCluster in-process.
              "remote" connects to an existing Dask scheduler.
        n_workers: Number of workers for LocalCluster. Defaults to CPU count.
        threads_per_worker: Threads per Dask worker.
            Set to 1 when using Numba (Numba handles its own threading via prange).
        scheduler_address: Scheduler address for remote mode.
            Defaults to DASK_SCHEDULER_ADDRESS env var.

    Yields:
        dask.distributed.Client
    """
    from dask.distributed import Client, LocalCluster

    if mode == "local":
        import multiprocessing

        workers = n_workers or multiprocessing.cpu_count()
        logger.info(
            "Starting Dask LocalCluster: %d workers, %d thread(s) each",
            workers,
            threads_per_worker,
        )
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=threads_per_worker,
        )
        client = Client(cluster)
        try:
            logger.info("Dask LocalCluster started: %s", client.dashboard_link)
            yield client
        finally:
            client.close()
            cluster.close()
            logger.info("Dask LocalCluster shut down")

    elif mode == "remote":
        addr = scheduler_address or os.environ.get("DASK_SCHEDULER_ADDRESS")
        if not addr:
            raise ValueError(
                "mode='remote' requires a scheduler address. "
                "Pass scheduler_address= or set DASK_SCHEDULER_ADDRESS."
            )
        logger.info("Connecting to remote Dask scheduler at %s", addr)
        client = Client(addr)
        try:
            logger.info(
                "Connected to Dask cluster: %s workers",
                len(client.scheduler_info()["workers"]),
            )
            yield client
        finally:
            client.close()
            logger.info("Dask client disconnected")

    else:
        raise ValueError(f"Unknown Dask mode: {mode!r}. Expected 'local' or 'remote'.")


def get_client_mode_from_config(config: dict) -> Literal["local", "remote"]:
    """
    Determine Dask client mode from pipeline config parameters.
    Returns "remote" if DASK_SCHEDULER_ADDRESS is set, else "local".
    """
    if config.get("dask_scheduler_address") or os.environ.get("DASK_SCHEDULER_ADDRESS"):
        return "remote"
    return "local"
