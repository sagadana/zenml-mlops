"""
helpers/resource_monitor.py

Lightweight per-epoch resource snapshot utilities.

Captures CPU utilisation, process RSS memory, and (optionally) GPU VRAM for
a running training process. Designed to be called once per epoch from within
a training callback — no background threads, minimal overhead.

Usage
-----
    from helpers.resource_monitor import ResourceSnapshot, capture_snapshot

    snap = capture_snapshot(use_gpu=True)   # → ResourceSnapshot
    print(snap.cpu_percent, snap.memory_mb, snap.gpu_memory_mb)
"""

from __future__ import annotations

import logging
import os

import psutil
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Process handle reused across calls to avoid repeated /proc lookups.
_PROC = psutil.Process(os.getpid())


class ResourceSnapshot(BaseModel):
    """Resource usage at a single point in time."""

    cpu_percent: float
    """CPU utilisation of the current process (%), averaged since last call."""

    memory_mb: float
    """Resident Set Size (RSS) of the current process in MiB."""

    gpu_memory_mb: float | None = None
    """GPU VRAM used by the current process in MiB, or None if not available."""


def capture_snapshot(use_gpu: bool = False) -> ResourceSnapshot:
    """
    Capture a lightweight resource snapshot for the current process.

    Args:
        use_gpu: When True, attempt to query GPU memory via pynvml (CUDA) or
                 pytorch's cuda module. Falls back to None if unavailable.

    Returns:
        ResourceSnapshot with cpu_percent, memory_mb, and gpu_memory_mb.
    """
    cpu = _PROC.cpu_percent()
    mem_mb = _PROC.memory_info().rss / (1024**2)
    gpu_mb: float | None = None

    if use_gpu:
        gpu_mb = _query_gpu_memory_mb()

    return ResourceSnapshot(cpu_percent=cpu, memory_mb=mem_mb, gpu_memory_mb=gpu_mb)


def _query_gpu_memory_mb() -> float | None:
    """
    Query GPU memory used by the current process.

    Tries pynvml first (works for any CUDA GPU without PyTorch), then falls
    back to torch.cuda if available. Returns None on any failure so callers
    are never broken by missing optional dependencies.
    """
    # --- pynvml path (preferred) -------------------------------------------
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        n_devs = pynvml.nvmlDeviceGetCount()
        total_mb = 0.0
        pid = os.getpid()
        for i in range(n_devs):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for p in procs:
                    if p.pid == pid:
                        total_mb += p.usedGpuMemory / (1024**2)
            except pynvml.NVMLError:
                pass
        pynvml.nvmlShutdown()
        return total_mb if total_mb > 0 else None
    except Exception:
        pass

    # --- torch.cuda fallback -----------------------------------------------
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
    except Exception:
        pass

    logger.debug("GPU memory query unavailable; returning None.")
    return None
