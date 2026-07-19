"""
helpers/checkpointing.py

Epoch-level checkpoint save/load for iterative ML training.

Atomic write protocol:
  1. Write epoch_{N:04d}_primary.npy
  2. Write epoch_{N:04d}_secondary.npy  (optional; skip if only one weight array)
  3. Write epoch_{N:04d}.done           ← written LAST; guarantees atomicity

On restart, load_latest_checkpoint() scans for .done files only — any epoch
without a .done marker (partial write from a crash) is silently ignored.

Storage backends:
  - Local filesystem: base_path = "./checkpoints/run_id/"
  - S3: base_path = "s3://my-bucket/checkpoints/run_id/"
    (s3fs makes np.save / np.load calls transparent)

Retention:
  - clean_run_checkpoints() deletes all checkpoint files for a completed run.
    Call this after model registration succeeds to avoid unbounded storage growth.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
from zenml import get_step_context

logger = logging.getLogger(__name__)

# s3fs is imported lazily so local-only usage doesn't require AWS credentials
_s3fs = None
_s3fs_endpoint: str | None = None


def _get_fs(
    path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
):
    """Return (filesystem, path_str) for local or S3 paths."""

    if path.startswith("s3://"):
        global _s3fs, _s3fs_endpoint

        if _s3fs is None or _s3fs_endpoint != seaweedfs_s3_internal_endpoint:
            import s3fs as _s3fs_lib

            if seaweedfs_s3_internal_endpoint:
                if not seaweedfs_access_key_id or not seaweedfs_secret_access_key:
                    raise ValueError(
                        "S3 path requested but SeaweedFS credentials are missing. "
                        "Please provide seaweedfs_access_key_id and seaweedfs_secret_access_key."
                    )

                _s3fs = _s3fs_lib.S3FileSystem(
                    anon=False,
                    endpoint_url=seaweedfs_s3_internal_endpoint,
                    key=seaweedfs_access_key_id,
                    secret=seaweedfs_secret_access_key,
                )
            else:
                _s3fs = _s3fs_lib.S3FileSystem(anon=False)

            _s3fs_endpoint = _s3fs_endpoint

        return _s3fs, path

    return None, path  # None signals "use local pathlib"


def _ls(
    path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> list[str]:
    """List files in a directory (local or S3). Returns full paths."""
    fs, p = _get_fs(
        path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if fs is not None:
        try:
            return [f"s3://{f}" for f in fs.ls(p, detail=False)]
        except FileNotFoundError:
            return []
    local = Path(p)
    if not local.exists():
        return []
    return [str(f) for f in local.iterdir()]


def _makedirs(
    path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> None:
    """Create directory (local only; S3 has no real directories)."""
    fs, p = _get_fs(
        path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if fs is None:
        Path(p).mkdir(parents=True, exist_ok=True)


def _save_npy(
    path: str,
    array: np.ndarray,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> None:
    """Save numpy array to local or S3 path."""
    fs, p = _get_fs(
        path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if fs is not None:
        buf = io.BytesIO()
        np.save(buf, array)
        buf.seek(0)
        with fs.open(p, "wb") as f:
            f.write(buf.read())  # type: ignore
    else:
        np.save(p, array)


def _load_npy(
    path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> np.ndarray:
    """Load numpy array from local or S3 path."""
    fs, p = _get_fs(
        path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if fs is not None:
        with fs.open(p, "rb") as f:
            return np.load(io.BytesIO(f.read()))  # type: ignore
    return np.load(p)


def _write_marker(
    path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> None:
    """Write an empty .done marker file."""
    fs, p = _get_fs(
        path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if fs is not None:
        with fs.open(p, "w") as f:
            f.write("")
    else:
        Path(p).write_text("")


def _exists(
    path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> bool:
    fs, p = _get_fs(
        path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if fs is not None:
        return fs.exists(p)
    return Path(p).exists()


def _delete(
    path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> None:
    fs, p = _get_fs(
        path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if fs is not None:
        fs.rm(p, recursive=True)
    else:
        import shutil

        local = Path(p)
        if local.is_dir():
            shutil.rmtree(local)
        elif local.exists():
            local.unlink()


# ── Public API ───────────────────────────────────────────────────────────────


def save_checkpoint(
    epoch: int,
    primary: np.ndarray,
    secondary: np.ndarray | None,
    base_path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> None:
    """
    Atomically save model weights for a completed epoch.

    Writes in order:
      1. epoch_{N:04d}_primary.npy
      2. epoch_{N:04d}_secondary.npy  (skipped if secondary is None)
      3. epoch_{N:04d}.done           ← marker only written after arrays are saved

    Args:
        epoch: 1-based epoch number (epoch 1 = first completed epoch).
        primary: Primary weight matrix (e.g. user factors, encoder weights).
        secondary: Optional secondary weight matrix (e.g. item factors). Pass None to skip.
        base_path: Base directory for checkpoints (local path or s3:// URI).
    """
    _makedirs(
        base_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    prefix = f"{base_path}/epoch_{epoch:04d}"

    logger.debug("Saving checkpoint epoch %d to %s", epoch, base_path)
    _save_npy(
        f"{prefix}_primary.npy",
        primary,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    if secondary is not None:
        _save_npy(
            f"{prefix}_secondary.npy",
            secondary,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
    _write_marker(
        f"{prefix}.done",
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )  # atomic commit — written last
    logger.info("Checkpoint saved: epoch %d (%s)", epoch, base_path)


def load_latest_checkpoint(
    base_path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    """
    Load the latest complete checkpoint from base_path.

    Scans for .done marker files to identify completed epochs.
    Ignores any .npy files without a corresponding .done (partial writes from crashes).

    Returns:
        (start_epoch, primary, secondary)
        - start_epoch: next epoch to train (0 if starting fresh)
        - primary: loaded primary weight array, or None if no checkpoint found
        - secondary: loaded secondary weight array, or None if absent/not checkpointed
    """
    all_files = _ls(
        base_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    done_files = sorted(f for f in all_files if re.search(r"epoch_(\d{4})\.done$", f))

    if not done_files:
        logger.info("No checkpoints found at %s — starting from epoch 0", base_path)
        return 0, None, None

    latest_done = done_files[-1]
    match = re.search(r"epoch_(\d{4})\.done$", latest_done)
    assert match, f"Unexpected .done filename: {latest_done}"
    latest_epoch = int(match.group(1))

    prefix = f"{base_path}/epoch_{latest_epoch:04d}"
    primary_path = f"{prefix}_primary.npy"

    if not _exists(
        primary_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    ):
        logger.warning(
            "Checkpoint epoch %d has .done marker but missing primary .npy — "
            "falling back to epoch %d",
            latest_epoch,
            latest_epoch - 1,
        )
        _delete(
            latest_done,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
        return load_latest_checkpoint(
            base_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )

    logger.info("Resuming from checkpoint epoch %d (%s)", latest_epoch, base_path)
    primary = _load_npy(
        primary_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    secondary_path = f"{prefix}_secondary.npy"
    secondary = (
        _load_npy(
            secondary_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
        if _exists(
            secondary_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
        else None
    )
    return latest_epoch, primary, secondary


def load_checkpoint(
    epoch: int,
    base_path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load a specific checkpoint epoch by its 1-based epoch number.

    Returns:
        (primary, secondary) arrays, or (None, None) if the epoch has no committed checkpoint.
    """
    prefix = f"{base_path}/epoch_{epoch:04d}"
    if not _exists(
        f"{prefix}.done",
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    ):
        return None, None
    primary_path = f"{prefix}_primary.npy"
    if not _exists(
        primary_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    ):
        return None, None
    primary = _load_npy(
        primary_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    secondary_path = f"{prefix}_secondary.npy"
    secondary = (
        _load_npy(
            secondary_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
        if _exists(
            secondary_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
        else None
    )
    return primary, secondary


def clean_run_checkpoints(
    base_path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> None:
    """
    Delete all checkpoint files for a completed training run.
    Call this after model registration succeeds to prevent unbounded storage growth.

    Args:
        base_path: Checkpoint directory for this run (local or s3://).
    """
    logger.info("Cleaning checkpoints at %s", base_path)
    for f in _ls(
        base_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    ):
        _delete(
            f,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_secret_access_key=seaweedfs_secret_access_key,
        )
    logger.info("Checkpoints cleaned: %s", base_path)


def list_checkpoints(
    base_path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
) -> list[int]:
    """
    Return a sorted list of completed epoch numbers found at base_path.
    Useful for inspection and debugging.
    """
    all_files = _ls(
        base_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_secret_access_key=seaweedfs_secret_access_key,
    )
    epochs = []
    for f in all_files:
        match = re.search(r"epoch_(\d{4})\.done$", f)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def get_zenml_step_checkpoint_path(base_path: str, namespace: str | None = None) -> str:
    """
    Return the checkpoint path for the active pipeline run.

    Args:
        base_path: Base directory for checkpoints (local path or s3:// URI).

    Returns:
        The full checkpoint path for the active run, optionally namespaced.
    """
    ctx = get_step_context()
    run_id = ctx.pipeline_run.id
    if namespace:
        return f"{base_path}/{run_id}/{namespace}"
    return f"{base_path}/{run_id}"
