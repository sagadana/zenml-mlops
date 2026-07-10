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

logger = logging.getLogger(__name__)

# s3fs is imported lazily so local-only usage doesn't require AWS credentials
_s3fs = None


def _get_fs(path: str):
    """Return (filesystem, path_str) for local or S3 paths."""
    if path.startswith("s3://"):
        global _s3fs
        if _s3fs is None:
            import s3fs as _s3fs_lib

            _s3fs = _s3fs_lib.S3FileSystem(anon=False)
        return _s3fs, path
    return None, path  # None signals "use local pathlib"


def _ls(path: str) -> list[str]:
    """List files in a directory (local or S3). Returns full paths."""
    fs, p = _get_fs(path)
    if fs is not None:
        try:
            return [f"s3://{f}" for f in fs.ls(p, detail=False)]
        except FileNotFoundError:
            return []
    local = Path(p)
    if not local.exists():
        return []
    return [str(f) for f in local.iterdir()]


def _makedirs(path: str) -> None:
    """Create directory (local only; S3 has no real directories)."""
    fs, p = _get_fs(path)
    if fs is None:
        Path(p).mkdir(parents=True, exist_ok=True)


def _save_npy(path: str, array: np.ndarray) -> None:
    """Save numpy array to local or S3 path."""
    fs, p = _get_fs(path)
    if fs is not None:
        buf = io.BytesIO()
        np.save(buf, array)
        buf.seek(0)
        with fs.open(p, "wb") as f:
            f.write(str(buf.read()))
    else:
        np.save(p, array)


def _load_npy(path: str) -> np.ndarray:
    """Load numpy array from local or S3 path."""
    fs, p = _get_fs(path)
    if fs is not None:
        with fs.open(p, "rb") as f:
            return np.load(f.read())
    return np.load(p)


def _write_marker(path: str) -> None:
    """Write an empty .done marker file."""
    fs, p = _get_fs(path)
    if fs is not None:
        with fs.open(p, "w") as f:
            f.write("")
    else:
        Path(p).write_text("")


def _exists(path: str) -> bool:
    fs, p = _get_fs(path)
    if fs is not None:
        return fs.exists(p)
    return Path(p).exists()


def _delete(path: str) -> None:
    fs, p = _get_fs(path)
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
    _makedirs(base_path)
    prefix = f"{base_path}/epoch_{epoch:04d}"

    logger.debug("Saving checkpoint epoch %d to %s", epoch, base_path)
    _save_npy(f"{prefix}_primary.npy", primary)
    if secondary is not None:
        _save_npy(f"{prefix}_secondary.npy", secondary)
    _write_marker(f"{prefix}.done")  # atomic commit — written last
    logger.info("Checkpoint saved: epoch %d (%s)", epoch, base_path)


def load_latest_checkpoint(
    base_path: str,
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
    all_files = _ls(base_path)
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

    if not _exists(primary_path):
        logger.warning(
            "Checkpoint epoch %d has .done marker but missing primary .npy — "
            "falling back to epoch %d",
            latest_epoch,
            latest_epoch - 1,
        )
        _delete(latest_done)
        return load_latest_checkpoint(base_path)

    logger.info("Resuming from checkpoint epoch %d (%s)", latest_epoch, base_path)
    primary = _load_npy(primary_path)
    secondary_path = f"{prefix}_secondary.npy"
    secondary = _load_npy(secondary_path) if _exists(secondary_path) else None
    return latest_epoch, primary, secondary


def clean_run_checkpoints(base_path: str) -> None:
    """
    Delete all checkpoint files for a completed training run.
    Call this after model registration succeeds to prevent unbounded storage growth.

    Args:
        base_path: Checkpoint directory for this run (local or s3://).
    """
    logger.info("Cleaning checkpoints at %s", base_path)
    for f in _ls(base_path):
        _delete(f)
    logger.info("Checkpoints cleaned: %s", base_path)


def list_checkpoints(base_path: str) -> list[int]:
    """
    Return a sorted list of completed epoch numbers found at base_path.
    Useful for inspection and debugging.
    """
    all_files = _ls(base_path)
    epochs = []
    for f in all_files:
        match = re.search(r"epoch_(\d{4})\.done$", f)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)
