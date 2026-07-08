"""
tests/unit/test_checkpointing.py

Unit tests for the shared helpers.checkpointing module.
Tests save/load roundtrip, atomic .done marker protocol, and resume logic.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from helpers.checkpointing import (
    clean_run_checkpoints,
    list_checkpoints,
    load_latest_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def checkpoint_dir(tmp_path):
    return str(tmp_path / "checkpoints" / "test_run")


def make_factors(n_users=10, n_items=15, rank=8, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_users, rank)).astype(np.float32),
        rng.standard_normal((n_items, rank)).astype(np.float32),
    )


def test_save_creates_three_files(checkpoint_dir):
    user_f, item_f = make_factors()
    save_checkpoint(1, user_f, item_f, checkpoint_dir)

    files = list(Path(checkpoint_dir).iterdir())
    names = {f.name for f in files}
    assert "epoch_0001_primary.npy" in names
    assert "epoch_0001_secondary.npy" in names
    assert "epoch_0001.done" in names


def test_load_returns_correct_epoch(checkpoint_dir):
    user_f, item_f = make_factors()
    save_checkpoint(3, user_f, item_f, checkpoint_dir)

    start_epoch, loaded_users, loaded_items = load_latest_checkpoint(checkpoint_dir)
    assert start_epoch == 3
    np.testing.assert_array_equal(loaded_users, user_f)
    np.testing.assert_array_equal(loaded_items, item_f)


def test_load_empty_dir_returns_zero(checkpoint_dir):
    start, users, items = load_latest_checkpoint(checkpoint_dir)
    assert start == 0
    assert users is None
    assert items is None


def test_load_picks_latest_epoch(checkpoint_dir):
    for epoch in range(1, 6):
        u, it = make_factors(seed=epoch)
        save_checkpoint(epoch, u, it, checkpoint_dir)

    start, loaded_users, loaded_items = load_latest_checkpoint(checkpoint_dir)
    expected_u, expected_it = make_factors(seed=5)
    assert start == 5
    np.testing.assert_array_equal(loaded_users, expected_u)
    np.testing.assert_array_equal(loaded_items, expected_it)


def test_missing_done_marker_ignored(checkpoint_dir):
    """If .done is missing (simulating crash), that epoch should be ignored."""
    u1, i1 = make_factors(seed=1)
    u2, i2 = make_factors(seed=2)

    # Save epoch 1 fully
    save_checkpoint(1, u1, i1, checkpoint_dir)

    # Simulate epoch 2 crash: write .npy files but not .done
    base = Path(checkpoint_dir)
    np.save(str(base / "epoch_0002_primary.npy"), u2)
    np.save(str(base / "epoch_0002_secondary.npy"), i2)
    # .done intentionally NOT written

    start, loaded_users, loaded_items = load_latest_checkpoint(checkpoint_dir)
    assert start == 1  # should fall back to epoch 1
    np.testing.assert_array_equal(loaded_users, u1)


def test_list_checkpoints(checkpoint_dir):
    for epoch in [1, 3, 5]:
        u, it = make_factors(seed=epoch)
        save_checkpoint(epoch, u, it, checkpoint_dir)

    epochs = list_checkpoints(checkpoint_dir)
    assert epochs == [1, 3, 5]


def test_clean_removes_all_files(checkpoint_dir):
    for epoch in range(1, 4):
        u, it = make_factors(seed=epoch)
        save_checkpoint(epoch, u, it, checkpoint_dir)

    clean_run_checkpoints(checkpoint_dir)

    remaining = list(Path(checkpoint_dir).iterdir())
    assert len(remaining) == 0


def test_resume_after_multiple_epochs(checkpoint_dir):
    """Simulate a multi-epoch training run with an intermediate load."""
    # "Train" epochs 1–5
    for epoch in range(1, 6):
        u, it = make_factors(seed=epoch)
        save_checkpoint(epoch, u, it, checkpoint_dir)

    # "Crash" — reload state
    start, loaded_users, loaded_items = load_latest_checkpoint(checkpoint_dir)
    assert start == 5

    # "Resume" from epoch 5 → train epochs 6–10
    for epoch in range(start, 10):
        u, it = make_factors(seed=epoch + 1)
        save_checkpoint(epoch + 1, u, it, checkpoint_dir)

    final_start, _, _ = load_latest_checkpoint(checkpoint_dir)
    assert final_start == 10
