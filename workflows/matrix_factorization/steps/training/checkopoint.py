from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from zenml import log_metadata, step

from helpers.checkpointing import (
    clean_run_checkpoints,
    get_zenml_step_checkpoint_path,
    list_checkpoints,
    load_latest_checkpoint,
    save_checkpoint,
)
from workflows.matrix_factorization.configs import CFG_FEATURES_FIELD_NAMES
from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logger = logging.getLogger(__name__)


@step(enable_cache=False)
def load_or_init_training_factors(
    train_data: pd.DataFrame,
    best_hyperparams: dict,
    checkpoint_path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_access_key_secret: str | None = None,
    autoresume: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load latest training checkpoint or initialize fresh ALS factors."""
    from workflows.matrix_factorization.utils.als_numba import warmup_jit

    training_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="training")
    if autoresume:
        latest_epoch, user_factors, item_factors = load_latest_checkpoint(
            training_checkpoint_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_access_key_secret=seaweedfs_access_key_secret,
        )
        if user_factors is not None and item_factors is not None:
            log_metadata(
                metadata={
                    "training_checkpoint_path": training_checkpoint_path,
                    "training_resume_epoch": latest_epoch,
                    "training_autoresume": True,
                }
            )
            return user_factors, item_factors, latest_epoch

    rank = int(best_hyperparams.get("rank", 50))
    n_users = int(train_data[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
    n_items = int(train_data[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1
    warmup_jit(rank=min(rank, 20))
    user_factors, item_factors = ALSRecommender.initialize_factors(
        n_users=n_users,
        n_items=n_items,
        rank=rank,
        seed=42,
    )
    log_metadata(
        metadata={
            "training_checkpoint_path": training_checkpoint_path,
            "training_resume_epoch": 0,
            "training_autoresume": False,
        }
    )
    return user_factors, item_factors, 0


@step(enable_cache=False)
def save_training_checkpoint(
    checkpoint_path: str,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    epoch: int,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_access_key_secret: str | None = None,
) -> int:
    """Add the training checkpoint path to the step metadata."""
    training_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="training")

    save_checkpoint(
        epoch=epoch + 1,
        primary=user_factors,
        secondary=item_factors,
        base_path=training_checkpoint_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_access_key_secret=seaweedfs_access_key_secret,
    )

    log_metadata(
        metadata={
            "training_checkpoint_path": training_checkpoint_path,
            "training_checkpoint_epoch": epoch + 1,
        }
    )
    return epoch + 1


@step(enable_cache=False)
def load_hpo_checkpoints(
   checkpoint_path: str,
   seaweedfs_s3_internal_endpoint: str | None = None,
   seaweedfs_access_key_id: str | None = None,
   seaweedfs_access_key_secret: str | None = None,
   autoresume: bool = True,
) -> list[int]:
    """Load completed HPO checkpoint epochs for the current pipeline run."""
    if not autoresume:
        return []
    try:
        hpo_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="hpo")
        checkpointed_trials = list_checkpoints(
            hpo_checkpoint_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_access_key_secret=seaweedfs_access_key_secret,
        )
        log_metadata(
            metadata={
                "hpo_checkpoint_path": hpo_checkpoint_path,
                "hpo_checkpointed_trials_count": len(checkpointed_trials),
            }
        )
        return checkpointed_trials
    except Exception as exc:
        logger.warning("Failed to load HPO checkpoints: %s", exc)
        return []


@step(enable_cache=False)
def save_hpo_trial_checkpoint(
    checkpoint_path: str,
    trial_result: dict,
    trial_idx: int,
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_access_key_secret: str | None = None,
) -> int:
    """Save HPO trial checkpoint if trial executed."""
    if trial_result.get("value") is None:
        return trial_idx + 1
    hpo_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="hpo")
    params = trial_result.get("params", {})
    save_checkpoint(
        epoch=trial_idx + 1,
        primary=np.array([float(trial_result["value"])], dtype=np.float64),
        secondary=np.array(
            [
                float(params["rank"]),
                float(params["regularization"]),
                float(params["alpha"]),
                float(params["n_iter"]),
            ],
            dtype=np.float64,
        ),
        base_path=hpo_checkpoint_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_access_key_secret=seaweedfs_access_key_secret,
    )
    log_metadata(
        metadata={
            "hpo_checkpoint_path": hpo_checkpoint_path,
            "hpo_trial_idx": trial_idx,
            "hpo_checkpoint_epoch": trial_idx + 1,
        }
    )
    return trial_idx + 1


@step(enable_cache=False)
def cleanup_pipeline_checkpoints(
   checkpoint_path: str,
   seaweedfs_s3_internal_endpoint: str | None = None,
   seaweedfs_access_key_id: str | None = None,
   seaweedfs_access_key_secret: str | None = None,
   enable_hpo: bool = False,
) -> None:
    """Cleanup training/HPO checkpoints for this pipeline run."""
    training_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="training")
    clean_run_checkpoints(
        training_checkpoint_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=seaweedfs_access_key_id,
        seaweedfs_access_key_secret=seaweedfs_access_key_secret,
    )
    if enable_hpo:
        hpo_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="hpo")
        clean_run_checkpoints(
            hpo_checkpoint_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            seaweedfs_access_key_id=seaweedfs_access_key_id,
            seaweedfs_access_key_secret=seaweedfs_access_key_secret,
        )
