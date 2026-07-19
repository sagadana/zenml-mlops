"""
steps/hpo/run_hpo.py

ZenML steps: run_hpo_trial, collect_best_hpo_params

Hyperparameter optimization for the ALS model using Optuna.
Each Optuna trial trains ALS on a subsample of the training data and
reports per-epoch val RMSE (enabling early pruning via HyperbandPruner).

Resumability: SQLite or PostgreSQL storage with load_if_exists=True means
the study persists across restarts. Interrupted studies resume automatically.

Parallelism: Each trial runs as an independent ZenML step via fan-out.
"""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel
from zenml import log_metadata, step
from zenml.client import Client

from helpers.checkpointing import (
    clean_run_checkpoints,
    get_zenml_step_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from helpers.s3_client import resolve_zenml_s3_credentials
from workflows.matrix_factorization.models.base_recommender import (
    BaseRecommender,
    Hyperparameters,
    load_recommender_class,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

HPO_SPACES = {
    "rank": (10, 100),
    "regularization": (1e-3, 10.0),
    "alpha": (1e-3, 1.0),
    "n_iter": (50, 200),
}


class TrialResult(BaseModel):
    idx: int
    value: float | None
    params: Hyperparameters
    skipped: bool = False


class HPOResult(BaseModel):
    best_params: Hyperparameters
    best_val_rmse: float
    n_trials: int


def _make_storage(optuna_storage: str):
    """Return an Optuna storage object from a URI string (SQLite or database URL)."""
    return optuna_storage


def _train_als_subsample(
    train_pd: pd.DataFrame,
    val_pd: pd.DataFrame,
    rank: int,
    regularization: float,
    alpha: float,
    n_iter: int,
    n_workers: int,
    trial: optuna.Trial,
    recommender_cls: type[BaseRecommender],
) -> float:
    """
    Train on a subsample and return the final validation RMSE for this trial.
    """

    def _on_epoch_end(epoch: int, loss: float) -> None:
        trial.report(loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    _, _, states = recommender_cls.train(
        train_data=train_pd,
        val_data=val_pd,
        rank=rank,
        regularization=regularization,
        alpha=alpha,
        n_iter=n_iter,
        n_workers=n_workers,
        seed=42,
        eval_every_n_epochs=1,
        epoch_end_callback=lambda state: _on_epoch_end(state.epoch, state.loss),
    )

    return states[-1].loss


@step(
    enable_cache=False, experiment_tracker=experiment_tracker.name if experiment_tracker else None
)
def run_hpo_trial(
    trial_idx: int,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    n_workers: int = 4,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
    recommender_class_name: str = "workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender",
    checkpoint_path: str = "./checkpoints",
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
    autoresume: bool = True,
) -> Annotated[TrialResult, "trial_result"]:
    """
    Run a single Optuna HPO trial. Multiple instances run in parallel via ZenML fan-out.

    Args:
        trial_idx: Index of this trial (used for logging and random seed offset).
        train_data: Training ratings pandas DataFrame.
        val_data: Validation ratings pandas DataFrame.
        n_workers: Number of parallel partition workers.
        hpo_subsample_fraction: Fraction of training data to use for this trial.
        optuna_storage: Optuna storage URI (SQLite or database URL).
        optuna_study_name: Optuna study name (used to resume existing studies).
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass to train.
        checkpoint_path: Base path for pipeline-run checkpoints.
        seaweedfs_s3_internal_endpoint: SeaweedFS internal S3 endpoint (local stack).
        zenml_local_s3_secret_name: ZenML secret name containing SeaweedFS credentials (local stack).
        autoresume: If True, skip trial if a checkpoint for trial_idx already exists.

    Returns:
        trial_result dict: {trial_idx, value, params}
    """
    recommender_cls: type[BaseRecommender] = load_recommender_class(recommender_class_name)

    train_pd = train_data
    val_pd = val_data

    # --- Resolve credentials and checkpoint path (used for resume check and save) ---

    access_key_id, secret_access_key = resolve_zenml_s3_credentials(zenml_local_s3_secret_name)
    hpo_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="hpo")

    # --- Check if this trial has already been checkpointed (resumable) ---

    if autoresume:
        try:
            primary, secondary = load_checkpoint(
                epoch=trial_idx + 1,
                base_path=hpo_checkpoint_path,
                seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
                seaweedfs_access_key_id=access_key_id,
                seaweedfs_secret_access_key=secret_access_key,
            )
            if primary is not None and secondary is not None and len(secondary) >= 4:
                logger.info("Trial %d already checkpointed. Skipping.", trial_idx)
                log_metadata(
                    metadata={
                        "hpo_checkpoint_path": hpo_checkpoint_path,
                        "hpo_latest_checkpointed_epoch": trial_idx + 1,
                    }
                )
                return TrialResult(
                    idx=trial_idx,
                    value=float(primary[0]),
                    params=Hyperparameters(
                        rank=int(secondary[0]),
                        regularization=float(secondary[1]),
                        alpha=float(secondary[2]),
                        n_iter=int(secondary[3]),
                    ),
                    skipped=True,
                )
        except Exception as exc:
            logger.warning(
                "Failed to load HPO checkpoint for trial %d: %s. Proceeding with trial.",
                trial_idx,
                exc,
            )

    if hpo_subsample_fraction < 1.0:
        train_pd = train_pd.sample(frac=hpo_subsample_fraction, random_state=42 + trial_idx)

    logger.info(
        "Trial %d: %d training ratings, %d val ratings",
        trial_idx,
        len(train_pd),
        len(val_pd),
    )

    # --- Create or load Optuna study (resumable) and run a single trial ---

    study = optuna.create_study(
        study_name=optuna_study_name,
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=3),
        sampler=optuna.samplers.TPESampler(constant_liar=True, seed=42),
        load_if_exists=True,
        storage=_make_storage(optuna_storage),
    )

    def objective(trial: optuna.Trial) -> float:
        rank = trial.suggest_int("rank", HPO_SPACES["rank"][0], HPO_SPACES["rank"][1])
        regularization = trial.suggest_float(
            "regularization",
            HPO_SPACES["regularization"][0],
            HPO_SPACES["regularization"][1],
            log=True,
        )
        alpha = trial.suggest_float(
            "alpha", HPO_SPACES["alpha"][0], HPO_SPACES["alpha"][1], log=True
        )
        n_iter = trial.suggest_int("n_iter", HPO_SPACES["n_iter"][0], HPO_SPACES["n_iter"][1])
        return _train_als_subsample(
            train_pd, val_pd, rank, regularization, alpha, n_iter, n_workers, trial, recommender_cls
        )

    study.optimize(objective, n_trials=1)
    best_params = study.best_params

    logger.info(
        "Trial %d complete. Best value so far: %.4f, params: %s",
        trial_idx,
        study.best_value,
        best_params,
    )

    result_params = Hyperparameters(
        rank=int(best_params["rank"]),
        regularization=float(best_params["regularization"]),
        alpha=float(best_params["alpha"]),
        n_iter=int(best_params["n_iter"]),
    )

    # ---- Save trial checkpoint to ZenML step checkpoint path (SeaweedFS S3) ----

    save_checkpoint(
        epoch=trial_idx + 1,
        primary=np.array([float(study.best_value)], dtype=np.float64),
        secondary=np.array(
            [
                float(result_params.rank),
                float(result_params.regularization),
                float(result_params.alpha),
                float(result_params.n_iter),
            ],
            dtype=np.float64,
        ),
        base_path=hpo_checkpoint_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )
    log_metadata(
        metadata={
            "hpo_checkpoint_path": hpo_checkpoint_path,
            "hpo_trial_idx": trial_idx,
            "hpo_checkpoint_epoch": trial_idx + 1,
        }
    )

    return TrialResult(
        idx=trial_idx,
        value=study.best_value,
        params=result_params,
    )


@step(enable_cache=False)
def collect_best_hpo_params(
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
) -> Annotated[Hyperparameters, "best_hyperparams"]:
    """
    Fan-in: load Optuna study and return best hyperparameters across all trials.

    Args:
        optuna_storage: Optuna storage URI (must match the URI used in run_hpo_trial).
        optuna_study_name: Optuna study name.

    Returns:
        best_hyperparams: Hyperparameters dataclass with the best hyperparameters found across all trials.
    """
    study = optuna.load_study(
        study_name=optuna_study_name,
        storage=_make_storage(optuna_storage),
    )

    best = study.best_params
    logger.info(
        "HPO complete. Best params: %s (val RMSE=%.4f) across %d trials",
        best,
        study.best_value,
        len(study.trials),
    )

    log_metadata(
        metadata={
            "hpo_best_params": best,
            "hpo_best_val_rmse": study.best_value,
            "hpo_n_trials": len(study.trials),
        }
    )

    return Hyperparameters(
        rank=int(best["rank"]),
        regularization=float(best["regularization"]),
        alpha=float(best["alpha"]),
        n_iter=int(best["n_iter"]),
    )


@step(enable_cache=False)
def cleanup_hpo_checkpoints(
    checkpoint_path: str,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> None:
    """Cleanup HPO checkpoints for this pipeline run."""
    access_key_id, secret_access_key = resolve_zenml_s3_credentials(zenml_local_s3_secret_name)
    hpo_checkpoint_path = get_zenml_step_checkpoint_path(checkpoint_path, namespace="hpo")
    clean_run_checkpoints(
        hpo_checkpoint_path,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id=access_key_id,
        seaweedfs_secret_access_key=secret_access_key,
    )
