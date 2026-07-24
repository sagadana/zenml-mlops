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

import functools
import logging
from typing import Annotated, Literal

import numpy as np
import optuna
import pandas as pd
from pydantic import BaseModel
from zenml import get_step_context, log_metadata, step
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
    EpochState,
    Hyperparameters,
    load_recommender_class,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

type HPOMetric = Literal["loss", "precision", "recall", "ndcg"]

HYPERBAND_PRUNER_MAX_RESOURCE = 60  # epochs
HYPERBAND_REDUCTION_FACTOR = 2  # halve the number of trials at each rung

HPO_SPACES = {
    "rank": (10, 100),
    "regularization": (1e-3, 1.0),
    "alpha": (1.0, 40.0),
    # Keep n_iter within the HyperbandPruner max_resource budget
    # so the pruner can actually make early-stopping decisions. The final
    # training n_iter is controlled separately in the pipeline config.
    "n_iter": (5, HYPERBAND_PRUNER_MAX_RESOURCE),
}

# Maps hpo_metric name → Optuna study direction.
_METRIC_DIRECTION: dict[HPOMetric, str] = {
    "loss": "minimize",
    "precision": "maximize",
    "recall": "maximize",
    "ndcg": "maximize",
}


class TrialResult(BaseModel):
    _study: str
    idx: int
    value: float | None
    params: Hyperparameters
    skipped: bool = False
    metric: HPOMetric


@functools.lru_cache(maxsize=1)
def _create_study(
    optuna_storage: str, optuna_study_name: str, direction: str = "minimize", seed: int = 42
) -> optuna.Study:
    """Create or load an Optuna study with Hyperband pruning and TPE sampling."""
    return optuna.create_study(
        study_name=optuna_study_name,
        direction=direction,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=HYPERBAND_PRUNER_MAX_RESOURCE, reduction_factor=2
        ),
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
        storage=optuna_storage,
    )


def _get_study_name(optuna_study_name: str) -> str:
    """Get the Optuna study name for this pipeline run."""
    step = get_step_context()
    return f"{optuna_study_name}_{step.pipeline_run.id}"


def _get_metric_value(state: EpochState, hpo_metric: str) -> float:
    """Extract the HPO objective value from an epoch state."""
    mapping: dict[str, float] = {
        "loss": state.loss,
        "precision": state.precision_at_k,
        "recall": state.recall_at_k,
        "ndcg": state.ndcg_at_k,
    }
    if hpo_metric not in mapping:
        raise ValueError(f"Unknown hpo_metric: {hpo_metric!r}. Choose from {list(mapping)}")
    return mapping[hpo_metric]


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
    hpo_metric: str = "loss",
) -> float:
    """
    Train on a subsample and return the final value of hpo_metric for this trial.
    """

    def _on_epoch_end(state: EpochState) -> None:
        trial.report(_get_metric_value(state, hpo_metric), step=state.epoch)
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
        epoch_end_callback=_on_epoch_end,
    )

    return _get_metric_value(states[-1], hpo_metric)


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
    hpo_metric: HPOMetric = "loss",
    seed: int = 42,
) -> Annotated[TrialResult, "trial_result"]:
    """
    Run a single Optuna HPO trial. Multiple instances run in parallel via ZenML fan-out.

    Args:
        trial_idx: Index of this trial
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
        hpo_metric: Metric to optimise. One of: ``loss`` (minimise), ``precision``,
            ``recall``, ``ndcg`` (all maximise).
        seed: Random seed for reproducibility (used for subsampling and Optuna TPE sampler).

    Returns:
        trial_result dict: {trial_idx, value, params}
    """
    recommender_cls: type[BaseRecommender] = load_recommender_class(recommender_class_name)

    train_pd = train_data
    val_pd = val_data
    study_name = _get_study_name(optuna_study_name)

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
                    _study=study_name,
                    idx=trial_idx,
                    value=float(primary[0]),
                    params=Hyperparameters(
                        rank=int(secondary[0]),
                        regularization=float(secondary[1]),
                        alpha=float(secondary[2]),
                        n_iter=int(secondary[3]),
                    ),
                    skipped=True,
                    metric=hpo_metric,
                )
        except Exception as exc:
            logger.warning(
                "Failed to load HPO checkpoint for trial %d: %s. Proceeding with trial.",
                trial_idx,
                exc,
            )

    if hpo_subsample_fraction < 1.0:
        train_pd = train_pd.sample(frac=hpo_subsample_fraction, random_state=seed)
        val_pd = val_pd.sample(frac=hpo_subsample_fraction, random_state=seed)

    logger.info(
        "Trial %d: %d training ratings, %d val ratings",
        trial_idx,
        len(train_pd),
        len(val_pd),
    )

    # --- Create or load Optuna study (resumable) and run a single trial ---
    if hpo_metric not in _METRIC_DIRECTION:
        raise ValueError(
            f"Unknown hpo_metric: {hpo_metric!r}. Choose from {list(_METRIC_DIRECTION)}"
        )
    direction = _METRIC_DIRECTION[hpo_metric]
    study = _create_study(optuna_storage, study_name, direction, seed=seed)

    def objective(trial: optuna.Trial) -> float:
        rank = trial.suggest_int("rank", HPO_SPACES["rank"][0], HPO_SPACES["rank"][1])
        regularization = trial.suggest_float(
            "regularization",
            HPO_SPACES["regularization"][0],
            HPO_SPACES["regularization"][1],
            log=True,
        )
        alpha = float(
            trial.suggest_int("alpha", int(HPO_SPACES["alpha"][0]), int(HPO_SPACES["alpha"][1]))
        )
        n_iter = trial.suggest_int("n_iter", HPO_SPACES["n_iter"][0], HPO_SPACES["n_iter"][1])
        return _train_als_subsample(
            train_pd,
            val_pd,
            rank,
            regularization,
            alpha,
            n_iter,
            n_workers,
            trial,
            recommender_cls,
            hpo_metric=hpo_metric,
        )

    study.optimize(objective, n_trials=1)
    result_params = Hyperparameters.model_validate(study.best_params)

    logger.info(
        "Trial %d complete. Best value so far: %.4f, params: %s",
        trial_idx,
        study.best_value,
        result_params,
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
        _study=study_name,
        idx=trial_idx,
        value=study.best_value,
        params=result_params,
        metric=hpo_metric,
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
    study_name = _get_study_name(optuna_study_name)
    study = optuna.load_study(
        study_name=study_name,
        storage=optuna_storage,
    )

    best = study.best_params
    logger.info(
        "HPO complete. Best params: %s (best_loss=%.4f) across %d trials",
        best,
        study.best_value,
        len(study.trials),
    )

    log_metadata(
        metadata={
            "hpo_best_params": best,
            "hpo_best_value": study.best_value,
            "hpo_n_trials": len(study.trials),
        }
    )

    return Hyperparameters.model_validate(best)


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
