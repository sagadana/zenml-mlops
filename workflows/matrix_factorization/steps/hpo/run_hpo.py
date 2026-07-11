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

import optuna
import pandas as pd
from zenml import step
from zenml.client import Client

from workflows.matrix_factorization.models.als_recommender import ALSRecommender

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker


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
) -> float:
    """
    Train ALS on a subsample and return final validation RMSE.
    Reports intermediate RMSE per epoch for Hyperband pruning.
    """

    def _on_epoch_end(epoch: int, rmse: float) -> None:
        trial.report(rmse, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    _, _, rmse = ALSRecommender.train(
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
    return rmse


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
    hpo_checkpoint_epochs: list[int] | None = None,
) -> Annotated[dict, "trial_result"]:
    """
    Run a single Optuna HPO trial. Multiple instances run in parallel via ZenML fan-out.

    Args:
        trial_idx: Index of this trial (used for logging and random seed offset).
        train_data: Training ratings pandas DataFrame.
        val_data: Validation ratings pandas DataFrame.
        already_checkpointed: If True, skip running this trial.
        n_workers: Number of parallel partition workers (ProcessPoolExecutor).
        hpo_subsample_fraction: Fraction of training data to use for this trial.
        optuna_storage: Optuna storage URI (SQLite or database URL).
        optuna_study_name: Optuna study name (used to resume existing studies).

    Returns:
        trial_result dict: {trial_idx, value, params}
    """
    if hpo_checkpoint_epochs is None:
        hpo_checkpoint_epochs = []

    train_pd = train_data
    val_pd = val_data

    if trial_idx in hpo_checkpoint_epochs:
        logger.info("Trial %d already checkpointed. Skipping.", trial_idx)
        return {"trial_idx": trial_idx, "value": None, "params": {}}

    if hpo_subsample_fraction < 1.0:
        train_pd = train_pd.sample(frac=hpo_subsample_fraction, random_state=42 + trial_idx)

    logger.info(
        "Trial %d: %d training ratings, %d val ratings",
        trial_idx,
        len(train_pd),
        len(val_pd),
    )

    study = optuna.create_study(
        study_name=optuna_study_name,
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=3),
        sampler=optuna.samplers.TPESampler(constant_liar=True, seed=42),
        load_if_exists=True,
        storage=_make_storage(optuna_storage),
    )

    def objective(trial: optuna.Trial) -> float:
        rank = trial.suggest_int("rank", 10, 200)
        regularization = trial.suggest_float("regularization", 1e-3, 10.0, log=True)
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        n_iter = trial.suggest_int("n_iter", 5, 25)
        return _train_als_subsample(
            train_pd, val_pd, rank, regularization, alpha, n_iter, n_workers, trial
        )

    study.optimize(objective, n_trials=1)
    best_params = study.best_params

    logger.info(
        "Trial %d complete. Best value so far: %.4f, params: %s",
        trial_idx,
        study.best_value,
        best_params,
    )

    return {
        "trial_idx": trial_idx,
        "value": study.best_value,
        "params": best_params,
    }


@step(enable_cache=False)
def collect_best_hpo_params(
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
) -> Annotated[dict, "best_hyperparams"]:
    """
    Fan-in: load Optuna study and return best hyperparameters across all trials.

    Args:
        optuna_storage: Optuna storage URI (must match the URI used in run_hpo_trial).
        optuna_study_name: Optuna study name.

    Returns:
        best_hyperparams dict: {rank, regularization, alpha, n_iter, best_val_rmse, n_trials}
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

    return {
        "rank": int(best["rank"]),
        "regularization": float(best["regularization"]),
        "alpha": float(best["alpha"]),
        "n_iter": int(best["n_iter"]),
        "best_val_rmse": float(study.best_value),
        "n_trials": len(study.trials),
    }
