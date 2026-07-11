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
from zenml import step

from workflows.matrix_factorization.configs import (
    CFG_FEATURES_FIELD_NAMES,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    trial: optuna.Trial,
) -> float:
    """
    Train ALS on a subsample and return final validation RMSE.
    Reports intermediate RMSE per epoch for Hyperband pruning.
    """
    from workflows.matrix_factorization.utils.als_numba import (
        compute_rmse_block,
        solve_user_factors,
    )

    n_users = int(train_pd[CFG_FEATURES_FIELD_NAMES.USER_ID.value].max()) + 1
    n_items = int(train_pd[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value].max()) + 1

    # Build dense rating matrix (small subsample, fits in memory)
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for _, row in train_pd.iterrows():
        R[
            int(row[CFG_FEATURES_FIELD_NAMES.USER_ID.value]),
            int(row[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value]),
        ] = float(row[CFG_FEATURES_FIELD_NAMES.RATING.value])

    rng = np.random.default_rng(42)
    user_factors = rng.standard_normal((n_users, rank)).astype(np.float32) * 0.01
    item_factors = rng.standard_normal((n_items, rank)).astype(np.float32) * 0.01

    rmse = float("inf")
    for epoch in range(n_iter):
        user_factors = solve_user_factors(R, item_factors, regularization, alpha)
        item_factors = solve_user_factors(R.T, user_factors, regularization, alpha)

        # Compute val RMSE for this epoch
        u_idx = np.asarray(val_pd[CFG_FEATURES_FIELD_NAMES.USER_ID.value], dtype=np.int32)
        i_idx = np.asarray(val_pd[CFG_FEATURES_FIELD_NAMES.ITEM_ID.value], dtype=np.int32)
        # Clip indices to factor matrix bounds (subsample may not cover all IDs)
        u_idx = np.clip(u_idx, 0, n_users - 1)
        i_idx = np.clip(i_idx, 0, n_items - 1)
        r = np.asarray(val_pd[CFG_FEATURES_FIELD_NAMES.RATING.value], dtype=np.float32)

        sse, count = compute_rmse_block(u_idx, i_idx, r, user_factors, item_factors)
        rmse = float(np.sqrt(sse / count)) if count > 0 else float("inf")

        trial.report(rmse, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return rmse


@step(enable_cache=False)
def run_hpo_trial(
    trial_idx: int,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
) -> Annotated[dict, "trial_result"]:
    """
    Run a single Optuna HPO trial. Multiple instances run in parallel via ZenML fan-out.

    Args:
        trial_idx: Index of this trial (used for logging and random seed offset).
        train_data: Training ratings pandas DataFrame.
        val_data: Validation ratings pandas DataFrame.
        hpo_subsample_fraction: Fraction of training data to use for this trial.
        optuna_storage: Optuna storage URI (SQLite or database URL).
        optuna_study_name: Optuna study name (used to resume existing studies).

    Returns:
        trial_result dict: {trial_idx, value, params}
    """
    train_pd = train_data
    val_pd = val_data

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
        return _train_als_subsample(train_pd, val_pd, rank, regularization, alpha, n_iter, trial)

    study.optimize(objective, n_trials=1)

    logger.info(
        "Trial %d complete. Best value so far: %.4f, params: %s",
        trial_idx,
        study.best_value,
        study.best_params,
    )

    return {
        "trial_idx": trial_idx,
        "value": study.best_value,
        "params": study.best_params,
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
