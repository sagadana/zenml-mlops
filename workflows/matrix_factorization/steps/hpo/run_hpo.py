"""
steps/hpo/run_hpo.py

ZenML step: run_hpo

Distributed hyperparameter optimization for the ALS model using Optuna.
Each Optuna trial trains ALS on a 20% subsample of the training data and
reports per-epoch val RMSE (enabling early pruning via HyperbandPruner).

Resumability: SQLite or PostgreSQL storage with load_if_exists=True means
the study persists across restarts. Interrupted studies resume automatically.

Parallelism: Each trial is submitted as a Dask future via
client.submit(study.optimize, objective, n_trials=1, pure=False).
"""

from __future__ import annotations

import logging
from typing import Annotated

import dask_expr as dd
import numpy as np
import optuna
import pandas as pd
from zenml import step

from helpers.dask_cluster import get_client_mode_from_config, get_dask_client
from workflows.matrix_factorization.configs import (
    CFG_DASK_SCHEDULER_ADDRESS,
    CFG_FEATURES_FIELD_NAMES,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
def run_hpo(
    train_data: dd.DataFrame,
    val_data: dd.DataFrame,
    hpo_n_trials: int = 20,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "mysql+pymysql://ops:ops@127.0.0.1:3306/optuna",
    optuna_study_name: str = "als_movielens",
    dask_scheduler_address: str | None = CFG_DASK_SCHEDULER_ADDRESS,
) -> Annotated[dict, "best_hyperparams"]:
    """
    Run distributed hyperparameter optimization for ALS.

    Args:
        train_data: Training split Dask DataFrame.
        val_data: Validation split Dask DataFrame.
        hpo_n_trials: Total number of Optuna trials.
        hpo_subsample_fraction: Fraction of training data to use per trial.
        optuna_storage: Optuna storage URI (MySQL for local ops-db, or full connection string for AWS).
            Setting load_if_exists=True makes this resumable on restart.
        optuna_study_name: Optuna study name (used to resume existing studies).

    Returns:
        best_hyperparams dict: {rank, regularization, alpha, n_iter}
    """
    # Materialize and subsample training data
    train_pd = train_data.compute()
    val_pd = val_data.compute()

    if hpo_subsample_fraction < 1.0:
        train_pd = train_pd.sample(frac=hpo_subsample_fraction, random_state=42)

    logger.info("HPO subsample: %d training ratings, %d val ratings", len(train_pd), len(val_pd))

    storage_kwargs: dict = {}
    if optuna_storage.startswith("sqlite://"):
        # SQLite — single-process storage, no DaskStorage wrapper needed
        storage_kwargs["storage"] = optuna_storage
    else:
        # Distributed storage (MySQL, PostgreSQL) — wrap in DaskStorage
        try:
            from optuna_integration import DaskStorage

            storage_kwargs["storage"] = DaskStorage(storage=optuna_storage)
        except ImportError:
            logger.warning("optuna-integration not available; using storage directly")
            storage_kwargs["storage"] = optuna_storage

    study = optuna.create_study(
        study_name=optuna_study_name,
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=3),
        sampler=optuna.samplers.TPESampler(constant_liar=True, seed=42),
        load_if_exists=True,  # Resume if study already exists
        **storage_kwargs,
    )

    existing = len(study.trials)
    if existing > 0:
        logger.info("Resuming HPO study '%s' with %d existing trials", optuna_study_name, existing)
    remaining = max(0, hpo_n_trials - existing)
    logger.info("Running %d additional trials (target: %d total)", remaining, hpo_n_trials)

    def objective(trial: optuna.Trial) -> float:
        rank = trial.suggest_int("rank", 10, 200)
        regularization = trial.suggest_float("regularization", 1e-3, 10.0, log=True)
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        n_iter = trial.suggest_int("n_iter", 5, 25)
        return _train_als_subsample(train_pd, val_pd, rank, regularization, alpha, n_iter, trial)

    with get_dask_client(
        mode=get_client_mode_from_config(dask_scheduler_address),
        scheduler_address=dask_scheduler_address,
    ) as client:
        futures = [
            client.submit(study.optimize, objective, n_trials=1, pure=False)
            for _ in range(remaining)
        ]
        client.gather(futures)

    best = study.best_params
    logger.info("HPO complete. Best params: %s (val RMSE=%.4f)", best, study.best_value)

    return {
        "rank": int(best["rank"]),
        "regularization": float(best["regularization"]),
        "alpha": float(best["alpha"]),
        "n_iter": int(best["n_iter"]),
        "best_val_rmse": float(study.best_value),
        "n_trials": len(study.trials),
    }
