"""
pipelines/matrix_factorization/hpo_pipeline.py

Hyperparameter optimization pipeline (optional, skippable).

Steps:
  run_hpo

Run:
  python run.py --pipeline hpo --config workflows/matrix_factorization/configs/local.yaml
  python run.py --pipeline hpo --config workflows/matrix_factorization/configs/aws.yaml --stack aws_stack

Skip: set enable_hpo: false in config (training_pipeline uses default params).
"""

import dask.dataframe as dd
import pandas as pd
from zenml import pipeline

from workflows.matrix_factorization.steps.hpo.run_hpo import run_hpo


@pipeline(name="matrix_factorization_hpo", enable_cache=False)
def hpo_pipeline(
    train_data: dd.DataFrame,
    val_data: dd.DataFrame,
    hpo_n_trials: int = 20,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
) -> None:
    """
    Distributed hyperparameter optimization for ALS using Optuna.

    Runs N trials, each training ALS on a subsample with HyperbandPruner.
    Persists the Optuna study to SQLite/PostgreSQL — resumable on restart.

    Consumes: train_data, val_data artifacts from data_pipeline.

    Run in isolation or as a prerequisite to training_pipeline.

    Args:
        train_data: Training split from data_pipeline.
        val_data: Validation split from data_pipeline.
        hpo_n_trials: Total number of trials to run.
        hpo_subsample_fraction: Fraction of training data per trial.
        optuna_storage: Optuna storage URI (SQLite or PostgreSQL).
        optuna_study_name: Study name for resumability.
    """
    best_hyperparams = run_hpo(
        train_data=train_data,
        val_data=val_data,
        hpo_n_trials=hpo_n_trials,
        hpo_subsample_fraction=hpo_subsample_fraction,
        optuna_storage=optuna_storage,
        optuna_study_name=optuna_study_name,
    )
