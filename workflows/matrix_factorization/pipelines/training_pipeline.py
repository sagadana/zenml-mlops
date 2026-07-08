"""
pipelines/matrix_factorization/training_pipeline.py

ALS model training, evaluation, and registration pipeline.

Steps:
  [run_hpo (optional)] → train_als → compute_metrics → register_model

Run:
  python run.py --pipeline training --config workflows/matrix_factorization/configs/local.yaml
  python run.py --pipeline training --config workflows/matrix_factorization/configs/aws.yaml --stack aws_stack

Checkpointing: The train_als step checkpoints after every epoch to
checkpoint_path. If the run is interrupted, re-running this command
automatically resumes from the last completed epoch.
"""

from __future__ import annotations

import dask_expr as dd
import pandas as pd
from zenml import Model, pipeline

from workflows.matrix_factorization.steps.hpo.run_hpo import run_hpo
from workflows.matrix_factorization.steps.model_evaluation.evaluate import compute_metrics
from workflows.matrix_factorization.steps.model_evaluation.register import register_model
from workflows.matrix_factorization.steps.training.train import train_als

_MODEL = Model(name="als_movie_recommender")


@pipeline(name="matrix_factorization_training", enable_cache=True, model=_MODEL)
def training_pipeline(
    train_data: dd.DataFrame,
    val_data: dd.DataFrame,
    test_data: dd.DataFrame,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    # ALS default hyperparams (overridden by HPO if enable_hpo=True)
    rank: int = 50,
    regularization: float = 0.01,
    alpha: float = 1.0,
    n_iter: int = 15,
    # HPO settings
    enable_hpo: bool = False,
    hpo_n_trials: int = 20,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
    # Training settings
    checkpoint_path: str = "./checkpoints",
    n_dask_partitions: int = 4,
    checkpoint_val_every_n_epochs: int = 5,
    # Registration settings
    rmse_threshold: float = 1.0,
    top_k: int = 10,
) -> None:
    """
    Full ALS training pipeline: HPO (optional) → train → evaluate → register.

    Artifacts consumed from data_pipeline:
      train_data, val_data, test_data, user_encoder, item_encoder.

    Key feature — checkpointing and resumability:
      The train_als step writes epoch checkpoints atomically. If the step
      fails mid-training (worker crash, spot instance preemption, OOM), simply
      re-running this pipeline will resume from the last completed epoch.
      ZenML's step-level cache ensures all other steps are also skipped.

    Args:
        train_data: Training split.
        val_data: Validation split.
        test_data: Test split.
        user_encoder: User ID encoder from data_pipeline.
        item_encoder: Item ID encoder from data_pipeline.
        rank: Latent factor dimensionality (overridden by HPO).
        regularization: L2 regularization lambda.
        alpha: Implicit feedback confidence weighting.
        n_iter: Number of ALS iterations (epochs).
        enable_hpo: If True, run Optuna HPO before training.
        hpo_n_trials: Number of Optuna trials.
        hpo_subsample_fraction: Fraction of data for HPO trials.
        optuna_storage: Optuna storage URI.
        optuna_study_name: Study name for resumable HPO.
        checkpoint_path: Directory for epoch checkpoints.
        n_dask_partitions: Dask parallelism for ALS updates.
        checkpoint_val_every_n_epochs: Val RMSE logging frequency.
        rmse_threshold: RMSE gate for promoting to 'staging'.
        top_k: K for ranking metrics evaluation.
    """
    # Default hyperparams (may be overridden by HPO)
    default_hyperparams = {
        "rank": rank,
        "regularization": regularization,
        "alpha": alpha,
        "n_iter": n_iter,
    }

    if enable_hpo:
        best_hyperparams = run_hpo(
            train_data=train_data,
            val_data=val_data,
            hpo_n_trials=hpo_n_trials,
            hpo_subsample_fraction=hpo_subsample_fraction,
            optuna_storage=optuna_storage,
            optuna_study_name=optuna_study_name,
        )
    else:
        best_hyperparams = default_hyperparams

    user_factors, item_factors = train_als(
        train_data=train_data,
        val_data=val_data,
        best_hyperparams=best_hyperparams,
        checkpoint_path=checkpoint_path,
        n_dask_partitions=n_dask_partitions,
        checkpoint_val_every_n_epochs=checkpoint_val_every_n_epochs,
    )

    eval_metrics = compute_metrics(
        test_data=test_data,
        user_factors=user_factors,
        item_factors=item_factors,
        best_hyperparams=best_hyperparams,
        top_k=top_k,
    )

    register_model(
        user_factors=user_factors,
        item_factors=item_factors,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        eval_metrics=eval_metrics,
        best_hyperparams=best_hyperparams,
        rmse_threshold=rmse_threshold,
        checkpoint_path=checkpoint_path,
    )
