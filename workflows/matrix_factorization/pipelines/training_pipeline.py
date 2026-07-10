"""
pipelines/matrix_factorization/training_pipeline.py

ALS end-to-end training pipeline.

Steps:
  ingest_data → validate_data → build_encoders → split_data
  → [run_hpo (optional)] → train_als → compute_metrics → register_model

Run:
  python run.py --pipeline training --config workflows/matrix_factorization/configs/local.yaml
  python run.py --pipeline training --config workflows/matrix_factorization/configs/aws.yaml --stack aws_stack

Checkpointing: The train_als step checkpoints after every epoch to
checkpoint_path. If the run is interrupted, re-running this command
automatically resumes from the last completed epoch.
"""

from __future__ import annotations

import logging

from zenml import Model, pipeline

from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_data
from workflows.matrix_factorization.steps.data_validation.validate import validate_data
from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders
from workflows.matrix_factorization.steps.feature_engineering.split import split_data
from workflows.matrix_factorization.steps.hpo.run_hpo import run_hpo
from workflows.matrix_factorization.steps.model_evaluation.evaluate import compute_metrics
from workflows.matrix_factorization.steps.model_evaluation.register import register_model
from workflows.matrix_factorization.steps.training.train import train_als

_MODEL = Model(
    name="als_movie_recommender", tags=["matrix_factorization", "als", "movie_recommender"]
)

logger = logging.getLogger(__name__)


@pipeline(name="matrix_factorization_training", enable_cache=True, model=_MODEL)
def training_pipeline(
    n_dask_partitions: int = 8,
    dataset_size: str = "1m",
    min_sparsity: float = 0.95,
    min_ratings: int = 100_000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    # ALS default hyperparams (overridden by HPO if enable_hpo=True)
    rank: int = 50,
    regularization: float = 0.01,
    alpha: float = 1.0,
    n_iter: int = 15,
    # HPO settings
    enable_hpo: bool = False,
    hpo_n_trials: int = 20,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "mysql+pymysql://ops:ops@127.0.0.1:3306/optuna",
    optuna_study_name: str = "als_movielens",
    # Training settings
    checkpoint_path: str = "./checkpoints",
    checkpoint_val_every_n_epochs: int = 5,
    # Registration settings
    rmse_threshold: float = 1.0,
    top_k: int = 10,
) -> None:
    """
    Full ALS training pipeline: data prep → HPO (optional) → train → evaluate → register.

    Key feature — checkpointing and resumability:
      The train_als step writes epoch checkpoints atomically. If the step
      fails mid-training (worker crash, spot instance preemption, OOM), simply
      re-running this pipeline will resume from the last completed epoch.
      ZenML's step-level cache ensures all other steps are also skipped.

    Args:
        n_dask_partitions: Number of Dask partitions for the raw ratings DataFrame.
        dataset_size: "1m" (local dev) or "25m" (AWS).
        min_sparsity: Minimum required sparsity for validation.
        min_ratings: Minimum number of ratings required.
        train_ratio: Training fraction (default 0.8).
        val_ratio: Validation fraction (default 0.1).
        test_ratio: Test fraction (default 0.1).
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
        checkpoint_val_every_n_epochs: Val RMSE logging frequency.
        rmse_threshold: RMSE gate for promoting to 'staging'.
        top_k: K for ranking metrics evaluation.
    """
    # Step 1: Ingest raw ratings data into a Dask DataFrame
    raw_ratings = ingest_data(
        dataset_size=dataset_size,
        n_dask_partitions=n_dask_partitions,
    )

    # Step 2: Validate the raw ratings data
    validation_report = validate_data(
        raw_ratings=raw_ratings,
        min_sparsity=min_sparsity,
        min_ratings=min_ratings,
    )
    logger.info("Data validation report: %s", validation_report)

    # Step 3: Build user and item encoders
    user_encoder, item_encoder = build_encoders(raw_ratings=raw_ratings)

    # Step 4: Split the data into train/val/test sets
    train_data, val_data, test_data = split_data(
        raw_ratings=raw_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Step 5: Run hyperparameter optimization (optional)
    # If enable_hpo=False, the default hyperparams are used for training.

    # --- Default hyperparams (may be overridden by HPO)
    default_hyperparams = {
        "rank": rank,
        "regularization": regularization,
        "alpha": alpha,
        "n_iter": n_iter,
    }
    if enable_hpo:
        # --- Run HPO to find the best hyperparameters
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

    logger.info("Best hyperparameters: %s", best_hyperparams)

    # Step 6: Train the ALS model with checkpointing and resumability
    user_factors, item_factors = train_als(
        train_data=train_data,
        val_data=val_data,
        best_hyperparams=best_hyperparams,
        n_dask_partitions=n_dask_partitions,
        checkpoint_path=checkpoint_path,
        checkpoint_val_every_n_epochs=checkpoint_val_every_n_epochs,
    )

    # Step 7: Evaluate the trained model on the test set
    eval_metrics = compute_metrics(
        test_data=test_data,
        user_factors=user_factors,
        item_factors=item_factors,
        best_hyperparams=best_hyperparams,
        top_k=top_k,
    )

    logger.info("Evaluation metrics: %s", eval_metrics)

    # Step 8: Register the trained model if it meets the RMSE threshold
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
