"""
pipelines/matrix_factorization/training_pipeline.py

ALS end-to-end training pipeline.

Steps:
  ingest_data → validate_data → build_encoders → split_data
  → [run_hpo (optional)] → train_als → compute_metrics → register_model

Run:
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --config workflows/matrix_factorization/configs/local/training_pipeline.yaml
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --config workflows/matrix_factorization/configs/aws/training_pipeline.yaml --stack aws_stack

Checkpointing: The train_als step checkpoints after every epoch to
checkpoint_path. If the run is interrupted, re-running this command
automatically resumes from the last completed epoch.
"""

from __future__ import annotations

import logging

from zenml import Model, pipeline

from workflows.matrix_factorization.configs import CFG_MODEL_NAME
from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_data
from workflows.matrix_factorization.steps.data_validation.validate import validate_data
from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders
from workflows.matrix_factorization.steps.feature_engineering.split import split_data
from workflows.matrix_factorization.steps.hpo.run_hpo import collect_best_hpo_params, run_hpo_trial
from workflows.matrix_factorization.steps.model_evaluation.evaluate import compute_metrics
from workflows.matrix_factorization.steps.model_evaluation.register import register_model
from workflows.matrix_factorization.steps.training.train import train_als

_MODEL = Model(name=CFG_MODEL_NAME, tags=["matrix_factorization", "als", "movie_recommender"])

logger = logging.getLogger(__name__)


@pipeline(name="matrix_factorization_training", enable_cache=True, model=_MODEL)
def training_pipeline(
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
) -> None:
    """
    Full ALS training pipeline: data prep → HPO (optional) → train → evaluate → register.

    Key feature — checkpointing and resumability:
      The train_als step writes epoch checkpoints atomically. If the step
      fails mid-training (worker crash, spot instance preemption, OOM), simply
      re-running this pipeline will resume from the last completed epoch.
      ZenML's step-level cache ensures all other steps are also skipped.

    Args:
        rank: Latent factor dimensionality (overridden by HPO).
        regularization: L2 regularization lambda.
        alpha: Implicit feedback confidence weighting.
        n_iter: Number of ALS iterations (epochs).
        enable_hpo: If True, run Optuna HPO before training (fan-out, one ZenML step per trial).
        hpo_n_trials: Total number of Optuna trials (fan-out width).
        hpo_subsample_fraction: Fraction of training data to use per HPO trial.
        optuna_storage: Optuna storage URI for cross-trial study persistence.
        optuna_study_name: Optuna study name.
        Other step-specific parameters are configured in step blocks of the
        pipeline run config YAML.
    """
    # Step 1: Ingest raw ratings data into a pandas DataFrame
    raw_ratings = ingest_data()

    # Step 2: Validate the raw ratings data
    validation_report = validate_data(raw_ratings=raw_ratings)
    logger.info("Data validation report: %s", validation_report)

    # Step 3: Build user and item encoders
    user_encoder, item_encoder = build_encoders(raw_ratings=raw_ratings)

    # Step 4: Split the data into train/val/test sets
    train_data, val_data, test_data = split_data(
        raw_ratings=raw_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
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
        # --- Fan-out: one ZenML step per HPO trial, run in parallel
        after = []
        for i in range(hpo_n_trials):
            trial = run_hpo_trial(
                trial_idx=i,
                train_data=train_data,
                val_data=val_data,
                hpo_subsample_fraction=hpo_subsample_fraction,
                optuna_storage=optuna_storage,
                optuna_study_name=optuna_study_name,
                id=f"hpo_trial_{i}",
            )
            after.append(trial)
        best_hyperparams = collect_best_hpo_params(
            optuna_storage=optuna_storage,
            optuna_study_name=optuna_study_name,
            after=after,
        )
    else:
        best_hyperparams = default_hyperparams

    logger.info("Best hyperparameters: %s", best_hyperparams)

    # Step 6: Train the ALS model with checkpointing and resumability
    user_factors, item_factors = train_als(
        train_data=train_data,
        val_data=val_data,
        best_hyperparams=best_hyperparams,
    )

    # Step 7: Evaluate the trained model on the test set
    eval_metrics = compute_metrics(
        test_data=test_data,
        user_factors=user_factors,
        item_factors=item_factors,
        best_hyperparams=best_hyperparams,
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
    )
