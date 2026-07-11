"""
pipelines/matrix_factorization/training_pipeline.py

ALS end-to-end training pipeline.

Steps:
  ingest_data → validate_data → build_encoders → split_data
  → [hpo_trial_0..N (fan-out, optional)] → collect_best_hpo_params
  → init_als_factors → als_epoch_0 → als_epoch_1 → ... → als_epoch_{n_iter-1}
  → compute_metrics → (register_model || mlflow_register_model_step)

Fan-out patterns:
  HPO:      hpo_n_trials parallel run_hpo_trial steps → collect_best_hpo_params
  Training: n_iter chained train_als_epoch steps (sequential, not parallel —
            each epoch depends on the previous epoch's factors)

Resumability via ZenML cache:
  Each train_als_epoch step is cached on its inputs (epoch index, factor matrices,
  train data, hyperparams). On pipeline restart, completed epochs are skipped
  automatically — no manual checkpointing needed.

Run:
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --config workflows/matrix_factorization/configs/local/training_pipeline.yaml
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --config workflows/matrix_factorization/configs/aws/training_pipeline.yaml --stack aws_stack
"""

from __future__ import annotations

import logging

from zenml import Model, pipeline
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)

from workflows.matrix_factorization.configs import (
    CFG_MODEL_NAME,
)
from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_data
from workflows.matrix_factorization.steps.data_validation.validate import validate_data
from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders
from workflows.matrix_factorization.steps.feature_engineering.split import split_data
from workflows.matrix_factorization.steps.hpo.run_hpo import collect_best_hpo_params, run_hpo_trial
from workflows.matrix_factorization.steps.model_evaluation.evaluate import compute_metrics
from workflows.matrix_factorization.steps.model_evaluation.register import register_model
from workflows.matrix_factorization.steps.training.als_epoch import (
    init_als_factors,
    train_als_epoch,
)

_MODEL = Model(name=CFG_MODEL_NAME, tags=["matrix_factorization", "als", "movie_recommender"])

logger = logging.getLogger(__name__)


@pipeline(name="matrix_factorization_training", enable_cache=True, model=_MODEL)
def training_pipeline(
    model_stage: str = "staging",
    # ALS default hyperparams (overridden by HPO if enable_hpo=True)
    rank: int = 50,
    regularization: float = 0.01,
    alpha: float = 1.0,
    n_iter: int = 15,
    # Training execution
    n_workers: int = 4,
    checkpoint_val_every_n_epochs: int = 1,
    # HPO settings
    enable_hpo: bool = False,
    hpo_n_trials: int = 20,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
) -> None:
    """
    Full ALS training pipeline: data prep → HPO (optional) → train → evaluate → register.

    Training uses ZenML fan-out/fan-in in two places:

    1. HPO fan-out (when enable_hpo=True):
       hpo_n_trials independent run_hpo_trial steps run in parallel,
       each optimizing one Optuna trial. collect_best_hpo_params fans in
       by reading the best result from shared Optuna study storage.

    2. Training chain (always):
       n_iter train_als_epoch steps are chained sequentially. Each step
       receives the previous epoch's factor matrices as inputs and produces
       updated factors. ZenML caching means completed epochs are skipped on
       pipeline restart — resumability with no manual checkpoint management.

    Args:
        model_stage: ZenML model stage to register the trained model ("staging" or "production").
        rank: Latent factor dimensionality (overridden by HPO).
        regularization: L2 regularization lambda.
        alpha: Implicit feedback confidence weighting.
        n_iter: Number of ALS epochs (chain length).
        n_workers: Partition workers per epoch (ProcessPoolExecutor).
        checkpoint_val_every_n_epochs: Log val RMSE every N epochs.
        enable_hpo: If True, fan-out hpo_n_trials HPO trials before training.
        hpo_n_trials: Width of the HPO fan-out.
        hpo_subsample_fraction: Data fraction used per HPO trial.
        optuna_storage: Optuna storage URI.
        optuna_study_name: Optuna study name.
    """
    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    raw_ratings = ingest_data()

    # ── Step 2: Validate ───────────────────────────────────────────────────────
    validate_data(raw_ratings=raw_ratings)

    # ── Step 3: Build encoders ─────────────────────────────────────────────────
    user_encoder, item_encoder = build_encoders(raw_ratings=raw_ratings)

    # ── Step 4: Split ──────────────────────────────────────────────────────────
    train_data, val_data, test_data = split_data(
        raw_ratings=raw_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )

    # ── Step 5: HPO (optional fan-out) ─────────────────────────────────────────
    default_hyperparams = {
        "rank": rank,
        "regularization": regularization,
        "alpha": alpha,
        "n_iter": n_iter,
    }
    if enable_hpo:
        after = []
        for i in range(hpo_n_trials):
            trial = run_hpo_trial(
                trial_idx=i,
                train_data=train_data,
                val_data=val_data,
                n_workers=n_workers,
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

    # ── Step 6: Initialize factor matrices ────────────────────────────────────
    user_factors, item_factors = init_als_factors(
        train_data=train_data,
        best_hyperparams=best_hyperparams,
    )

    # ── Step 7: Chain n_iter epoch steps ──────────────────────────────────────
    # Each step depends on the previous epoch's output — sequential chain, not parallel.
    # ZenML cache provides epoch-level resume: if epoch N's inputs are unchanged,
    # the step is skipped and its cached output is used.
    for epoch in range(n_iter):
        user_factors, item_factors = train_als_epoch(
            epoch=epoch,
            user_factors=user_factors,
            item_factors=item_factors,
            train_data=train_data,
            val_data=val_data,
            best_hyperparams=best_hyperparams,
            n_workers=n_workers,
            checkpoint_val_every_n_epochs=checkpoint_val_every_n_epochs,
            id=f"als_epoch_{epoch}",
        )

    # ── Step 8: Evaluate ──────────────────────────────────────────────────────
    eval_metrics = compute_metrics(
        test_data=test_data,
        user_factors=user_factors,
        item_factors=item_factors,
        best_hyperparams=best_hyperparams,
    )

    # ── Step 9: Register (parallel fan-out) ──────────────────────────────────
    model = register_model(
        user_factors=user_factors,
        item_factors=item_factors,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        eval_metrics=eval_metrics,
        best_hyperparams=best_hyperparams,
        model_stage=model_stage,
    )

    # Register model with MLflow Model Registry if available
    try:
        mlflow_register_model_step(
            model=model,
            name=CFG_MODEL_NAME,
            metadata={
                "n_users": model.n_users,
                "n_items": model.n_items,
                **eval_metrics,
                **best_hyperparams,
            },
        )
    except Exception as exc:
        logger.warning("MLflow model registry registration skipped: %s", exc)


# TODO: Trigger serving pipeline automatically after training_pipeline completes successfully.
# TODO: Run training pipeline on schedule (e.g., weekly) to retrain model with new data and update serving endpoint.
