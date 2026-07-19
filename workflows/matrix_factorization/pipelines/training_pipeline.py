"""
pipelines/matrix_factorization/training_pipeline.py

ALS end-to-end training pipeline.

Steps:
    load_features_artifact → split_data
  → [hpo_trial_0..N (fan-out, optional)] → collect_best_hpo_params
  → train_als (all epochs, with checkpointing) → visualize_training → compute_metrics → register_model

Fan-out patterns:
  HPO:      hpo_n_trials parallel run_hpo_trial steps → collect_best_hpo_params
  Training: single train_als step trains all n_iter epochs internally
            (sequential, with per-epoch checkpoints for autoresume)

Resumability via explicit checkpoints:
  - training checkpoints: <checkpoint_path>/<run_id>/training
  - hpo checkpoints:      <checkpoint_path>/<run_id>/hpo

Run:
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --config workflows/matrix_factorization/configs/local/training_pipeline.yaml
    python run.py run --workflow matrix_factorization --pipeline training_pipeline --config workflows/matrix_factorization/configs/aws/training_pipeline.yaml --stack aws_stack
"""

from __future__ import annotations

import logging

from zenml import pipeline
from zenml.config import StepRetryConfig
from zenml.enums import ModelStages

from workflows.matrix_factorization.configs import (
    CFG_TRAINING_PIPELINE_NAME,
    CFG_TRAINING_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_TRAINING_PIPELINE_SNAPSHOT_NAME,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.models.base_recommender import Hyperparameters
from workflows.matrix_factorization.steps.evaluation.evaluate import compute_metrics
from workflows.matrix_factorization.steps.evaluation.register import MODEL, register_model
from workflows.matrix_factorization.steps.features.artifacts import (
    load_features_artifact,
)
from workflows.matrix_factorization.steps.features.split import split_data
from workflows.matrix_factorization.steps.hpo.run_hpo import (
    cleanup_hpo_checkpoints,
    collect_best_hpo_params,
    run_hpo_trial,
)
from workflows.matrix_factorization.steps.training.train_als import train_als
from workflows.matrix_factorization.steps.training.visualize import visualize_training

logger = logging.getLogger(__name__)


@pipeline(
    name=CFG_TRAINING_PIPELINE_NAME,
    model=MODEL,  # Configure model for the pipeline context
    retry=StepRetryConfig(max_retries=2, backoff=2, delay=5),  # Exponential backoff: 5s, 10s,
)
def training_pipeline(
    model_stage: str = ModelStages.STAGING,
    # ALS default hyperparams (overridden by HPO if enable_hpo=True)
    rank: int = 50,
    regularization: float = 0.01,
    alpha: float = 1.0,
    n_iter: int = 15,
    # Training execution
    k: int = 10,
    n_workers: int = 4,
    eval_every_n_epochs: int = 1,
    checkpoint_every_n_epochs: int = 1,
    # Model selection — any BaseRecommender subclass, specified as a fully-qualified class path
    recommender_class_name: str = "workflows.matrix_factorization.models.als_implicit_recommender.ALSImplicitRecommender",
    # HPO settings
    enable_hpo: bool = False,
    hpo_n_trials: int = 20,
    hpo_subsample_fraction: float = 0.2,
    optuna_storage: str = "sqlite:///optuna.db",
    optuna_study_name: str = "als_movielens",
    checkpoint_path: str = "./checkpoints",
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
    trigger_serving_on_complete: bool = True,
) -> None:
    """
    Full ALS training pipeline: load features artifact → split → HPO (optional) → train → evaluate → register.

    Training uses ZenML fan-out/fan-in in two places:

    1. HPO fan-out (when enable_hpo=True):
       hpo_n_trials independent run_hpo_trial steps run in parallel,
       each optimizing one Optuna trial. collect_best_hpo_params fans in
       by reading the best result from shared Optuna study storage. Per-trial
       completion markers are checkpointed for autoresume.

    2. Training (single train_als step):
       A single train_als step trains all n_iter epochs with internal
       checkpointing. Supports automatic resume from the latest checkpoint.
       The recommender_class_name parameter controls which model is used.

    Args:
        model_stage: ZenML model stage to register the trained model ("staging" or "production").
        rank: Latent factor dimensionality (overridden by HPO).
        regularization: L2 regularization lambda.
        alpha: Implicit feedback confidence weighting.
        n_iter: Number of ALS epochs.
        k: K for ranking metrics (Precision@K, Recall@K, NDCG@K).
        n_workers: Parallel workers (interpretation is model-specific).
        eval_every_n_epochs: Compute and log val RMSE every N epochs.
        checkpoint_every_n_epochs: Save checkpoint every N epochs.
        recommender_class_name: Fully-qualified class path of a BaseRecommender subclass.
            Any subclass can be used without modifying the pipeline.
        enable_hpo: If True, fan-out hpo_n_trials HPO trials before training.
        hpo_n_trials: Width of the HPO fan-out.
        hpo_subsample_fraction: Data fraction used per HPO trial.
        optuna_storage: Optuna storage URI.
        optuna_study_name: Optuna study name.
        checkpoint_path: Base path for pipeline-run checkpoints.
        seaweedfs_s3_internal_endpoint: SeaweedFS internal S3 endpoint (local stack).
        zenml_local_s3_secret_name: ZenML secret name containing SeaweedFS access_key_id and secret_access_key (local stack).
        trigger_serving_on_complete: If True, trigger serving pipeline after model registration.
    """

    # ── Step 1: Load precomputed features artifact ───────────────────────────
    _, user_encoder, item_encoder, scaled_ratings = load_features_artifact()

    # ── Step 2: Split ──────────────────────────────────────────────────────────
    train_data, val_data, test_data = split_data(
        id="split_data",
        raw_ratings=scaled_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )

    # ── Step 3: HPO (optional fan-out) ─────────────────────────────────────────
    default_hyperparams = Hyperparameters(
        rank=rank,
        regularization=regularization,
        alpha=alpha,
        n_iter=n_iter,
    )

    if enable_hpo:
        # Run HPO trials in parallel (fan-out) and collect best hyperparameters
        after = []
        for i in range(hpo_n_trials):
            trial = run_hpo_trial(
                id=f"hpo_trial_{i}",
                trial_idx=i,
                train_data=train_data,
                val_data=val_data,
                n_workers=n_workers,
                hpo_subsample_fraction=hpo_subsample_fraction,
                optuna_storage=optuna_storage,
                optuna_study_name=optuna_study_name,
                recommender_class_name=recommender_class_name,
                checkpoint_path=checkpoint_path,
                seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
                zenml_local_s3_secret_name=zenml_local_s3_secret_name,
            )
            after.append(trial)

        # Collect best hyperparameters from the completed HPO trials (fan-in)
        best_hyperparams = collect_best_hpo_params(
            optuna_storage=optuna_storage,
            optuna_study_name=optuna_study_name,
            after=after,
        )

        # Cleanup HPO checkpoints after the best hyperparameters have been collected
        cleanup_hpo_checkpoints(
            checkpoint_path=checkpoint_path,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            zenml_local_s3_secret_name=zenml_local_s3_secret_name,
            after=[best_hyperparams],
        )
    else:
        best_hyperparams = default_hyperparams

    # ── Step 4: Train all epochs (single step with internal checkpointing) ────
    user_factors, item_factors, training_states = train_als(
        train_data=train_data,
        val_data=val_data,
        best_hyperparams=best_hyperparams,
        checkpoint_path=checkpoint_path,
        n_workers=n_workers,
        eval_at_k=k,
        eval_every_n_epochs=eval_every_n_epochs,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        recommender_class_name=recommender_class_name,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        zenml_local_s3_secret_name=zenml_local_s3_secret_name,
        id="train_als",
    )

    # ── Step 5: Visualize training metrics ─────────────────────────────────
    visualize_training(
        training_states=training_states,
    )

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    eval_metrics = compute_metrics(
        test_data=test_data,
        user_factors=user_factors,
        item_factors=item_factors,
        top_k=k,
    )

    # ── Step 7: Register ──────────────────────────────────────────────────────
    register_model(
        id="register_model",
        user_factors=user_factors,
        item_factors=item_factors,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        eval_metrics=eval_metrics,
        best_hyperparams=best_hyperparams,
        model_stage=model_stage,
        recommender_class_name=recommender_class_name,
    )


# Create a snapshot of the training pipeline for reproducibility and versioning
training_pipeline.create_snapshot(
    name=CFG_TRAINING_PIPELINE_SNAPSHOT_NAME,
    description=CFG_TRAINING_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "training"],
    replace=True,
)

# TODO: Run training pipeline on schedule (e.g., weekly) to retrain model with new data and update serving endpoint.
