"""
pipelines/matrix_factorization/monitoring_pipeline.py

Drift monitoring and conditional retraining pipeline.

Two parallel monitoring flows feed a single retrain trigger:

  Flow 1 — Inference logs (real-time serving):
    load_scaled_ratings_artifact → select_feature_columns
    ingest_logs → select_feature_columns
    evidently_report_step (id="evidently_logs")

  Flow 2 — Batch recommendations (S3 Parquet shards):
    load_scaled_ratings_artifact (shared reference)
    ingest_batch_recommendations → select_feature_columns
    evidently_report_step (id="evidently_batch")

  Fan-in:
    check_retrain_trigger(report_json=<logs>, report_json_batch=<batch>)

Run:
    python run.py run --workflow matrix_factorization --pipeline monitoring_pipeline --config workflows/matrix_factorization/configs/aws/monitoring_pipeline.yaml --stack aws_stack

Scheduled: configure via ZenML schedules or AWS EventBridge (daily recommended).
"""

from zenml import pipeline
from zenml.integrations.evidently.column_mapping import (
    EvidentlyColumnMapping,
)
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps.evidently_report import (
    evidently_report_step,
)

from steps.monitoring.retrain import check_retrain_trigger
from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_MODEL_NAME,
    CFG_MONITORING_PIPELINE_NAME,
    CFG_MONITORING_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_MONITORING_PIPELINE_SNAPSHOT_NAME,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.steps.data.ingest import (
    ingest_batch_recommendations,
    ingest_logs,
)
from workflows.matrix_factorization.steps.features.artifacts import (
    load_scaled_ratings_artifact,
)
from workflows.matrix_factorization.steps.features.select import select_feature_columns


@pipeline(name=CFG_MONITORING_PIPELINE_NAME)
def monitoring_pipeline() -> None:
    """
    Monitor model health and trigger retraining when drift is detected.

    # NOTE: Drift check should be done on real item events (clicks, watches, ratings, purchases, etc.) that would be later used to retrain the model, not on inference logs.
    # Inference logs are only used here as a proxy for recent events, since they are the only data available in this demo workflow.

    Two parallel Evidently drift checks are run against the same training
    reference dataset:

    Flow 1 — Inference logs:
        Compares recent real-time serving logs against the training reference.
    Flow 2 — Batch recommendations:
        Compares recent S3 Parquet batch-inference outputs against the same reference.

    Retraining is triggered when EITHER source shows drift, OR when the model
    age exceeds ``max_age_days``.

    Step-specific parameters are configured in step blocks of the pipeline run
    config YAML.
    """
    # --- Shared reference dataset (training data) ---
    scaled_ratings = load_scaled_ratings_artifact()
    reference_dataset = select_feature_columns(
        features=scaled_ratings,
        columns=[
            CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            CFG_DATASET_FIELD_NAMES.RATING.value,
        ],
        force=True,
        id="select_reference_features",
    )

    # --- Flow 1: Inference logs ---
    inference_logs = ingest_logs(model_name=CFG_MODEL_NAME)
    current_dataset_logs = select_feature_columns(
        features=inference_logs,
        columns=[
            CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            CFG_DATASET_FIELD_NAMES.RATING.value,
        ],
        force=True,
        id="select_logs_features",
    )

    report_json_logs, _ = evidently_report_step(
        reference_dataset=reference_dataset,
        comparison_dataset=current_dataset_logs,
        column_mapping=EvidentlyColumnMapping(
            target=CFG_DATASET_FIELD_NAMES.RATING.value,
            numerical_features=[
                CFG_DATASET_FIELD_NAMES.USER_ID.value,
                CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            ],
        ),
        metrics=[
            EvidentlyMetricConfig.metric("DataQualityPreset"),
            EvidentlyMetricConfig.metric("DataDriftPreset"),
        ],
        id="evidently_logs",
    )

    # --- Flow 2: Batch recommendations ---
    batch_recs = ingest_batch_recommendations(model_name=CFG_MODEL_NAME)
    current_dataset_batch = select_feature_columns(
        features=batch_recs,
        columns=[
            CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            CFG_DATASET_FIELD_NAMES.RATING.value,
        ],
        force=True,
        id="select_batch_features",
    )

    report_json_batch, _ = evidently_report_step(
        reference_dataset=reference_dataset,
        comparison_dataset=current_dataset_batch,
        column_mapping=EvidentlyColumnMapping(
            target=CFG_DATASET_FIELD_NAMES.RATING.value,
            numerical_features=[
                CFG_DATASET_FIELD_NAMES.USER_ID.value,
                CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            ],
        ),
        metrics=[
            EvidentlyMetricConfig.metric("DataQualityPreset"),
            EvidentlyMetricConfig.metric("DataDriftPreset"),
        ],
        id="evidently_batch",
    )

    # --- Fan-in: evaluate both reports ---
    _ = check_retrain_trigger(
        report_json=report_json_logs,
        report_json_batch=report_json_batch,
        model_name=CFG_MODEL_NAME,
    )

    # TODO: Use another solution to trigger the retraining pipeline
    # This is only available for Pro and Enterprise users with ZenML Cloud.
    # trigger_retraining(should_retrain=should_retrain)


# Create a snapshot of the serving pipeline for reproducibility and versioning
monitoring_pipeline.create_snapshot(
    name=CFG_MONITORING_PIPELINE_SNAPSHOT_NAME,
    description=CFG_MONITORING_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "monitoring"],
    replace=True,
)
