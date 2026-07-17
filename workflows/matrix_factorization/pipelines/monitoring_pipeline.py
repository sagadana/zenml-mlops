"""
pipelines/matrix_factorization/monitoring_pipeline.py

Drift monitoring and conditional retraining pipeline.

Steps:
    load_raw_ratings_artifact → ingest_logs → run_drift_detection → check_retrain_trigger → trigger_retraining

Run:
    python run.py run --workflow matrix_factorization --pipeline monitoring_pipeline --config workflows/matrix_factorization/configs/aws/monitoring_pipeline.yaml --stack aws_stack

Scheduled: configure via ZenML schedules or AWS EventBridge (daily recommended).
"""

from zenml import pipeline

from steps.monitoring.drift_detection import run_drift_detection
from steps.monitoring.retrain import check_retrain_trigger
from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_MODEL_NAME,
    CFG_MONITORING_PIPELINE_NAME,
    CFG_MONITORING_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_MONITORING_PIPELINE_SNAPSHOT_NAME,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_logs
from workflows.matrix_factorization.steps.feature_engineering.artifacts import (
    load_raw_ratings_artifact,
)
from workflows.matrix_factorization.steps.feature_engineering.select import select_feature_columns


@pipeline(name=CFG_MONITORING_PIPELINE_NAME)
def monitoring_pipeline() -> None:
    """
    Monitor model health and trigger retraining when drift is detected.

    Flow:
        1. load_raw_ratings_artifact: Load persisted training reference dataset
        2. ingest_logs: Load recent inference logs from S3
        3. run_drift_detection: Compare vs. training reference with Evidently
        4. check_retrain_trigger: Evaluate drift score + model age
        5. trigger_retraining: Fire new training pipeline if triggered

    Step-specific parameters are configured in step blocks of the
    pipeline run config YAML.
    """

    raw_ratings = load_raw_ratings_artifact()
    reference_dataset = select_feature_columns(
        features=raw_ratings,
        columns=[
            CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_DATASET_FIELD_NAMES.RATING.value,
        ],
    )

    inference_logs = ingest_logs()
    current_dataset = select_feature_columns(
        features=inference_logs,
        columns=[
            CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_DATASET_FIELD_NAMES.RATING.value,
        ],
    )

    drift_report = run_drift_detection(
        reference_dataset=reference_dataset,
        current_dataset=current_dataset,
    )

    _ = check_retrain_trigger(
        drift_report=drift_report,
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
