"""
pipelines/matrix_factorization/monitoring_pipeline.py

Drift monitoring and conditional retraining pipeline.

Steps:
  ingest_data → collect_inference_logs → run_drift_detection → check_retrain_trigger → trigger_retraining

Run:
    python run.py run --workflow matrix_factorization --pipeline monitoring_pipeline --config workflows/matrix_factorization/configs/aws/monitoring_pipeline.yaml --stack aws_stack

Scheduled: configure via ZenML schedules or AWS EventBridge (daily recommended).
"""

from zenml import pipeline

from steps.monitoring.collect_logs import collect_inference_logs
from steps.monitoring.drift_detection import run_drift_detection
from steps.monitoring.retrain import trigger_retraining
from steps.monitoring.trigger import check_retrain_trigger
from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_MODEL_NAME,
    CFG_MONITORING_PIPELINE_NAME,
    CFG_MONITORING_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_MONITORING_PIPELINE_SNAPSHOT_NAME,
)
from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_data
from workflows.matrix_factorization.steps.feature_engineering.select import select_feature_columns


@pipeline(name=CFG_MONITORING_PIPELINE_NAME, enable_cache=False)
def monitoring_pipeline() -> None:
    """
    Monitor model health and trigger retraining when drift is detected.

    Flow:
      1. ingest_data: Build a fresh training reference dataset
      2. collect_inference_logs: Load recent inference logs from S3
      3. run_drift_detection: Compare vs. training reference with Evidently
      4. check_retrain_trigger: Evaluate drift score + model age
      5. trigger_retraining: Fire new training pipeline if triggered

    Step-specific parameters are configured in step blocks of the
    pipeline run config YAML.
    """

    raw_ratings = ingest_data()

    inference_logs = collect_inference_logs()

    baseline_dataset = select_feature_columns(
        features=raw_ratings,
        columns=[
            CFG_DATASET_FIELD_NAMES.USER_ID.value,
            CFG_DATASET_FIELD_NAMES.RATING.value,
        ],
    )

    drift_report = run_drift_detection(
        baseline_dataset=baseline_dataset,
        inference_logs=inference_logs,
    )

    should_retrain = check_retrain_trigger(
        drift_report=drift_report,
        model_name=CFG_MODEL_NAME,
    )

    trigger_retraining(should_retrain=should_retrain)


# Create a snapshot of the serving pipeline for reproducibility and versioning
monitoring_pipeline.create_snapshot(
    name=CFG_MONITORING_PIPELINE_SNAPSHOT_NAME,
    description=CFG_MONITORING_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=["matrix_factorization", "als", "monitoring"],
    replace=True,
)
