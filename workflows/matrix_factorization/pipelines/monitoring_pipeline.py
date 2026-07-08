"""
pipelines/matrix_factorization/monitoring_pipeline.py

Drift monitoring and conditional retraining pipeline.

Steps:
  collect_inference_logs → run_drift_detection → check_retrain_trigger → trigger_retraining

Run:
  python run.py --pipeline monitoring --config workflows/matrix_factorization/configs/aws.yaml --stack aws_stack

Scheduled: configure via ZenML schedules or AWS EventBridge (daily recommended).
"""

import dask.dataframe as dd
from zenml import pipeline

from workflows.matrix_factorization.steps.monitoring.collect_logs import collect_inference_logs
from workflows.matrix_factorization.steps.monitoring.drift_detection import run_drift_detection
from workflows.matrix_factorization.steps.monitoring.retrain import trigger_retraining
from workflows.matrix_factorization.steps.monitoring.trigger import check_retrain_trigger


@pipeline(name="matrix_factorization_monitoring", enable_cache=False)
def monitoring_pipeline(
    raw_ratings: dd.DataFrame,  # type: ignore[arg-type]
    logs_path: str = "s3://aips-zenml-predictions/logs",
    lookback_days: int = 7,
    monitoring_output_path: str = "s3://aips-zenml-predictions/monitoring",
    drift_threshold_n_features: int = 2,
    max_age_days: int = 30,
    retrain_config_path: str = "workflows/matrix_factorization/configs/aws.yaml",
) -> None:
    """
    Monitor model health and trigger retraining when drift is detected.

    Flow:
      1. collect_inference_logs: Load recent inference logs from S3
      2. run_drift_detection: Compare vs. training reference with Evidently
      3. check_retrain_trigger: Evaluate drift score + model age
      4. trigger_retraining: Fire new training pipeline if triggered

    Args:
        raw_ratings: Training reference dataset (from data_pipeline artifacts).
        logs_path: S3 prefix for inference log files.
        lookback_days: Number of days of logs to analyze.
        monitoring_output_path: S3 path for Evidently HTML/JSON reports.
        drift_threshold_n_features: Drift trigger threshold.
        max_age_days: Maximum model age before scheduled retrain.
        retrain_config_path: Pipeline config to use for retrain run.
    """
    inference_logs = collect_inference_logs(
        logs_path=logs_path,
        lookback_days=lookback_days,
    )

    drift_report = run_drift_detection(
        inference_logs=inference_logs,
        raw_ratings=raw_ratings,
        monitoring_output_path=monitoring_output_path,
    )

    should_retrain = check_retrain_trigger(
        drift_report=drift_report,
        drift_threshold_n_features=drift_threshold_n_features,
        max_age_days=max_age_days,
    )

    trigger_retraining(
        should_retrain=should_retrain,
        config_path=retrain_config_path,
    )
