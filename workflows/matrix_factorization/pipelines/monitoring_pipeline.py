"""
pipelines/matrix_factorization/monitoring_pipeline.py

Drift monitoring and conditional retraining pipeline.

Steps:
  ingest_data → collect_inference_logs → run_drift_detection → check_retrain_trigger → trigger_retraining

Run:
  python run.py --pipeline monitoring --config workflows/matrix_factorization/configs/aws.yaml --stack aws_stack

Scheduled: configure via ZenML schedules or AWS EventBridge (daily recommended).
"""

import dask_expr as dd
from zenml import pipeline

from steps.monitoring.collect_logs import collect_inference_logs
from steps.monitoring.drift_detection import run_drift_detection
from steps.monitoring.retrain import trigger_retraining
from steps.monitoring.trigger import check_retrain_trigger
from workflows.matrix_factorization.configs import CFG_MODEL_NAME
from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_data


@pipeline(name="matrix_factorization_monitoring", enable_cache=False)
def monitoring_pipeline(
    logs_path: str = "s3://aips-recs-zenml-predictions/logs",
    lookback_days: int = 7,
    monitoring_output_path: str = "s3://aips-recs-zenml-predictions/monitoring",
    drift_threshold_n_features: int = 2,
    max_age_days: int = 30,
    retrain_pipeline_module: str = "workflows.matrix_factorization.pipelines.training_pipeline",
    retrain_pipeline_function: str = "training_pipeline",
    retrain_config_path: str = "workflows/matrix_factorization/configs/aws.yaml",
) -> None:
    """
    Monitor model health and trigger retraining when drift is detected.

    Flow:
      1. ingest_data: Build a fresh training reference dataset
      2. collect_inference_logs: Load recent inference logs from S3
      3. run_drift_detection: Compare vs. training reference with Evidently
      4. check_retrain_trigger: Evaluate drift score + model age
      5. trigger_retraining: Fire new training pipeline if triggered

    Args:
        logs_path: S3 prefix for inference log files.
        lookback_days: Number of days of logs to analyze.
        monitoring_output_path: S3 path for Evidently HTML/JSON reports.
        drift_threshold_n_features: Drift trigger threshold.
        max_age_days: Maximum model age before scheduled retrain.
        retrain_pipeline_module: Dotted module path to the training pipeline.
        retrain_pipeline_function: Name of the training pipeline function.
        retrain_config_path: Pipeline config to use for retrain run.
    """
    raw_ratings: dd.DataFrame = ingest_data()

    inference_logs = collect_inference_logs(
        logs_path=logs_path,
        lookback_days=lookback_days,
    )

    # Convert Dask DataFrame to pandas sample for drift detection
    reference_data = raw_ratings[["userId", "rating"]].compute()

    drift_report = run_drift_detection(
        inference_logs=inference_logs,
        reference_data=reference_data,
        reference_id_column="userId",
        current_id_column="user_id",
        numerical_columns=["userId"],
        monitoring_output_path=monitoring_output_path,
    )

    should_retrain = check_retrain_trigger(
        drift_report=drift_report,
        model_name=CFG_MODEL_NAME,
        drift_threshold_n_features=drift_threshold_n_features,
        max_age_days=max_age_days,
    )

    trigger_retraining(
        should_retrain=should_retrain,
        pipeline_module=retrain_pipeline_module,
        pipeline_function=retrain_pipeline_function,
        config_path=retrain_config_path,
    )
