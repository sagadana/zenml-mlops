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
from workflows.matrix_factorization.steps.data.ingest import ingest_logs
from workflows.matrix_factorization.steps.features.artifacts import (
    load_scaled_ratings_artifact,
)
from workflows.matrix_factorization.steps.features.select import select_feature_columns


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

    scaled_ratings = load_scaled_ratings_artifact()
    reference_dataset = select_feature_columns(
        features=scaled_ratings,
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

    report_json, _ = evidently_report_step(
        reference_dataset=reference_dataset,
        comparison_dataset=current_dataset,
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
    )

    _ = check_retrain_trigger(
        report_json=report_json,
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
