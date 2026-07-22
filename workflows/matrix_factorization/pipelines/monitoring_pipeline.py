"""
pipelines/matrix_factorization/monitoring_pipeline.py

Data Drift & Data Quality monitoring pipeline.

Compares a newly ingested dataset (reference) against the training baseline
(comparison) to detect data drift and quality degradation:

  load_raw_ratings_artifact  → select_feature_columns  (comparison / training baseline)
  ingest_data(lookback_days) → select_feature_columns  (reference  / new data)
  evidently_report_step (DataQualityPreset + DataDriftPreset)
  check_retrain_trigger

NOTE: ingest_data downloads the static MovieLens dataset and simulates recency by
shifting timestamps to the present and filtering to the last ``lookback_days``.
In production this step would fetch recent ratings from a live data source.

For online ranking evaluation (PrecisionTopK, RecallTopK, NDCG, MAP,
ScoreDistribution) see the sibling ``online_evaluation_pipeline``.

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
from workflows.matrix_factorization.steps.data.ingest import ingest_data
from workflows.matrix_factorization.steps.features.artifacts import (
    load_raw_ratings_artifact,
)
from workflows.matrix_factorization.steps.features.select import select_feature_columns

_DRIFT_COLUMNS = [
    CFG_DATASET_FIELD_NAMES.USER_ID.value,
    CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
    CFG_DATASET_FIELD_NAMES.RATING.value,
]


@pipeline(name=CFG_MONITORING_PIPELINE_NAME)
def monitoring_pipeline() -> None:
    """
    Monitor data quality and distribution drift, triggering retraining when needed.

    Compares a freshly ingested dataset (new/recent data) against the training
    baseline (stored artifact) using Evidently DataQualityPreset and
    DataDriftPreset.  Retraining is triggered when EITHER drift OR data quality
    thresholds are exceeded, OR when the model age exceeds ``max_age_days``.

    Step-specific parameters (e.g. lookback_days, dataset_size) are configured
    in the pipeline run config YAML.
    """
    # --- Comparison: training baseline ---
    raw_ratings = load_raw_ratings_artifact()
    comparison_dataset = select_feature_columns(
        features=raw_ratings,
        columns=_DRIFT_COLUMNS,
        force=True,
        id="select_comparison_features",
    )

    # --- Reference: new / recent data ---
    new_ratings = ingest_data()
    reference_dataset = select_feature_columns(
        features=new_ratings,
        columns=_DRIFT_COLUMNS,
        force=True,
        id="select_reference_features",
    )

    # --- Data quality & drift report ---
    report_json, _ = evidently_report_step(
        reference_dataset=reference_dataset,
        comparison_dataset=comparison_dataset,
        column_mapping=EvidentlyColumnMapping(
            id=CFG_DATASET_FIELD_NAMES.USER_ID.value,
            target=CFG_DATASET_FIELD_NAMES.RATING.value,
            prediction=CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            numerical_features=[
                CFG_DATASET_FIELD_NAMES.USER_ID.value,
                CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
            ],
        ),
        metrics=[
            EvidentlyMetricConfig.metric("DataQualityPreset"),
            EvidentlyMetricConfig.metric("DataDriftPreset"),
        ],
        id="evidently_drift",
    )

    # --- Retrain trigger ---
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
