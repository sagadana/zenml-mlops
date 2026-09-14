"""
pipelines/matrix_factorization/online_evaluation_pipeline.py

Online ranking evaluation pipeline.

Evaluates model recommendation quality using Evidently Ranking metrics against
recent inference logs, with the training ratings as ground-truth reference:

  Flow:
    load_train_dataset_artifact → select_feature_columns  (reference / ground truth)
    ingest_prediction_logs               → select_feature_columns  (current  / predictions)
    evidently_report (id="evidently_ranking") with RankingPreset metrics

Ranking metrics (k=10):
  PrecisionTopK, RecallTopK, NDCG, MAP, ScoreDistribution

Column mapping:
  - user_id    → userId
  - item_id    → movieId
  - predictions → rating  (predicted score in the current / inference-log dataset)
  - target      → rating  (actual rating in the reference / raw-ratings dataset)
  - recommendations_type → "score"

Run:
    python run.py run --workflow matrix_factorization --pipeline online_evaluation_pipeline --config workflows/matrix_factorization/configs/local/online_evaluation_pipeline.yaml
    python run.py run --workflow matrix_factorization --pipeline online_evaluation_pipeline --config workflows/matrix_factorization/configs/aws/online_evaluation_pipeline.yaml --stack aws_stack

Scheduled: configure via ZenML schedules or AWS EventBridge (daily recommended).
"""

from zenml import pipeline
from zenml.integrations.evidently.column_mapping import EvidentlyColumnMapping
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig

from workflows.matrix_factorization.configs import (
    CFG_DATASET_FIELD_NAMES,
    CFG_MODEL_NAME,
    CFG_ONLINE_EVALUATION_PIPELINE_NAME,
    CFG_ONLINE_EVALUATION_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_ONLINE_EVALUATION_PIPELINE_SNAPSHOT_NAME,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.steps.data.ingest import (
    ingest_batch_predictions,
)
from workflows.matrix_factorization.steps.data.preprocess import (
    preprocess_evaluation_datasets,
)
from workflows.matrix_factorization.steps.evaluation.evaluate import evidently_report
from workflows.matrix_factorization.steps.features.artifacts import (
    load_train_dataset_artifact,
)
from workflows.matrix_factorization.steps.features.select import select_feature_columns

_RANKING_COLUMNS = [
    CFG_DATASET_FIELD_NAMES.USER_ID.value,
    CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
    CFG_DATASET_FIELD_NAMES.RATING.value,
]


@pipeline(name=CFG_ONLINE_EVALUATION_PIPELINE_NAME)
def online_evaluation_pipeline(
    top_k: int = 10,
    max_users: int | None = 1_000,
    max_user_items: int | None = 10,
) -> None:
    """
    Evaluate online recommendation quality using Evidently Ranking metrics.

    Uses the training ratings as ground-truth reference (actual user-item
    interactions) and recent inference logs as the current dataset (model
    predictions).  Computes Precision, Recall, NDCG, MAP, and score
    distribution at k=20.

    Step-specific parameters (e.g. lookback_days, logs_path) are configured
    in the pipeline run config YAML.
    """
    # --- Reference: ground-truth ratings from training data ---
    train_dataset = load_train_dataset_artifact()
    reference_dataset = select_feature_columns(
        features=train_dataset,
        columns=_RANKING_COLUMNS,
        force=True,
        id="select_reference_features",
    )

    # --- Current: recent inference logs (model predictions) ---
    # TODO: Use this for real-time logs instead of batch recommendations
    # inference_logs = ingest_prediction_logs(model_name=CFG_MODEL_NAME)
    inference_logs = ingest_batch_predictions(
        model_name=CFG_MODEL_NAME,
        max_users=max_users,
        max_user_items=max_user_items,
        limit=(max_users * max_user_items)
        if max_users is not None and max_user_items is not None
        else None,
    )
    current_dataset = select_feature_columns(
        features=inference_logs,
        columns=_RANKING_COLUMNS,
        force=True,
        id="select_current_features",
    )

    reference_dataset, current_dataset = preprocess_evaluation_datasets(
        reference_dataset=reference_dataset,
        current_dataset=current_dataset,
        max_user_items=max_user_items,
    )

    # --- Ranking evaluation report ---
    evidently_report(
        reference_dataset=reference_dataset,
        comparison_dataset=current_dataset,
        column_mapping=EvidentlyColumnMapping(
            target=CFG_DATASET_FIELD_NAMES.RATING.value,
            prediction=CFG_DATASET_FIELD_NAMES.RATING.value,
        ),
        user_id_column=CFG_DATASET_FIELD_NAMES.USER_ID.value,
        item_id_column=CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
        metrics=[
            EvidentlyMetricConfig.metric("RecsysPreset", k=top_k),
        ],
        id="evidently_report",
    )


online_evaluation_pipeline.create_snapshot(
    name=CFG_ONLINE_EVALUATION_PIPELINE_SNAPSHOT_NAME,
    description=CFG_ONLINE_EVALUATION_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "online-evaluation", "ranking"],
    replace=True,
)
