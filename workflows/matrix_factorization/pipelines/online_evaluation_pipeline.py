"""
pipelines/matrix_factorization/online_evaluation_pipeline.py

Online ranking evaluation pipeline.

Evaluates model recommendation quality using Evidently Ranking metrics against
recent inference logs, with the training ratings as ground-truth reference:

  Flow:
    load_raw_ratings_artifact → select_feature_columns  (reference / ground truth)
    ingest_logs               → select_feature_columns  (current  / predictions)
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

from evidently.legacy.metrics.recsys.map_k import MAPKMetric
from evidently.legacy.metrics.recsys.ndcg_k import NDCGKMetric
from evidently.legacy.metrics.recsys.precision_top_k import PrecisionTopKMetric
from evidently.legacy.metrics.recsys.recall_top_k import RecallTopKMetric
from evidently.legacy.metrics.recsys.scores_distribution import ScoreDistribution
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
from workflows.matrix_factorization.steps.data.ingest import ingest_logs
from workflows.matrix_factorization.steps.evaluation.evaluate import evidently_report
from workflows.matrix_factorization.steps.features.artifacts import load_scaled_ratings_artifact
from workflows.matrix_factorization.steps.features.select import select_feature_columns

_RANKING_COLUMNS = [
    CFG_DATASET_FIELD_NAMES.USER_ID.value,
    CFG_DATASET_FIELD_NAMES.ITEM_ID.value,
    CFG_DATASET_FIELD_NAMES.RATING.value,
]


@pipeline(name=CFG_ONLINE_EVALUATION_PIPELINE_NAME)
def online_evaluation_pipeline(
    top_k: int = 10,
) -> None:
    """
    Evaluate online recommendation quality using Evidently Ranking metrics.

    Uses the training ratings as ground-truth reference (actual user-item
    interactions) and recent inference logs as the current dataset (model
    predictions).  Computes Precision, Recall, NDCG, MAP, and score
    distribution at k=10.

    Step-specific parameters (e.g. lookback_days, logs_path) are configured
    in the pipeline run config YAML.
    """
    # --- Reference: ground-truth ratings from training data ---
    raw_ratings = load_scaled_ratings_artifact()
    reference_dataset = select_feature_columns(
        features=raw_ratings,
        columns=_RANKING_COLUMNS,
        force=True,
        id="select_reference_features",
    )

    # --- Current: recent inference logs (model predictions) ---
    inference_logs = ingest_logs(model_name=CFG_MODEL_NAME)
    current_dataset = select_feature_columns(
        features=inference_logs,
        columns=_RANKING_COLUMNS,
        force=True,
        id="select_current_features",
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
            EvidentlyMetricConfig.metric(PrecisionTopKMetric, k=top_k),
            EvidentlyMetricConfig.metric(RecallTopKMetric, k=top_k),
            EvidentlyMetricConfig.metric(NDCGKMetric, k=top_k),
            EvidentlyMetricConfig.metric(MAPKMetric, k=top_k),
            EvidentlyMetricConfig.metric(ScoreDistribution, k=top_k),
        ],
        id="evidently_report",
    )


online_evaluation_pipeline.create_snapshot(
    name=CFG_ONLINE_EVALUATION_PIPELINE_SNAPSHOT_NAME,
    description=CFG_ONLINE_EVALUATION_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "online-evaluation", "ranking"],
    replace=True,
)
