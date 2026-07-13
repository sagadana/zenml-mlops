"""
pipelines/matrix_factorization/serving_pipeline.py

Model serving pipeline: batch pre-computation + real-time endpoint deployment.

Fan-out/fan-in pattern for batch serving:
  load_als_model
      ↓
  batch_0  batch_1  ...  batch_{n_batches-1}   ← fan-out (parallel predict_user_batch)
      ↓         ↓              ↓
  collect_batch_recommendations               ← fan-in (writes to S3 / DynamoDB)

Real-time flow (independent):
  build_serving_image → deploy_endpoint

Run:
  python run.py run --workflow matrix_factorization --pipeline serving_pipeline --config workflows/matrix_factorization/configs/local/serving_pipeline.yaml
  python run.py run --workflow matrix_factorization --pipeline serving_pipeline --config workflows/matrix_factorization/configs/aws/serving_pipeline.yaml --stack aws_stack
"""

import logging

from zenml import pipeline

from steps.serving.build_image import build_serving_image
from steps.serving.deploy import deploy_endpoint
from workflows.matrix_factorization.configs import (
    CFG_SERVING_PIPELINE_NAME,
    CFG_SERVING_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_SERVING_PIPELINE_SNAPSHOT_NAME,
)
from workflows.matrix_factorization.steps.serving.batch_predict import (
    collect_batch_recommendations,
    load_als_model,
)
from workflows.matrix_factorization.steps.serving.batch_predict_user import predict_user_batch

logger = logging.getLogger(__name__)


@pipeline(name=CFG_SERVING_PIPELINE_NAME, enable_cache=False)
def serving_pipeline(
    n_batches: int = 1,
    batch_top_k: int = 50,
    user_batch_size: int = 10_000,
    model_stage: str = "staging",
    batch_output_path: str = "./predictions/batch",
    dynamodb_table: str | None = None,
    dynamodb_partition_key: str = "id",
) -> None:
    """
    Deploy the trained model for both batch and real-time serving.

    Batch fan-out/fan-in:
      1. load_als_model: load production model from ZenML MCP.
      2. Fan-out: n_batches parallel predict_user_batch steps (id="batch_0", "batch_1", ...).
         Each step processes user_batch_size users determined by its batch_idx.
      3. Fan-in: collect_batch_recommendations reads all batch_* outputs via the
         ZenML Client API, writes Parquet shards to batch_output_path, and optionally
         loads to DynamoDB.

    Real-time flow (independent of batch):
      build_serving_image → deploy_endpoint

    Args:
        n_batches: Number of parallel batch steps (fan-out width).
            Set based on dataset size: ceil(n_users / user_batch_size).
            Local 1M: 1, AWS 25M: ~17 (162K users / 10K batch).
        batch_top_k: Recommendations per user.
        user_batch_size: Users per batch step.
        model_stage: ZenML model stage ("production" or "staging").
        batch_output_path: Base path (local or S3) for Parquet output.
        dynamodb_table: DynamoDB table. If set, loads recs there after writing.
        dynamodb_partition_key: DynamoDB partition key attribute name.
    """

    # ── Batch fan-out flow ─────────────────────────────────────────────────────
    als_model, model_version_name = load_als_model(model_stage=model_stage)

    after = []
    for i in range(n_batches):
        batch = predict_user_batch(
            als_model=als_model,
            batch_idx=i,
            user_batch_size=user_batch_size,
            batch_top_k=batch_top_k,
            model_version_name=model_version_name,
            id=f"batch_{i}",
        )
        after.append(batch)

    batch_report = collect_batch_recommendations(
        n_batches=n_batches,
        model_version_name=model_version_name,
        batch_output_path=batch_output_path,
        batch_top_k=batch_top_k,
        dynamodb_table=dynamodb_table,
        dynamodb_partition_key=dynamodb_partition_key,
        after=after,
    )
    logger.info("Batch job report: %s", batch_report)

    # ── Real-time endpoint flow ────────────────────────────────────────────────
    serving_image_uri = build_serving_image()
    endpoint_url = deploy_endpoint(serving_image_uri=serving_image_uri)
    logger.info("Real-time endpoint deployed at: %s", endpoint_url)


# Create a snapshot of the serving pipeline for reproducibility and versioning
serving_pipeline.create_snapshot(
    name=CFG_SERVING_PIPELINE_SNAPSHOT_NAME,
    description=CFG_SERVING_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=["matrix_factorization", "als", "serving"],
    replace=True
)


# TODO: Trigger monitoring pipeline on schedule (e.g., hourly) to check for drift and trigger retraining if needed.
