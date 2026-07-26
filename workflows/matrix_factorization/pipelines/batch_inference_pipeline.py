"""
pipelines/matrix_factorization/batch_inference_pipeline.py

Batch fan-out/fan-in recommendation pipeline.

Flow:
  load_als_model
      ↓
  get_total_users (→ total_users: int, batch_size: int)
      ↓
  predict_user_batch_0   predict_user_batch_1  ...  predict_user_batch_{n-1}        ← fan-out (predict + write)
      ↓                       ↓                             ↓
  collect_batch_recommendations                                                      ← fan-in (aggregates summaries)

Run:
  python run.py run --workflow matrix_factorization --pipeline batch_inference_pipeline --config workflows/matrix_factorization/configs/local/batch_inference_pipeline.yaml
  python run.py run --workflow matrix_factorization --pipeline batch_inference_pipeline --config workflows/matrix_factorization/configs/aws/batch_inference_pipeline.yaml --stack aws_stack
"""

import logging

from zenml import pipeline
from zenml.enums import ModelStages

from workflows.matrix_factorization.configs import (
    CFG_BATCH_INFERENCE_PIPELINE_NAME,
    CFG_BATCH_INFERENCE_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_BATCH_INFERENCE_PIPELINE_SNAPSHOT_NAME,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.steps.prediction.batch_predict import (
    collect_batch_recommendations,
    load_als_model,
)
from workflows.matrix_factorization.steps.prediction.batch_predict_user import (
    get_total_users,
    predict_user_batch,
)

logger = logging.getLogger(__name__)


@pipeline(name=CFG_BATCH_INFERENCE_PIPELINE_NAME)
def batch_inference_pipeline(
    n_batches: int = 1,
    batch_top_k: int = 50,
    min_user_batch_size: int = 10_000,
    model_stage: ModelStages = ModelStages.STAGING,
    batch_output_path: str = "./predictions/batch",
    dynamodb_table: str | None = None,
    dynamodb_partition_key: str = "id",
    dynamodb_region: str | None = None,
    seaweedfs_s3_internal_endpoint: str | None = None,
    zenml_local_s3_secret_name: str | None = None,
) -> None:
    """Run batch recommendation inference with fan-out/fan-in execution."""
    als_model, model_name, model_version = load_als_model(model_stage=model_stage)
    total_users, batch_size = get_total_users(
        model=als_model,
        n_batches=n_batches,
        min_user_batch_size=min_user_batch_size,
    )

    # Fan-out: each step computes its own slice and writes predictions independently
    step_prefix = "predict_user_batch_"
    after = []
    for i in range(n_batches):
        batch = predict_user_batch(
            total_users=total_users,
            batch_size=batch_size,
            batch_idx=i,
            model=als_model,
            model_name=model_name,
            model_version=model_version,
            batch_top_k=batch_top_k,
            batch_output_path=batch_output_path,
            dynamodb_table=dynamodb_table,
            dynamodb_partition_key=dynamodb_partition_key,
            dynamodb_region=dynamodb_region,
            seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
            zenml_local_s3_secret_name=zenml_local_s3_secret_name,
            id=f"{step_prefix}{i}",
        )
        after.append(batch)

    # Fan-in: collect per-batch summaries and return an aggregated report
    batch_report = collect_batch_recommendations(
        n_batches=n_batches,
        step_prefix=step_prefix,
        after=after,
    )
    logger.info("Batch job report: %s", batch_report)


batch_inference_pipeline.create_snapshot(
    name=CFG_BATCH_INFERENCE_PIPELINE_SNAPSHOT_NAME,
    description=CFG_BATCH_INFERENCE_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "batch_inference"],
    replace=True,
)
