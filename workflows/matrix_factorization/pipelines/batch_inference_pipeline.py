"""
pipelines/matrix_factorization/batch_inference_pipeline.py

Batch fan-out/fan-in recommendation pipeline.

Flow:
  load_als_model
      ↓
  get_user_ids (→ user_ids, batch_size)
      ↓
  get_user_batch_slice_0  get_user_batch_slice_1  ...  get_user_batch_slice_{n-1}   ← fan-out (slice)
      ↓                       ↓                             ↓
  predict_user_batch_0   predict_user_batch_1  ...  predict_user_batch_{n-1}        ← fan-out (predict)
      ↓                       ↓                             ↓
  collect_batch_recommendations                                                      ← fan-in (writes to S3 / DynamoDB)

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
    CFG_BATCH_USER_PREDICTION_OUTPUT,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.steps.prediction.batch_predict import (
    collect_batch_recommendations,
    load_als_model,
)
from workflows.matrix_factorization.steps.prediction.batch_predict_user import (
    get_user_batch_slice,
    get_user_ids,
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
    user_ids, batch_size = get_user_ids(
        model=als_model,
        n_batches=n_batches,
        min_user_batch_size=min_user_batch_size,
    )

    # Fan-out: slice user_ids into n_batches and predict per slice
    step_prefix = "predict_user_batch_"
    after = []
    for i in range(n_batches):
        batch_ids = get_user_batch_slice(
            user_ids=user_ids,
            batch_size=batch_size,
            batch_idx=i,
            id=f"get_user_batch_slice_{i}",
        )
        batch = predict_user_batch(
            batch_ids=batch_ids,
            model=als_model,
            model_name=model_name,
            model_version=model_version,
            batch_top_k=batch_top_k,
            id=f"{step_prefix}{i}",
        )
        after.append(batch)

    # Fan-in: collect all batch predictions and write to S3 / DynamoDB
    batch_report = collect_batch_recommendations(
        n_batches=n_batches,
        model_name=model_name,
        model_version=model_version,
        batch_output_path=batch_output_path,
        batch_top_k=batch_top_k,
        step_prefix=step_prefix,
        output_name=CFG_BATCH_USER_PREDICTION_OUTPUT,
        dynamodb_table=dynamodb_table,
        dynamodb_partition_key=dynamodb_partition_key,
        dynamodb_region=dynamodb_region,
        seaweedfs_s3_internal_endpoint=seaweedfs_s3_internal_endpoint,
        zenml_local_s3_secret_name=zenml_local_s3_secret_name,
        after=after,
    )
    logger.info("Batch job report: %s", batch_report)


batch_inference_pipeline.create_snapshot(
    name=CFG_BATCH_INFERENCE_PIPELINE_SNAPSHOT_NAME,
    description=CFG_BATCH_INFERENCE_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "batch_inference"],
    replace=True,
)
