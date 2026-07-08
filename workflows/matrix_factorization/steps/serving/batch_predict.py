"""
steps/serving/batch_predict.py

ZenML step: generate_batch_recommendations

For each user in the encoder, generates top-K recommendations using the
production ALSRecommender model. Writes results to S3 as Parquet and
optionally loads them into a DynamoDB table for real-time lookup.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import pandas as pd
from zenml import step
from zenml.client import Client

from workflows.matrix_factorization.models.als_recommender import ALSRecommender

logger = logging.getLogger(__name__)

_MODEL_NAME = "als_movie_recommender"


@step(enable_cache=False)
def generate_batch_recommendations(
    batch_top_k: int = 50,
    batch_output_path: str = "s3://aips-zenml-predictions/batch",
    dynamodb_table: str | None = None,
    model_stage: str = "production",
) -> Annotated[dict, "batch_job_report"]:
    """
    Pre-compute top-K recommendations for all users and write to S3.

    Args:
        batch_top_k: Number of recommendations per user.
        batch_output_path: S3 path (or local path) for Parquet output.
        dynamodb_table: DynamoDB table name. If set, loads recommendations there.
            Table schema: userId (PK, String), recommendations (JSON), updated_at (TTL).
        model_stage: ZenML model stage to load from ("production" or "staging").

    Returns:
        Batch job report dict with counts and output path.
    """
    # Load model from ZenML Model Control Plane
    client = Client()
    model_version = client.get_model_version(_MODEL_NAME, model_stage)
    artifact = model_version.get_artifact("als_model")
    if artifact is None:
        raise ValueError(f"Model artifact 'als_model' not found for {_MODEL_NAME}")
    als_model: ALSRecommender = artifact.load()
    logger.info("Loaded %s for batch prediction", als_model)

    all_user_ids = np.array(als_model.user_encoder.index.tolist())
    logger.info("Generating top-%d recs for %d users...", batch_top_k, len(all_user_ids))

    # Generate recommendations in batches to avoid OOM
    batch_size = 10_000
    records = []
    for start in range(0, len(all_user_ids), batch_size):
        batch_ids = all_user_ids[start : start + batch_size]
        batch_results = als_model.batch_predict(batch_ids, top_k=batch_top_k)
        for result in batch_results:
            uid = result["user_id"]
            for rank_pos, rec in enumerate(result["recommendations"]):
                records.append(
                    {
                        "userId": uid,
                        "itemId": rec["item_id"],
                        "score": rec["score"],
                        "rank": rank_pos + 1,
                    }
                )

    df = pd.DataFrame(records)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    output_path = f"{batch_output_path}/{date_str}/recommendations.parquet"

    df.to_parquet(output_path, index=False)
    logger.info("Batch recommendations written to %s (%d rows)", output_path, len(df))

    if dynamodb_table:
        _load_to_dynamodb(df, dynamodb_table, batch_top_k)

    report = {
        "n_users": len(all_user_ids),
        "top_k": batch_top_k,
        "output_path": output_path,
        "n_records": len(df),
        "date": date_str,
        "dynamodb_loaded": dynamodb_table is not None,
    }
    return report


def _load_to_dynamodb(df: pd.DataFrame, table_name: str, top_k: int) -> None:
    """Write user recommendation lists to DynamoDB."""
    import json
    import time

    import boto3

    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)  # type: ignore[arg-type]
    ttl_seconds = int(time.time()) + 48 * 3600  # 48-hour TTL

    # Group by userId
    grouped = df.sort_values(["userId", "rank"]).groupby("userId")
    items_to_write = []
    for user_id, group in grouped:
        recs = group[["itemId", "score"]].rename(columns={"itemId": "item_id"}).to_dict("records")
        items_to_write.append(
            {
                "userId": str(user_id),
                "recommendations": json.dumps(recs[:top_k]),
                "updated_at": ttl_seconds,
            }
        )

    # Batch write (max 25 items per DynamoDB batch)
    with table.batch_writer() as batch:
        for item in items_to_write:
            batch.put_item(Item=item)

    logger.info(
        "Loaded %d user recommendation lists to DynamoDB table '%s'",
        len(items_to_write),
        table_name,
    )
