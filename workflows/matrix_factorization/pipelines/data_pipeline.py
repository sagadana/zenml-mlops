"""
pipelines/matrix_factorization/data_pipeline.py

Data preparation pipeline for ALS feature artifacts.

Steps:
  ingest_data -> validate_data -> build_encoders -> create_features_artifact

Run:
  python run.py run --workflow matrix_factorization --pipeline data_pipeline --config workflows/matrix_factorization/configs/local/data_pipeline.yaml
  python run.py run --workflow matrix_factorization --pipeline data_pipeline --config workflows/matrix_factorization/configs/aws/data_pipeline.yaml --stack aws_stack
"""

from zenml import pipeline

from workflows.matrix_factorization.configs import (
    CFG_DATA_PIPELINE_NAME,
    CFG_DATA_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_DATA_PIPELINE_SNAPSHOT_NAME,
)
from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_data
from workflows.matrix_factorization.steps.data_validation.validate import validate_data
from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders
from workflows.matrix_factorization.steps.feature_engineering.features_artifact import (
    create_features_artifact,
)


@pipeline(name=CFG_DATA_PIPELINE_NAME, enable_cache=True)
def data_pipeline() -> None:
    """Build and persist encoder features used by the training pipeline."""
    raw_ratings = ingest_data()
    validation = validate_data(raw_ratings=raw_ratings)

    user_encoder, item_encoder = build_encoders(
        raw_ratings=raw_ratings,
        after=[validation],
    )

    create_features_artifact(
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )


data_pipeline.create_snapshot(
    name=CFG_DATA_PIPELINE_SNAPSHOT_NAME,
    description=CFG_DATA_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=["matrix_factorization", "als", "data"],
    replace=True,
)
