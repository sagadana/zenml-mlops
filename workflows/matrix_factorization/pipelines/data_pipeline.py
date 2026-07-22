"""
pipelines/matrix_factorization/data_pipeline.py

Data preparation pipeline for ALS feature artifacts.

Steps:
  ingest_data -> validate_data -> preprocess_data -> build_encoders -> create_features_artifact

Run:
  python run.py run --workflow matrix_factorization --pipeline data_pipeline --config workflows/matrix_factorization/configs/local/data_pipeline.yaml
  python run.py run --workflow matrix_factorization --pipeline data_pipeline --config workflows/matrix_factorization/configs/aws/data_pipeline.yaml --stack aws_stack
"""

from zenml import pipeline

from workflows.matrix_factorization.configs import (
    CFG_DATA_PIPELINE_NAME,
    CFG_DATA_PIPELINE_SNAPSHOT_DESCRIPTION,
    CFG_DATA_PIPELINE_SNAPSHOT_NAME,
    CFG_WORKFLOW_NAME,
)
from workflows.matrix_factorization.steps.data.ingest import ingest_data
from workflows.matrix_factorization.steps.data.preprocess import preprocess_data
from workflows.matrix_factorization.steps.data.validate import validate_data
from workflows.matrix_factorization.steps.features.artifacts import (
    create_features_artifact,
)
from workflows.matrix_factorization.steps.features.encoders import build_encoders


@pipeline(name=CFG_DATA_PIPELINE_NAME)
def data_pipeline() -> None:
    """Build and persist encoder features used by the training pipeline."""
    raw_ratings = ingest_data()
    validation = validate_data(raw_ratings=raw_ratings)

    processed_ratings = preprocess_data(
        raw_ratings=raw_ratings,
        after=[validation],
    )

    user_encoder, item_encoder, scaled_ratings = build_encoders(
        raw_ratings=processed_ratings,
    )

    create_features_artifact(
        raw_ratings=raw_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        scaled_ratings=scaled_ratings,
    )


data_pipeline.create_snapshot(
    name=CFG_DATA_PIPELINE_SNAPSHOT_NAME,
    description=CFG_DATA_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "data"],
    replace=True,
)
