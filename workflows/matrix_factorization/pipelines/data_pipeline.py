"""
pipelines/matrix_factorization/data_pipeline.py

Data preparation pipeline for ALS feature artifacts.

Steps:
  ingest_data -> validate_data -> preprocess_data -> build_encoders
  -> prepare_features -> split_data -> create_features_artifact

Run:
  python run.py run --workflow matrix_factorization --pipeline data_pipeline --config workflows/matrix_factorization/configs/local/data_pipeline.yaml
  python run.py run --workflow matrix_factorization --pipeline data_pipeline --config workflows/matrix_factorization/configs/aws/data_pipeline.yaml --stack aws_stack
"""

from zenml import pipeline

from workflows.matrix_factorization.configs import (
    BUILD_VERSION,
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
from workflows.matrix_factorization.steps.features.split import (
    prepare_features,
    split_data,
)


@pipeline(name=CFG_DATA_PIPELINE_NAME)
def data_pipeline() -> None:
    """Build and persist encoded train/validation features used by the training pipeline."""
    raw_ratings = ingest_data()
    validation = validate_data(raw_ratings=raw_ratings)

    processed_ratings = preprocess_data(
        raw_ratings=raw_ratings,
        after=[validation],
    )

    user_encoder, item_encoder = build_encoders(
        processed_ratings=processed_ratings,
    )

    features = prepare_features(
        raw_ratings=processed_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )

    train_dataset, validation_dataset = split_data(
        features=features,
    )

    # TODO: Get rid of the extra step and directly create the features artifact from the processed features
    create_features_artifact(
        raw_ratings=raw_ratings,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )


data_pipeline.create_snapshot(
    name=CFG_DATA_PIPELINE_SNAPSHOT_NAME,
    description=CFG_DATA_PIPELINE_SNAPSHOT_DESCRIPTION,
    tags=[CFG_WORKFLOW_NAME, "als", "data", BUILD_VERSION],
    replace=True,
)
