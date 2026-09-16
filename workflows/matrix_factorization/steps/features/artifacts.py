"""
steps/feature_engineering/artifacts.py

ZenML steps for persisting and loading feature artifacts independently.

Each artifact (raw ratings, train/validation datasets, user/item encoders) is
saved and loaded as its own named ZenML artifact using the built-in pandas
materializer, so downstream steps can load only the artifact they need instead
of deserializing one bundled payload.
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml.enums import ArtifactType

from workflows.matrix_factorization.configs import BUILD_VERSION, CFG_FEATURES_ARTIFACTS

logger = logging.getLogger(__name__)


def _load_artifact[T](name: str, expected_type: type[T], version: str | None = None) -> T:
    """Load the latest (or specified) version of a single named artifact from the ZenML artifact store."""
    client = Client()
    artifact_version = None

    try:
        # Try to get the latest artifact version by name and project
        artifact_version = client.get_artifact_version(
            name_id_or_prefix=name, project=client.active_project.name, version=version
        )
    except Exception:
        try:
            # If the above fails, list all versions for this artifact name and take the latest one
            versions = client.list_artifact_versions(name=name, project=client.active_project.name)
            if hasattr(versions, "items"):
                versions = versions.items
            if versions:
                artifact_version = versions[0]
        except Exception as exc:
            raise ValueError(
                f"Could not find artifact version for '{name}'. "
                "Run data_pipeline first to generate feature artifacts."
            ) from exc

    if artifact_version is None:
        raise ValueError(f"Artifact '{name}' not found. Run data_pipeline first.")

    data = artifact_version.load()
    if not isinstance(data, expected_type):
        raise TypeError(f"Artifact '{name}' has unsupported type: {type(data)!r}.")
    return data


@step(enable_cache=True)
def create_features_artifact(
    raw_ratings: pd.DataFrame,
    train_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
) -> tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(
            name=CFG_FEATURES_ARTIFACTS.RAW_RATINGS.value,
            artifact_type=ArtifactType.DATA,
            tags=["als", "features", "matrix_factorization"],
            version=BUILD_VERSION,
        ),
    ],
    Annotated[
        pd.DataFrame,
        ArtifactConfig(
            name=CFG_FEATURES_ARTIFACTS.TRAIN_DATASET.value,
            artifact_type=ArtifactType.DATA,
            tags=["als", "features", "matrix_factorization"],
            version=BUILD_VERSION,
        ),
    ],
    Annotated[
        pd.DataFrame,
        ArtifactConfig(
            name=CFG_FEATURES_ARTIFACTS.VALIDATION_DATASET.value,
            artifact_type=ArtifactType.DATA,
            tags=["als", "features", "matrix_factorization"],
            version=BUILD_VERSION,
        ),
    ],
    Annotated[
        pd.Series,
        ArtifactConfig(
            name=CFG_FEATURES_ARTIFACTS.USER_ENCODER.value,
            artifact_type=ArtifactType.DATA,
            tags=["als", "features", "matrix_factorization"],
            version=BUILD_VERSION,
        ),
    ],
    Annotated[
        pd.Series,
        ArtifactConfig(
            name=CFG_FEATURES_ARTIFACTS.ITEM_ENCODER.value,
            artifact_type=ArtifactType.DATA,
            tags=["als", "features", "matrix_factorization"],
            version=BUILD_VERSION,
        ),
    ],
]:
    """Persist raw ratings, encoded train/validation datasets, and user/item encoders as independently loadable artifacts."""
    return raw_ratings, train_dataset, validation_dataset, user_encoder, item_encoder


@step(enable_cache=False)
def load_features_artifact(
    version: str = BUILD_VERSION,
) -> tuple[
    Annotated[pd.Series, "user_encoder"],
    Annotated[pd.Series, "item_encoder"],
    Annotated[pd.DataFrame, "train_dataset"],
    Annotated[pd.DataFrame, "validation_dataset"],
]:
    """Load latest user/item encoders and encoded train/validation datasets, each from its own artifact."""
    user_encoder = _load_artifact(
        CFG_FEATURES_ARTIFACTS.USER_ENCODER.value, pd.Series, version=version
    )
    item_encoder = _load_artifact(
        CFG_FEATURES_ARTIFACTS.ITEM_ENCODER.value, pd.Series, version=version
    )
    train_dataset = _load_artifact(
        CFG_FEATURES_ARTIFACTS.TRAIN_DATASET.value, pd.DataFrame, version=version
    )
    validation_dataset = _load_artifact(
        CFG_FEATURES_ARTIFACTS.VALIDATION_DATASET.value, pd.DataFrame, version=version
    )

    logger.info(
        "Loaded features artifacts with %d train rows, %d validation rows, %d users and %d items",
        len(train_dataset),
        len(validation_dataset),
        len(user_encoder),
        len(item_encoder),
    )
    return user_encoder, item_encoder, train_dataset, validation_dataset


@step(enable_cache=False)
def load_raw_ratings_artifact(
    version: str = BUILD_VERSION,
    sample_fraction: float | None = None,
) -> Annotated[pd.DataFrame, "raw_ratings"]:
    """Load only the raw_ratings artifact."""
    raw_ratings = _load_artifact(
        CFG_FEATURES_ARTIFACTS.RAW_RATINGS.value, pd.DataFrame, version=version
    )

    logger.info(
        "Loaded raw_ratings artifact '%s' with %d rows",
        CFG_FEATURES_ARTIFACTS.RAW_RATINGS.value,
        len(raw_ratings),
    )
    if sample_fraction is not None:
        raw_ratings = raw_ratings.sample(frac=sample_fraction).reset_index(drop=True)
    return raw_ratings


@step(enable_cache=False)
def load_train_dataset_artifact(
    version: str = BUILD_VERSION,
    sample_fraction: float | None = None,
) -> Annotated[pd.DataFrame, "train_dataset"]:
    """Load only the train_dataset artifact."""
    train_dataset = _load_artifact(
        CFG_FEATURES_ARTIFACTS.TRAIN_DATASET.value, pd.DataFrame, version=version
    )

    logger.info(
        "Loaded train_dataset artifact '%s' with %d rows",
        CFG_FEATURES_ARTIFACTS.TRAIN_DATASET.value,
        len(train_dataset),
    )
    if sample_fraction is not None:
        train_dataset = train_dataset.sample(frac=sample_fraction).reset_index(drop=True)
    return train_dataset
