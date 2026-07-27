"""
steps/feature_engineering/artifacts.py

ZenML steps for packaging and loading encoder artifacts.
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml.enums import ArtifactType

from workflows.matrix_factorization.configs import CFG_FEATURES_ARTIFACT_NAME
from workflows.matrix_factorization.materializers import ALSFeaturesArtifactMaterializer
from workflows.matrix_factorization.models import (
    ModelFeaturesArtifact,
)

logger = logging.getLogger(__name__)


def _load_features_artifact_payload() -> ModelFeaturesArtifact:
    """Load raw artifact payload by name from the ZenML artifact store."""
    client = Client()
    artifact_version = None

    try:
        # Try to get the latest artifact version by name and project
        artifact_version = client.get_artifact_version(
            name_id_or_prefix=CFG_FEATURES_ARTIFACT_NAME, project=client.active_project.name
        )
    except Exception:
        try:
            # If the above fails, list all feature artifact versions and take the latest one
            versions = client.list_artifact_versions(name=CFG_FEATURES_ARTIFACT_NAME)
            if hasattr(versions, "items"):
                versions = versions.items
            if versions:
                artifact_version = versions[0]
        except Exception as exc:
            raise ValueError(
                f"Could not find artifact version for '{CFG_FEATURES_ARTIFACT_NAME}'. "
                "Run data_pipeline first to generate encoder artifacts."
            ) from exc

    if artifact_version is None:
        raise ValueError(
            f"Artifact '{CFG_FEATURES_ARTIFACT_NAME}' not found. Run data_pipeline first."
        )

    features = artifact_version.load()
    if isinstance(features, ModelFeaturesArtifact):
        return features

    if isinstance(features, dict):
        raw_ratings = features.get("raw_ratings")
        user_encoder = features.get("user_encoder")
        item_encoder = features.get("item_encoder")
        scaled_ratings = features.get("scaled_ratings")

        if not isinstance(user_encoder, pd.Series) or not isinstance(item_encoder, pd.Series):
            raise TypeError(
                f"Artifact '{CFG_FEATURES_ARTIFACT_NAME}' is missing required user_encoder/item_encoder."
            )
        if not isinstance(scaled_ratings, pd.DataFrame) or not isinstance(
            raw_ratings, pd.DataFrame
        ):
            raise TypeError(
                f"Artifact '{CFG_FEATURES_ARTIFACT_NAME}' is missing required scaled_ratings/raw_ratings DataFrame."
            )

        return ModelFeaturesArtifact(
            raw_ratings=raw_ratings,
            scaled_ratings=scaled_ratings,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
        )

    raise TypeError(
        f"Artifact '{CFG_FEATURES_ARTIFACT_NAME}' has unsupported type: {type(features)!r}."
    )


@step(
    enable_cache=True,
    output_materializers={
        CFG_FEATURES_ARTIFACT_NAME: ALSFeaturesArtifactMaterializer,
    },
)
def create_features_artifact(
    raw_ratings: pd.DataFrame,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
    scaled_ratings: pd.DataFrame,
) -> Annotated[
    ModelFeaturesArtifact,
    ArtifactConfig(
        name=CFG_FEATURES_ARTIFACT_NAME,
        artifact_type=ArtifactType.DATA,
        tags=["als", "features", "matrix_factorization"],
    ),
]:
    """Package raw ratings, scaled ratings, and user/item encoders into a single named artifact."""
    return ModelFeaturesArtifact(
        raw_ratings=raw_ratings,
        scaled_ratings=scaled_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
    )


@step(enable_cache=False)
def load_features_artifact() -> tuple[
    Annotated[pd.Series, "user_encoder"],
    Annotated[pd.Series, "item_encoder"],
    Annotated[pd.DataFrame, "scaled_ratings"],
]:
    """Load latest raw ratings + encoders + scaled ratings artifact by name from the ZenML artifact store."""
    features = _load_features_artifact_payload()

    logger.info(
        "Loaded features artifact '%s' with %d ratings, %d users and %d items",
        CFG_FEATURES_ARTIFACT_NAME,
        len(features.scaled_ratings),
        len(features.user_encoder),
        len(features.item_encoder),
    )
    return features.user_encoder, features.item_encoder, features.scaled_ratings


@step(enable_cache=False)
def load_raw_ratings_artifact() -> Annotated[pd.DataFrame, "raw_ratings"]:
    """Load only the raw_ratings DataFrame from the named features artifact."""
    features = _load_features_artifact_payload()

    logger.info(
        "Loaded raw_ratings from artifact '%s' with %d rows",
        CFG_FEATURES_ARTIFACT_NAME,
        len(features.raw_ratings),
    )
    return features.raw_ratings


@step(enable_cache=False)
def load_scaled_ratings_artifact() -> Annotated[pd.DataFrame, "scaled_ratings"]:
    """Load only the scaled_ratings DataFrame from the named features artifact."""
    features = _load_features_artifact_payload()

    logger.info(
        "Loaded scaled_ratings from artifact '%s' with %d rows",
        CFG_FEATURES_ARTIFACT_NAME,
        len(features.scaled_ratings),
    )
    return features.scaled_ratings
