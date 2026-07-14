"""
steps/feature_engineering/features_artifact.py

ZenML steps for packaging and loading encoder artifacts.
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from zenml import step
from zenml.client import Client

from workflows.matrix_factorization.configs import CFG_FEATURES_ARTIFACT_NAME

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def create_features_artifact(
    raw_ratings: pd.DataFrame,
    user_encoder: pd.Series,
    item_encoder: pd.Series,
) -> Annotated[dict[str, pd.DataFrame | pd.Series], CFG_FEATURES_ARTIFACT_NAME]:
    """Package raw ratings and user/item encoders into a single named artifact."""
    return {
        "raw_ratings": raw_ratings,
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
    }


@step(enable_cache=False)
def load_features_artifact() -> (
    tuple[
        Annotated[pd.DataFrame, "raw_ratings"],
        Annotated[pd.Series, "user_encoder"],
        Annotated[pd.Series, "item_encoder"],
    ]
):
    """Load latest raw ratings + encoders artifact by name from the ZenML artifact store."""
    client = Client()
    artifact_version = None

    try:
        artifact_version = client.get_artifact_version(name_id_or_prefix=CFG_FEATURES_ARTIFACT_NAME)
    except Exception:
        try:
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
    if not isinstance(features, dict):
        raise TypeError(
            f"Artifact '{CFG_FEATURES_ARTIFACT_NAME}' has unsupported type: {type(features)!r}."
        )

    raw_ratings = features.get("raw_ratings")
    user_encoder = features.get("user_encoder")
    item_encoder = features.get("item_encoder")

    if not isinstance(raw_ratings, pd.DataFrame):
        raise TypeError(
            f"Artifact '{CFG_FEATURES_ARTIFACT_NAME}' is missing required raw_ratings DataFrame."
        )

    if not isinstance(user_encoder, pd.Series) or not isinstance(item_encoder, pd.Series):
        raise TypeError(
            f"Artifact '{CFG_FEATURES_ARTIFACT_NAME}' is missing required user_encoder/item_encoder Series."
        )

    logger.info(
        "Loaded features artifact '%s' with %d ratings, %d users and %d items",
        CFG_FEATURES_ARTIFACT_NAME,
        len(raw_ratings),
        len(user_encoder),
        len(item_encoder),
    )
    return raw_ratings, user_encoder, item_encoder
