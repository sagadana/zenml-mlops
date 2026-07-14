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
    user_encoder: pd.Series,
    item_encoder: pd.Series,
) -> Annotated[dict[str, pd.Series], CFG_FEATURES_ARTIFACT_NAME]:
    """Package user/item encoders into a single named artifact."""
    return {
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
    }


@step(enable_cache=False)
def load_features_artifact(
    artifact_name: str = CFG_FEATURES_ARTIFACT_NAME,
) -> tuple[
    Annotated[pd.Series, "user_encoder"],
    Annotated[pd.Series, "item_encoder"],
]:
    """Load the latest encoder artifact by name from the ZenML artifact store."""
    client = Client()
    artifact_version = None

    try:
        artifact_version = client.get_artifact_version(name_id_or_prefix=artifact_name)
    except Exception:
        try:
            versions = client.list_artifact_versions(name=artifact_name)
            if hasattr(versions, "items"):
                versions = versions.items
            if versions:
                artifact_version = versions[0]
        except Exception as exc:
            raise ValueError(
                f"Could not find artifact version for '{artifact_name}'. "
                "Run data_pipeline first to generate encoder artifacts."
            ) from exc

    if artifact_version is None:
        raise ValueError(f"Artifact '{artifact_name}' not found. Run data_pipeline first.")

    features = artifact_version.load()
    if not isinstance(features, dict):
        raise TypeError(f"Artifact '{artifact_name}' has unsupported type: {type(features)!r}.")

    user_encoder = features.get("user_encoder")
    item_encoder = features.get("item_encoder")

    if not isinstance(user_encoder, pd.Series) or not isinstance(item_encoder, pd.Series):
        raise TypeError(f"Artifact '{artifact_name}' is missing required encoder Series keys.")

    logger.info(
        "Loaded encoder artifact '%s' with %d users and %d items",
        artifact_name,
        len(user_encoder),
        len(item_encoder),
    )
    return user_encoder, item_encoder
