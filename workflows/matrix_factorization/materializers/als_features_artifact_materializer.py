"""
materializers/als_recommender_materializer.py

ZenML materializer for BaseRecommender model objects.

Handles all BaseRecommender subclasses (ALSRecommender, ALSImplicitRecommender)
using cloudpickle serialization. The ASSOCIATED_TYPES tuple covers the base class
so ZenML auto-selects this materializer for any registered subclass artifact.
"""

from __future__ import annotations

import json
import os

import cloudpickle
from zenml.enums import ArtifactType, VisualizationType
from zenml.materializers.base_materializer import BaseMaterializer

from workflows.matrix_factorization.configs import CFG_FEATURES_PICKLE_FILENAME
from workflows.matrix_factorization.models import ModelFeaturesArtifact


class ALSFeaturesArtifactMaterializer(BaseMaterializer):
    """ZenML materializer for ModelFeaturesArtifact (cloudpickle)."""

    ASSOCIATED_TYPES = (ModelFeaturesArtifact,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: type[ModelFeaturesArtifact]) -> ModelFeaturesArtifact:
        """Load a ModelFeaturesArtifact from cloudpickle file."""
        pkl_path = os.path.join(self.uri, CFG_FEATURES_PICKLE_FILENAME)
        with self.artifact_store.open(pkl_path, "rb") as f:
            return cloudpickle.load(f)

    def save(self, data: ModelFeaturesArtifact) -> None:
        """Save a ModelFeaturesArtifact as cloudpickle file."""
        pkl_path = os.path.join(self.uri, CFG_FEATURES_PICKLE_FILENAME)
        with self.artifact_store.open(pkl_path, "wb") as f:
            cloudpickle.dump(data, f)

    def save_visualizations(self, data: ModelFeaturesArtifact) -> dict[str, VisualizationType]:
        """Return a string representation of the model for visualization."""

        visualization_uri = os.path.join(self.uri, "visualization.json")
        with self.artifact_store.open(visualization_uri, "w") as f:
            f.write(
                json.dumps(
                    {
                        "n_users": len(data.user_encoder),
                        "n_items": len(data.item_encoder),
                        "shape_raw_ratings": data.raw_ratings.shape,
                        "shape_scaled_ratings": data.scaled_ratings.shape,
                    },
                    indent=4,
                )
            )

        return {
            visualization_uri: VisualizationType.JSON,
        }
