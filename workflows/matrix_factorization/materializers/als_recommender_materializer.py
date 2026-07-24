"""
materializers/als_recommender_materializer.py

ZenML materializer for BaseRecommender model objects.

Handles all BaseRecommender subclasses (ALSRecommender, ALSImplicitRecommender)
using cloudpickle serialization. The ASSOCIATED_TYPES tuple covers the base class
so ZenML auto-selects this materializer for any registered subclass artifact.
"""

from __future__ import annotations

import os

import cloudpickle
from zenml.enums import ArtifactType, VisualizationType
from zenml.materializers.base_materializer import BaseMaterializer

from workflows.matrix_factorization.configs import CFG_MODEL_PICKLE_FILENAME
from workflows.matrix_factorization.models.base_recommender import BaseRecommender


class ALSRecommenderMaterializer(BaseMaterializer):
    """ZenML materializer for BaseRecommender subclasses (cloudpickle)."""

    ASSOCIATED_TYPES = (BaseRecommender,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: type[BaseRecommender]) -> BaseRecommender:
        """Load a BaseRecommender subclass from cloudpickle file."""
        pkl_path = os.path.join(self.uri, CFG_MODEL_PICKLE_FILENAME)
        with self.artifact_store.open(pkl_path, "rb") as f:
            return cloudpickle.load(f)

    def save(self, data: BaseRecommender) -> None:
        """Save a BaseRecommender subclass as cloudpickle file."""
        pkl_path = os.path.join(self.uri, CFG_MODEL_PICKLE_FILENAME)
        with self.artifact_store.open(pkl_path, "wb") as f:
            cloudpickle.dump(data, f)

    def save_visualizations(self, data: BaseRecommender) -> dict[str, VisualizationType]:
        """Return a string representation of the model for visualization."""

        visualization_uri = os.path.join(self.uri, "visualization.json")
        with self.artifact_store.open(visualization_uri, "w") as f:
            f.write(f"<code>{str(data)}</code>")

        return {
            visualization_uri: VisualizationType.HTML,
        }
