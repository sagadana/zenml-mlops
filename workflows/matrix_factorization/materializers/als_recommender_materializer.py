"""
materializers/als_recommender_materializer.py

ZenML materializer for ALSRecommender model objects.

Serialization format: cloudpickle (supports Numba-compiled functions and
numpy arrays stored in the model).

Usage: imported automatically by ZenML when a step returns ALSRecommender.
"""

from __future__ import annotations

import os

import cloudpickle
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from workflows.matrix_factorization.models.als_recommender import ALSRecommender

_PICKLE_FILENAME = "als_recommender.pkl"


class ALSRecommenderMaterializer(BaseMaterializer):
    """ZenML materializer that stores ALSRecommender via cloudpickle."""

    ASSOCIATED_TYPES = (ALSRecommender,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: type[ALSRecommender]) -> ALSRecommender:
        """Load ALSRecommender from cloudpickle file."""
        pkl_path = os.path.join(self.uri, _PICKLE_FILENAME)
        with self.artifact_store.open(pkl_path, "rb") as f:
            return cloudpickle.load(f)

    def save(self, model: ALSRecommender) -> None:
        """Save ALSRecommender as cloudpickle file."""
        pkl_path = os.path.join(self.uri, _PICKLE_FILENAME)
        with self.artifact_store.open(pkl_path, "wb") as f:
            cloudpickle.dump(model, f)
