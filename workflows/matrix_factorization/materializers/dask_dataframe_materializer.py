"""
materializers/dask_dataframe_materializer.py

ZenML materializer for Dask DataFrames.

Serialization format: Parquet (via dask.dataframe.to_parquet / read_parquet).
Storage location: determined by the active ZenML artifact store (local or S3).

Usage: imported automatically by ZenML when a step returns dd.DataFrame.
Registration: referenced in pipeline definitions via `materializer_classes=`.
"""

from __future__ import annotations

import os
from typing import Type

import dask.dataframe as dd
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


class DaskDataFrameMaterializer(BaseMaterializer):
    """ZenML materializer that stores Dask DataFrames as partitioned Parquet."""

    ASSOCIATED_TYPES = (dd.DataFrame,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[dd.DataFrame]) -> dd.DataFrame:
        """Load a Dask DataFrame from Parquet files in the artifact URI."""
        parquet_path = os.path.join(self.uri, "data")
        return dd.read_parquet(parquet_path)

    def save(self, df: dd.DataFrame) -> None:
        """Save a Dask DataFrame as Parquet files to the artifact URI."""
        parquet_path = os.path.join(self.uri, "data")
        df.to_parquet(
            parquet_path,
            write_index=True,
            overwrite=True,
            schema="infer",
        )
