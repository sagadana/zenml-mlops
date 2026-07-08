"""
pipelines/matrix_factorization/data_pipeline.py

Data ingestion and preparation pipeline.

Steps:
  ingest_data → validate_data → build_encoders → split_data

Run:
  python run.py --pipeline data --config workflows/matrix_factorization/configs/local.yaml
"""

from zenml import pipeline

from workflows.matrix_factorization.steps.data_ingestion.ingest import ingest_data
from workflows.matrix_factorization.steps.data_validation.validate import validate_data
from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders
from workflows.matrix_factorization.steps.feature_engineering.split import split_data


@pipeline(name="matrix_factorization_data", enable_cache=True)
def data_pipeline(
    dataset_size: str = "1m",
    n_dask_partitions: int = 4,
    min_sparsity: float = 0.95,
    min_ratings: int = 100_000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> None:
    """
    End-to-end data preparation pipeline for Matrix Factorization.

    Downloads MovieLens dataset, validates it, builds user/item encoders,
    and produces stratified train/val/test splits as ZenML artifacts.
    All downstream pipelines (hpo, training) consume artifacts from this pipeline.

    Args:
        dataset_size: "1m" (local dev) or "25m" (AWS).
        n_dask_partitions: Number of Dask partitions for the rating DataFrame.
        min_sparsity: Minimum required sparsity for validation.
        min_ratings: Minimum number of ratings required.
        train_ratio: Training fraction (default 0.8).
        val_ratio: Validation fraction (default 0.1).
        test_ratio: Test fraction (default 0.1).
    """
    raw_ratings = ingest_data(
        dataset_size=dataset_size,
        n_dask_partitions=n_dask_partitions,
    )

    validation_report = validate_data(
        raw_ratings=raw_ratings,
        min_sparsity=min_sparsity,
        min_ratings=min_ratings,
    )

    user_encoder, item_encoder = build_encoders(raw_ratings=raw_ratings)

    train_data, val_data, test_data = split_data(
        raw_ratings=raw_ratings,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
