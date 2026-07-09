"""
steps/data_ingestion/ingest.py

ZenML step: ingest_data

Downloads MovieLens dataset (1M or 25M), parses ratings into a Dask DataFrame
partitioned by userId range, and returns it as a ZenML artifact.

Config parameters (from pipeline YAML):
    dataset_size: "1m" | "25m"
    n_dask_partitions: int
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Annotated

import dask_expr as dd
import pandas as pd
from zenml import step

from workflows.matrix_factorization.materializers.dask_dataframe_materializer import (
    DaskDataFrameMaterializer,
)

logger = logging.getLogger(__name__)

_MOVIELENS_URLS = {
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

_RATINGS_FILES = {
    "1m": "ml-1m/ratings.dat",
    "25m": "ml-25m/ratings.csv",
}


def _download_movielens(dataset_size: str, cache_dir: Path) -> Path:
    """Download and extract MovieLens zip if not already cached."""
    import ssl
    import urllib.request

    url = _MOVIELENS_URLS[dataset_size]
    zip_path = cache_dir / f"ml-{dataset_size}.zip"
    extract_dir = cache_dir / f"ml-{dataset_size}-extracted"

    if extract_dir.exists():
        logger.info("Using cached MovieLens %s at %s", dataset_size, extract_dir)
        return extract_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MovieLens %s from %s ...", dataset_size, url)

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    logger.warning(
        "SSL certificate verification disabled for download (self-signed cert detected)."
    )

    chunk_size = 1024 * 64  # 64 KB
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            while chunk := response.read(chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = min(downloaded / total_size * 100, 100)
                    logger.debug("  %.1f%% (%d / %d bytes)", pct, downloaded, total_size)
    logger.info("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    zip_path.unlink()  # remove zip to save space
    logger.info("Extracted to %s", extract_dir)
    return extract_dir


def _parse_ratings(extract_dir: Path, dataset_size: str) -> pd.DataFrame:
    """Parse ratings file into a pandas DataFrame with canonical column names."""
    ratings_rel = _RATINGS_FILES[dataset_size]
    ratings_path = extract_dir / ratings_rel

    if dataset_size == "1m":
        # Format: UserID::MovieID::Rating::Timestamp
        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            dtype={
                "userId": "int32",
                "movieId": "int32",
                "rating": "float32",
                "timestamp": "int64",
            },
        )
    else:
        # Format: userId,movieId,rating,timestamp (CSV with header)
        df = pd.read_csv(
            ratings_path,
            dtype={
                "userId": "int32",
                "movieId": "int32",
                "rating": "float32",
                "timestamp": "int64",
            },
        )

    logger.info(
        "Parsed %d ratings (%d users, %d items)",
        len(df),
        df["userId"].nunique(),
        df["movieId"].nunique(),
    )
    return df


@step(enable_cache=True, output_materializers={"raw_ratings": DaskDataFrameMaterializer})
def ingest_data(
    dataset_size: str = "1m",
    n_dask_partitions: int = 4,
) -> Annotated[dd.DataFrame, "raw_ratings"]:
    """
    Download and ingest MovieLens ratings into a Dask DataFrame.

    Args:
        dataset_size: "1m" for MovieLens 1M (local dev) or "25m" for 25M (AWS).
        n_dask_partitions: Number of Dask partitions. Each partition covers a
            range of userId values and is later used as one ALS update task.

    Returns:
        Dask DataFrame with columns: userId, movieId, rating, timestamp.
        Partitioned by userId range.
    """
    if dataset_size not in _MOVIELENS_URLS:
        raise ValueError(
            f"Unknown dataset_size: {dataset_size!r}. Choose from {list(_MOVIELENS_URLS)}"
        )

    # Cache raw downloads in ./data/ (gitignored)
    cache_dir = Path(os.environ.get("MOVIELENS_CACHE_DIR", "./data"))
    extract_dir = _download_movielens(dataset_size, cache_dir)
    df_pandas = _parse_ratings(extract_dir, dataset_size)

    # Convert to Dask DataFrame partitioned by userId range for ALS efficiency
    ddf = dd.from_pandas(df_pandas, npartitions=n_dask_partitions)
    # Repartition by userId so each partition covers contiguous user blocks
    ddf = ddf.set_index("userId").repartition(npartitions=n_dask_partitions)
    # Persist userId as a column as well (it becomes the index after set_index)
    ddf = ddf.reset_index()

    logger.info(
        "Created Dask DataFrame: %d partitions, %d rows",
        ddf.npartitions,
        len(df_pandas),
    )
    return ddf
