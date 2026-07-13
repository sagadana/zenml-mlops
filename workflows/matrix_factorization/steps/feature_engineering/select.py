from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)


@step(enable_cache=True)
def select_feature_columns(
    features: pd.DataFrame,
    columns: list[str] | None = None,
) -> Annotated[pd.DataFrame, "selected_features"]:
    """
    Select a subset of columns from the input features DataFrame.

    Args:
        features: Input features DataFrame.
        columns: List of column names to select.

    Returns:
        DataFrame containing only the selected columns.
    """
    if columns is None:
        columns = list()

    if not columns:
        raise ValueError("columns list cannot be empty.")

    missing_columns = [col for col in columns if col not in features.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in features DataFrame: {missing_columns}")

    selected_features = features[columns].copy()
    logger.info("Selected feature columns: %s", selected_features.columns.tolist())
    return selected_features
