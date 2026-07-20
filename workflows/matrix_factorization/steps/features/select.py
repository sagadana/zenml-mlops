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
    force: bool = False,
) -> Annotated[pd.DataFrame, "selected_features"]:
    """
    Select a subset of columns from the input features DataFrame.

    Args:
        features: Input features DataFrame.
        columns: List of column names to select.
        force: If True, raise an error if the input DataFrame is empty or if any of the specified columns are missing. If False, return an empty DataFrame in these cases.

    Returns:
        DataFrame containing only the selected columns.
    """
    if columns is None:
        columns = list()

    if not columns:
        raise ValueError("columns list cannot be empty.")

    # Empty DataFrame check
    if features.empty:
        if force:
            raise ValueError("Input features DataFrame is empty. Cannot select columns.")

        logger.warning("Input features DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=columns)

    # Check for missing columns
    missing_columns = [col for col in columns if col not in features.columns]
    if missing_columns:
        if force:
            raise ValueError(f"Missing columns in features DataFrame: {missing_columns}")

        logger.warning(
            "Missing columns in features DataFrame: %s. Returning empty DataFrame.",
            missing_columns,
        )
        return pd.DataFrame(columns=columns)

    selected_features = features[columns].copy()
    logger.info("Selected feature columns: %s", selected_features.columns.tolist())
    return selected_features
