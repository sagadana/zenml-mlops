"""
tests/unit/test_encoders.py

Unit tests for the build_encoders step.
Validates encoder correctness, determinism, and roundtrip (encode → decode).
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest


def make_ratings_ddf(n_users=100, n_items=200, n_ratings=1000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "userId": rng.integers(1, n_users + 1, n_ratings, dtype="int32"),
        "movieId": rng.integers(1, n_items + 1, n_ratings, dtype="int32"),
        "rating": rng.uniform(0.5, 5.0, n_ratings).astype("float32"),
        "timestamp": rng.integers(1_000_000, 2_000_000, n_ratings, dtype="int64"),
    })
    return dd.from_pandas(df, npartitions=2)


def test_encoder_covers_all_users():
    from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders

    ddf = make_ratings_ddf()
    user_enc, item_enc = build_encoders.entrypoint(raw_ratings=ddf)

    unique_users = ddf["userId"].unique().compute().tolist()
    assert set(unique_users) == set(user_enc.index.tolist())


def test_encoder_covers_all_items():
    from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders

    ddf = make_ratings_ddf()
    user_enc, item_enc = build_encoders.entrypoint(raw_ratings=ddf)

    unique_items = ddf["movieId"].unique().compute().tolist()
    assert set(unique_items) == set(item_enc.index.tolist())


def test_encoder_dense_consecutive():
    """Dense indices must be 0-based and consecutive."""
    from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders

    ddf = make_ratings_ddf()
    user_enc, item_enc = build_encoders.entrypoint(raw_ratings=ddf)

    assert sorted(user_enc.values.tolist()) == list(range(len(user_enc)))
    assert sorted(item_enc.values.tolist()) == list(range(len(item_enc)))


def test_encoder_deterministic():
    """Same input should always produce the same encoder."""
    from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders

    ddf = make_ratings_ddf()
    enc1_user, _ = build_encoders.entrypoint(raw_ratings=ddf)
    enc2_user, _ = build_encoders.entrypoint(raw_ratings=ddf)

    pd.testing.assert_series_equal(enc1_user, enc2_user)


def test_encoder_roundtrip():
    """Encode then decode should recover original IDs."""
    from workflows.matrix_factorization.steps.feature_engineering.encoders import build_encoders

    ddf = make_ratings_ddf()
    user_enc, item_enc = build_encoders.entrypoint(raw_ratings=ddf)

    # Build decoder (dense_idx → raw_id)
    user_dec = pd.Series(user_enc.index, index=user_enc.values)
    item_dec = pd.Series(item_enc.index, index=item_enc.values)

    # Encode a sample of raw IDs, then decode back
    sample_users = list(user_enc.index[:10])
    for raw_uid in sample_users:
        dense = user_enc[raw_uid]
        recovered = user_dec[dense]
        assert recovered == raw_uid, f"Roundtrip failed for user {raw_uid}: {dense} → {recovered}"
