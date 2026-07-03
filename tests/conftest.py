from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import ALL_FEATURES, DATETIME_COL, JUNCTION_COL, TARGET_COL


@pytest.fixture
def raw_traffic_df() -> pd.DataFrame:
    """Small synthetic raw traffic dataframe: 2 junctions x 400 hourly rows.

    400 hours comfortably exceeds the largest lag (168h) so lag/rolling
    features are non-trivial, while staying fast to process in tests.
    """
    rng = np.random.default_rng(42)
    hours = pd.date_range("2024-01-01", periods=400, freq="h")

    frames = []
    for junction in (1, 2):
        base = 20 + junction * 5
        vehicles = base + 10 * np.sin(np.arange(400) * 2 * np.pi / 24) + rng.normal(0, 2, 400)
        frames.append(
            pd.DataFrame(
                {
                    DATETIME_COL: hours,
                    JUNCTION_COL: junction,
                    TARGET_COL: np.clip(vehicles, 0, None).round().astype(int),
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values([DATETIME_COL, JUNCTION_COL])
        .reset_index(drop=True)
    )


@pytest.fixture
def feature_df(raw_traffic_df: pd.DataFrame) -> pd.DataFrame:
    """raw_traffic_df run through the real feature-engineering pipeline."""
    from app.data_pipeline import (
        _add_cyclical_time_features,
        _add_lag_features,
        _add_rolling_features,
    )

    df = raw_traffic_df.copy()
    df = _add_cyclical_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    return df.dropna().reset_index(drop=True)


@pytest.fixture
def scaled_splits(feature_df: pd.DataFrame):
    """(train, val, test) dataframes for junction 1, scaled like production."""
    from app.data_pipeline import chronological_split, fit_scalers, scale_dataframe

    train, val, test = chronological_split(feature_df, junction_id=1)
    feat_scaler, tgt_scaler = fit_scalers(train, ALL_FEATURES)
    train = scale_dataframe(train, feat_scaler, tgt_scaler, ALL_FEATURES)
    val = scale_dataframe(val, feat_scaler, tgt_scaler, ALL_FEATURES)
    test = scale_dataframe(test, feat_scaler, tgt_scaler, ALL_FEATURES)
    return train, val, test, tgt_scaler
