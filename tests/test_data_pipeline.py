from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import ALL_FEATURES, JUNCTION_COL, TARGET_COL


def test_cyclical_features_are_bounded_and_correct(feature_df: pd.DataFrame) -> None:
    assert feature_df["hour_sin"].between(-1, 1).all()
    assert feature_df["hour_cos"].between(-1, 1).all()

    midnight = feature_df[feature_df["DateTime"].dt.hour == 0].iloc[0]
    assert midnight["hour_sin"] == pytest.approx(0.0, abs=1e-9)
    assert midnight["hour_cos"] == pytest.approx(1.0, abs=1e-9)

    saturday = feature_df[feature_df["DateTime"].dt.dayofweek == 5]
    sunday = feature_df[feature_df["DateTime"].dt.dayofweek == 6]
    assert (saturday["is_weekend"] == 1).all()
    assert (sunday["is_weekend"] == 1).all()

    weekday = feature_df[feature_df["DateTime"].dt.dayofweek < 5]
    assert (weekday["is_weekend"] == 0).all()


def test_lag_features_do_not_cross_junction_boundary(raw_traffic_df: pd.DataFrame) -> None:
    from app.data_pipeline import _add_lag_features

    df = _add_lag_features(raw_traffic_df.copy())

    for junction, group in df.groupby(JUNCTION_COL):
        group = group.sort_values("DateTime")
        expected_lag_1 = group[TARGET_COL].shift(1)
        pd.testing.assert_series_equal(
            group["lag_1"].reset_index(drop=True),
            expected_lag_1.reset_index(drop=True),
            check_names=False,
        )
    # first row of each junction has no lag_1 yet (NaN), confirming no leakage
    # from the other junction's tail values
    first_rows = df.sort_values("DateTime").groupby(JUNCTION_COL).head(1)
    assert first_rows["lag_1"].isna().all()


def test_rolling_features_match_manual_computation(raw_traffic_df: pd.DataFrame) -> None:
    from app.data_pipeline import _add_rolling_features

    df = _add_rolling_features(raw_traffic_df.copy())

    for junction, group in df.groupby(JUNCTION_COL):
        group = group.sort_values("DateTime")
        expected_mean = group[TARGET_COL].rolling(24, min_periods=1).mean()
        pd.testing.assert_series_equal(
            group["rolling_mean_24"].reset_index(drop=True),
            expected_mean.reset_index(drop=True),
            check_names=False,
        )


def test_chronological_split_has_no_temporal_leakage(feature_df: pd.DataFrame) -> None:
    from app.data_pipeline import chronological_split

    train, val, test = chronological_split(feature_df, junction_id=1)

    assert len(train) + len(val) + len(test) == len(feature_df[feature_df[JUNCTION_COL] == 1])
    assert train["DateTime"].max() <= val["DateTime"].min()
    assert val["DateTime"].max() <= test["DateTime"].min()


def test_scale_dataframe_round_trips_via_inverse_transform(feature_df: pd.DataFrame) -> None:
    from app.data_pipeline import chronological_split, fit_scalers, scale_dataframe

    train, val, test = chronological_split(feature_df, junction_id=1)
    feat_scaler, tgt_scaler = fit_scalers(train, ALL_FEATURES)

    original_train_target = train[TARGET_COL].values.astype(float)
    scaled_train = scale_dataframe(train, feat_scaler, tgt_scaler, ALL_FEATURES)

    assert scaled_train[TARGET_COL].between(-1e-9, 1 + 1e-9).all()

    recovered = tgt_scaler.inverse_transform(scaled_train[[TARGET_COL]].values).ravel()
    np.testing.assert_allclose(recovered, original_train_target, atol=1e-6)


def test_scale_dataframe_fits_only_on_train(feature_df: pd.DataFrame) -> None:
    """Scaler bounds must come from train alone, not leak from val/test."""
    from app.data_pipeline import chronological_split, fit_scalers

    train, val, test = chronological_split(feature_df, junction_id=1)
    _, tgt_scaler = fit_scalers(train, ALL_FEATURES)

    assert tgt_scaler.data_min_[0] == pytest.approx(train[TARGET_COL].min())
    assert tgt_scaler.data_max_[0] == pytest.approx(train[TARGET_COL].max())


def test_load_and_engineer_features_on_real_dataset() -> None:
    """Guard the actual shipped dataset against schema/NaN regressions."""
    from app.data_pipeline import load_and_engineer_features

    df = load_and_engineer_features(save=False)

    assert not df.empty
    assert set(ALL_FEATURES).issubset(df.columns)
    assert df[ALL_FEATURES + [TARGET_COL]].isna().sum().sum() == 0


def test_validate_raw_schema_rejects_missing_columns() -> None:
    from app.data_pipeline import _validate_raw_schema

    bad_df = pd.DataFrame({"DateTime": ["2024-01-01"], "Junction": [1]})  # no Vehicles
    with pytest.raises(ValueError, match="missing required column"):
        _validate_raw_schema(bad_df)


def test_validate_raw_schema_rejects_empty_dataframe() -> None:
    from app.data_pipeline import _validate_raw_schema

    empty_df = pd.DataFrame(columns=["DateTime", "Junction", "Vehicles"])
    with pytest.raises(ValueError, match="empty"):
        _validate_raw_schema(empty_df)


def test_validate_raw_schema_accepts_well_formed_data(raw_traffic_df: pd.DataFrame) -> None:
    from app.data_pipeline import _validate_raw_schema

    _validate_raw_schema(raw_traffic_df)  # should not raise
