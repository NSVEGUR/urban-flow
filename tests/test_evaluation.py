from __future__ import annotations

import numpy as np
import pytest

from app.evaluation import (
    build_comparison_table,
    calibration_score,
    compute_all_metrics,
    crps_gaussian,
    mae,
    mape,
    per_junction_report,
    rmse,
)


def test_rmse_known_value() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0, 5.0])
    assert rmse(actual, predicted) == pytest.approx(0.5)


def test_mae_known_value() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0, 5.0])
    assert mae(actual, predicted) == pytest.approx(0.25)


def test_mape_known_value() -> None:
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0, 18.0])
    assert mape(actual, predicted) == pytest.approx(15.0, abs=1e-3)


def test_compute_all_metrics_matches_individual_functions() -> None:
    actual = np.array([5.0, 10.0, 15.0])
    predicted = np.array([6.0, 9.0, 14.0])
    result = compute_all_metrics(actual, predicted)

    assert result["RMSE"] == pytest.approx(rmse(actual, predicted))
    assert result["MAE"] == pytest.approx(mae(actual, predicted))
    assert result["MAPE"] == pytest.approx(mape(actual, predicted))


def test_calibration_score_counts_coverage_correctly() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    upper = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    result = calibration_score(actual, lower, upper, nominal_coverage=0.9)

    assert result["empirical_coverage"] == pytest.approx(0.8)  # 4/5 inside
    assert result["interval_width"] == pytest.approx(5.0)
    assert result["miscalibration"] == pytest.approx(0.1, abs=1e-9)


def test_crps_gaussian_is_lower_for_a_better_calibrated_forecast() -> None:
    actual = np.full(50, 10.0)
    good_mu = np.full(50, 10.0)
    bad_mu = np.full(50, 20.0)
    sigma = np.full(50, 2.0)

    good_crps = crps_gaussian(actual, good_mu, sigma)
    bad_crps = crps_gaussian(actual, bad_mu, sigma)

    assert good_crps >= 0
    assert good_crps < bad_crps


def test_build_comparison_table_sorts_ascending() -> None:
    results = {
        "worst": {"RMSE": 10.0, "MAE": 8.0},
        "best": {"RMSE": 2.0, "MAE": 1.0},
        "middle": {"RMSE": 5.0, "MAE": 4.0},
    }
    table = build_comparison_table(results, sort_by="RMSE")
    assert list(table.index) == ["best", "middle", "worst"]


def test_per_junction_report_builds_expected_multiindex() -> None:
    junction_results = {
        1: {"XGBoost": {"RMSE": 5.0}},
        2: {"XGBoost": {"RMSE": 6.0}},
    }
    report = per_junction_report(junction_results)
    assert report.loc[(1, "XGBoost"), "RMSE"] == 5.0
    assert report.loc[(2, "XGBoost"), "RMSE"] == 6.0
