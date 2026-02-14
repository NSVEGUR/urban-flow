"""
UrbanFlow – Evaluation Metrics
================================
RMSE, MAE, MAPE, CRPS, calibration score, and comparison table builder.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Point-Forecast Metrics
# ──────────────────────────────────────────────

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (×100).  Uses epsilon to avoid div-by-zero."""
    return float(np.mean(np.abs((actual - predicted) / (actual + eps))) * 100)


def compute_all_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> Dict[str, float]:
    """Return a dict with RMSE, MAE, and MAPE."""
    return {
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "MAPE": mape(actual, predicted),
    }


# ──────────────────────────────────────────────
# Probabilistic Metrics
# ──────────────────────────────────────────────

def crps_gaussian(
    actual: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> float:
    """Continuous Ranked Probability Score assuming Gaussian predictive dist.

    Uses the closed-form solution:
        CRPS = σ [ z Φ(z) + φ(z) - 1/√π ]
    where z = (actual - mu) / sigma.

    Falls back to ``properscoring`` if available.
    """
    try:
        from properscoring import crps_gaussian as _crps
        return float(np.mean(_crps(actual, mu, sigma)))
    except ImportError:
        from scipy.stats import norm
        z = (actual - mu) / (sigma + 1e-8)
        score = sigma * (
            z * norm.cdf(z) + norm.pdf(z) - 1.0 / np.sqrt(np.pi)
        )
        return float(np.mean(score))


def calibration_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.90,
) -> Dict[str, float]:
    """Evaluate calibration of prediction intervals.

    Returns
    -------
    dict with keys:
        empirical_coverage : fraction of actuals inside [lower, upper]
        interval_width     : average width of the interval
        miscalibration     : |empirical - nominal|
    """
    inside = (actual >= lower) & (actual <= upper)
    empirical = float(np.mean(inside))
    width = float(np.mean(upper - lower))
    return {
        "empirical_coverage": empirical,
        "interval_width": width,
        "miscalibration": abs(empirical - nominal_coverage),
    }


# ──────────────────────────────────────────────
# Comparison Table
# ──────────────────────────────────────────────

def build_comparison_table(
    results: Dict[str, Dict[str, float]],
    sort_by: str = "RMSE",
) -> pd.DataFrame:
    """Build a comparison DataFrame from model results.

    Parameters
    ----------
    results : dict
        ``{model_name: {metric_name: value, ...}, ...}``
    sort_by : str
        Column to sort by (ascending).

    Returns
    -------
    pd.DataFrame  sorted by *sort_by*, with model names as the index.
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    if sort_by in df.columns:
        df = df.sort_values(sort_by)
    return df


def per_junction_report(
    junction_results: Dict[int, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """Build a multi-level report across junctions and models.

    Parameters
    ----------
    junction_results : dict
        ``{junction_id: {model_name: {metric: value}}}``

    Returns
    -------
    pd.DataFrame with MultiIndex (Junction, Model).
    """
    rows = []
    for jid, models in sorted(junction_results.items()):
        for model_name, metrics in models.items():
            row = {"Junction": jid, "Model": model_name, **metrics}
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index(["Junction", "Model"])
    return df


def print_comparison(table: pd.DataFrame) -> None:
    """Pretty-print a comparison table to stdout."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(table.to_string(float_format="{:.4f}".format))
    print("=" * 60 + "\n")
