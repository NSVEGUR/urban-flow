"""
UrbanFlow – Baseline Models
==============================
Naive Seasonal, Auto-ARIMA, and XGBoost baselines for benchmark comparison.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.config import (
    ALL_FEATURES,
    FORECAST_HORIZON,
    TARGET_COL,
)
from app.evaluation import compute_all_metrics

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Naive Seasonal Baseline
# ──────────────────────────────────────────────

class NaiveSeasonalBaseline:
    """Predict using the value from same hour last week (lag 168)."""

    def __init__(self, seasonal_period: int = 168):
        self.seasonal_period = seasonal_period

    def predict(self, series: np.ndarray) -> np.ndarray:
        """Generate predictions by shifting the series by *seasonal_period*.

        Returns an array aligned to the *last* portion of *series* (after the
        initial seasonal_period observations are consumed).
        """
        preds = series[:-self.seasonal_period]
        return preds

    def evaluate(
        self, train: np.ndarray, test: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Evaluate on test set.

        We take the last ``seasonal_period`` values from train + all of test,
        and forecast each test point as the value ``seasonal_period`` steps back.
        """
        combined = np.concatenate([train[-self.seasonal_period:], test])
        preds = combined[:len(test)]  # values from seasonal_period back
        metrics = compute_all_metrics(test, preds)
        logger.info("Naive Seasonal  → %s", metrics)
        return preds, metrics


# ──────────────────────────────────────────────
# ARIMA Baseline
# ──────────────────────────────────────────────

class ARIMABaseline:

    def __init__(self):
        from statsmodels.tsa.arima.model import ARIMA
        self.ARIMA = ARIMA
        self.model = None
        self.results = None

    def fit(self, train: np.ndarray):
        logger.info("Fitting ARIMA(order=(2,1,2))")

        self.model = self.ARIMA(
            train,
            order=(2, 1, 2)
        )

        self.results = self.model.fit()
        return self

    def predict(self, n_periods: int):
        return self.results.forecast(steps=n_periods)

    def evaluate(self, train, test):
        self.fit(train)
        preds = self.predict(len(test))
        metrics = compute_all_metrics(test, preds)
        logger.info("ARIMA  → %s", metrics)
        return preds, metrics



