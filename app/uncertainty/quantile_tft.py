"""
UrbanFlow – Quantile TFT Wrapper
====================================
Thin wrapper around the TFT model (Pipeline 2) for probabilistic evaluation.
Extracts quantile predictions and computes calibration metrics.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.config import CONFIDENCE_LEVEL, TFT_QUANTILES
from app.evaluation import calibration_score, compute_all_metrics, crps_from_interval

logger = logging.getLogger(__name__)


def evaluate_quantile_tft(
    model,
    test_dl,
    confidence: float = CONFIDENCE_LEVEL,
) -> Dict:
    """Evaluate TFT's quantile outputs as prediction intervals.

    Uses 10th and 90th percentile quantiles for the 90% CI.

    Returns a dict with:
    - actuals, median, lower, upper  (flattened arrays)
    - quantile_preds: {quantile: array} for all quantiles
    - point_metrics (RMSE, MAE, MAPE on median)
    - calibration (coverage, width, miscalibration)
    """
    # mode="quantiles" is required (not "prediction") to keep the quantile
    # dimension intact — "prediction" calls the loss's to_prediction(), which
    # for QuantileLoss collapses the output to the median only, silently
    # making lower == upper == median (0.00 coverage/width bug).
    predictions = model.predict(test_dl, return_y=True, mode="quantiles")

    # Extract actuals
    if isinstance(predictions.y, tuple):
        actuals = predictions.y[0].cpu().numpy().ravel()
    else:
        actuals = predictions.y.cpu().numpy().ravel()

    # Extract quantile predictions
    output = predictions.output  # (N, horizon, n_quantiles)

    quantile_preds = {}
    if output.ndim == 3:
        for i, q in enumerate(TFT_QUANTILES):
            quantile_preds[q] = output[:, :, i].cpu().numpy().ravel()
    else:
        # Fallback: single output = point forecast
        quantile_preds[0.5] = output.cpu().numpy().ravel()

    # Median prediction
    median = quantile_preds.get(0.5, quantile_preds[TFT_QUANTILES[len(TFT_QUANTILES) // 2]])

    # Confidence interval (use 0.1 and 0.9 for 90% CI)
    alpha = (1 - confidence) / 2
    lower_q = min(TFT_QUANTILES, key=lambda q: abs(q - alpha))
    upper_q = min(TFT_QUANTILES, key=lambda q: abs(q - (1 - alpha)))

    lower = quantile_preds.get(lower_q, median)
    upper = quantile_preds.get(upper_q, median)

    # Metrics
    point_metrics = compute_all_metrics(actuals, median)
    cal = calibration_score(actuals, lower, upper, nominal_coverage=confidence)
    # TFT_QUANTILES only has 0.1/0.9 as its outermost pair, so the true
    # nominal coverage of (lower, upper) is (upper_q - lower_q), not
    # necessarily the requested `confidence` — use the real bracket width.
    crps = crps_from_interval(actuals, median, lower, upper, confidence=upper_q - lower_q)

    logger.info("Quantile TFT  → point: %s", point_metrics)
    logger.info("Quantile TFT  → calibration: %s", cal)
    logger.info("Quantile TFT  → CRPS: %.4f", crps)

    return {
        "actuals": actuals,
        "median": median,
        "lower": lower,
        "upper": upper,
        "quantile_preds": quantile_preds,
        "point_metrics": point_metrics,
        "calibration": cal,
        "crps": crps,
    }
