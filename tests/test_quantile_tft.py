"""Regression test for a real bug: evaluate_quantile_tft() called
model.predict(..., mode="prediction"), which for QuantileLoss models collapses
the quantile dimension to the median only (pytorch-forecasting's
to_prediction() vs. to_quantiles()). That silently made lower == upper ==
median, producing 0.00 coverage / 0.00 interval width. The fix uses
mode="quantiles" to keep the full quantile tensor.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from app.config import TFT_QUANTILES
from app.uncertainty.quantile_tft import evaluate_quantile_tft


def _fake_predictions(n: int = 20, horizon: int = 4):
    actual = torch.rand(n, horizon) * 50
    # Distinct, ordered quantile predictions per quantile index.
    output = torch.stack(
        [actual + (q - 0.5) * 20 for q in TFT_QUANTILES],
        dim=-1,
    )  # (n, horizon, n_quantiles)
    return SimpleNamespace(y=(actual, None), output=output)


def test_evaluate_quantile_tft_requests_quantiles_mode() -> None:
    model = MagicMock()
    model.predict.return_value = _fake_predictions()

    evaluate_quantile_tft(model, test_dl=MagicMock())

    _, kwargs = model.predict.call_args
    assert kwargs.get("mode") == "quantiles"


def test_evaluate_quantile_tft_produces_a_non_degenerate_interval() -> None:
    model = MagicMock()
    model.predict.return_value = _fake_predictions()

    result = evaluate_quantile_tft(model, test_dl=MagicMock())

    assert not np.allclose(result["lower"], result["upper"])
    assert (result["lower"] <= result["median"]).all()
    assert (result["median"] <= result["upper"]).all()
    assert result["calibration"]["interval_width"] > 0
