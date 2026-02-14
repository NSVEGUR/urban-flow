"""
UrbanFlow – MC Dropout GRU
=============================
GRU with dropout kept active at inference time to approximate Bayesian
uncertainty via Monte Carlo dropout.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from  app.config import (
    CONFIDENCE_LEVEL,
    DEVICE,
    FORECAST_HORIZON,
    GRU_HIDDEN_SIZE,
    GRU_NUM_LAYERS,
    MC_DROPOUT,
    MC_SAMPLES,
    MODELS_DIR,
)
from  app.classic.univariate_gru import UniGRU, train_uni_gru
from  app.evaluation import calibration_score, compute_all_metrics

logger = logging.getLogger(__name__)


class MCDropoutGRU(UniGRU):
    """UniGRU subclass that keeps dropout active during inference.

    On ``predict_with_uncertainty``, performs *N* stochastic forward passes
    and returns mean, std, and configurable confidence intervals.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = GRU_HIDDEN_SIZE,
        num_layers: int = GRU_NUM_LAYERS,
        dropout: float = MC_DROPOUT,
        horizon: int = FORECAST_HORIZON,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
        )

    def enable_mc_dropout(self) -> None:
        """Force all dropout layers to stay in training mode."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = MC_SAMPLES,
        confidence: float = CONFIDENCE_LEVEL,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run *n_samples* stochastic forward passes.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, features)
        n_samples : number of MC forward passes
        confidence : e.g. 0.90 for 90% CI

        Returns
        -------
        (mean, std, lower, upper) – each of shape (batch, horizon)
        """
        self.eval()
        self.enable_mc_dropout()  # keep dropout on

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)  # (batch, horizon)
                samples.append(pred.cpu().numpy())

        samples = np.stack(samples, axis=0)   # (n_samples, batch, horizon)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)

        alpha = (1 - confidence) / 2
        lower = np.percentile(samples, alpha * 100, axis=0)
        upper = np.percentile(samples, (1 - alpha) * 100, axis=0)

        return mean, std, lower, upper


# ──────────────────────────────────────────────
# Training  (uses same loop as UniGRU)
# ──────────────────────────────────────────────

def train_mc_gru(
    model: MCDropoutGRU,
    train_dl: DataLoader,
    val_dl: DataLoader,
    **kwargs,
) -> Dict[str, list]:
    """Train MCDropoutGRU (delegates to ``train_uni_gru``)."""
    kwargs.setdefault("checkpoint_name", "mc_dropout_gru")
    return train_uni_gru(model, train_dl, val_dl, **kwargs)


# ──────────────────────────────────────────────
# Evaluation with Uncertainty
# ──────────────────────────────────────────────

def evaluate_mc_gru(
    model: MCDropoutGRU,
    test_dl: DataLoader,
    pipeline,             
    junction_id: int,
    n_samples: int = MC_SAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    device: torch.device = DEVICE,
) -> Dict:
    """Run MC Dropout inference on the test set.

    Returns a dict with:
    - actuals, mean, std, lower, upper  (flattened arrays)
    - point_metrics  (RMSE, MAE, MAPE on mean prediction)
    - calibration    (empirical coverage, interval width)
    """
    model = model.to(device)

    all_means, all_stds, all_lowers, all_uppers, all_actuals = [], [], [], [], []

    for X_batch, y_batch in test_dl:
        X_batch = X_batch.to(device)
        mean, std, lower, upper = model.predict_with_uncertainty(
            X_batch, n_samples=n_samples, confidence=confidence,
        )
        all_means.append(mean)
        all_stds.append(std)
        all_lowers.append(lower)
        all_uppers.append(upper)
        all_actuals.append(y_batch.numpy())

    concat = lambda lst: np.concatenate(lst).ravel()
    actuals = concat(all_actuals)
    means = concat(all_means)
    stds = concat(all_stds)
    lowers = concat(all_lowers)
    uppers = concat(all_uppers)

    actuals_inv = pipeline.inverse_transform_target(actuals, junction_id)
    means_inv = pipeline.inverse_transform_target(means, junction_id)
    lowers_inv = pipeline.inverse_transform_target(lowers, junction_id)
    uppers_inv = pipeline.inverse_transform_target(uppers, junction_id)

    _, tgt_scaler = pipeline.scalers[junction_id]
    scale_factor = tgt_scaler.data_range_[0]  # max - min
    stds_inv = stds * scale_factor

    point_metrics = compute_all_metrics(actuals_inv, means_inv)
    cal = calibration_score(actuals_inv, lowers_inv, uppers_inv, nominal_coverage=confidence)

    logger.info("MC Dropout GRU  → point: %s", point_metrics)
    logger.info("MC Dropout GRU  → calibration: %s", cal)

    return {
        "actuals": actuals_inv,
        "mean": means_inv,
        "std": stds_inv,
        "lower": lowers_inv,
        "upper": uppers_inv,
        "point_metrics": point_metrics,
        "calibration": cal,
    }
