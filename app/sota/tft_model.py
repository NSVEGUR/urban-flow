"""
UrbanFlow – Temporal Fusion Transformer (SOTA)
=================================================
TFT model training and evaluation using ``pytorch-forecasting``.

Uses ``TemporalFusionTransformer`` with quantile loss for probabilistic outputs,
learnable variable selection, and interpretable attention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from  app.config import (
    ALL_FEATURES,
    BATCH_SIZE,
    DATETIME_COL,
    DEVICE,
    EPOCHS,
    FORECAST_HORIZON,
    JUNCTION_COL,
    JUNCTION_IDS,
    MODELS_DIR,
    PATIENCE,
    SEQ_LEN,
    TARGET_COL,
    TFT_ATTENTION_HEAD_SIZE,
    TFT_DROPOUT,
    TFT_HIDDEN_SIZE,
    TFT_LEARNING_RATE,
    TFT_QUANTILES,
)
from  app.evaluation import compute_all_metrics

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Preparation for pytorch-forecasting
# ──────────────────────────────────────────────

def prepare_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame in the format required by ``pytorch-forecasting``.

    Adds:
    - ``time_idx``: integer time index per junction (required).
    - ``junction_str``: junction as string (for group normalization).
    """
    df = df.copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values([JUNCTION_COL, DATETIME_COL]).reset_index(drop=True)

    # Create integer time index per junction
    df["time_idx"] = df.groupby(JUNCTION_COL).cumcount()

    # String version of junction for pytorch-forecasting categorical
    df["junction_str"] = df[JUNCTION_COL].astype(str)

    return df


def build_tft_datasets(
    df: pd.DataFrame,
    max_encoder_length: int = SEQ_LEN,
    max_prediction_length: int = FORECAST_HORIZON,
):
    """Build ``TimeSeriesDataSet`` for train and validation.

    Returns (training_dataset, validation_dataset, training_dataloader, val_dataloader).
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    # Chronological split using time_idx
    max_time = df["time_idx"].max()
    train_cutoff = int(max_time * 0.70)
    val_cutoff = int(max_time * 0.85)

    train_df = df[df["time_idx"] <= train_cutoff]
    val_df = df[(df["time_idx"] > train_cutoff - max_encoder_length) & (df["time_idx"] <= val_cutoff)]
    test_df = df[df["time_idx"] > val_cutoff - max_encoder_length]

    # Known time-varying features the model knows in the future
    time_varying_known = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend",
    ]

    # Observed features only available up to present
    time_varying_unknown = [
        TARGET_COL,
        "lag_1", "lag_24", "lag_168",
        "rolling_mean_24", "rolling_std_24",
    ]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET_COL,
        group_ids=["junction_str"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known,
        time_varying_unknown_reals=time_varying_unknown,
        target_normalizer=GroupNormalizer(groups=["junction_str"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=True, stop_randomization=True,
    )

    testing = TimeSeriesDataSet.from_dataset(
        training, test_df, predict=True, stop_randomization=True,
    )

    train_dl = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=9,
    )
    val_dl = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=9,
    )
    test_dl = testing.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=9,
    )

    return training, validation, testing, train_dl, val_dl, test_dl


# ──────────────────────────────────────────────
# TFT Model
# ──────────────────────────────────────────────

def build_tft(training_dataset):
    """Instantiate a ``TemporalFusionTransformer`` from a training dataset."""
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=TFT_LEARNING_RATE,
        hidden_size=TFT_HIDDEN_SIZE,
        attention_head_size=TFT_ATTENTION_HEAD_SIZE,
        dropout=TFT_DROPOUT,
        hidden_continuous_size=TFT_HIDDEN_SIZE // 2,
        output_size=len(TFT_QUANTILES),
        loss=QuantileLoss(quantiles=TFT_QUANTILES),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    logger.info("TFT model built  |  params=%s", f"{sum(p.numel() for p in model.parameters()):,}")
    return model


def train_tft(
    model,
    train_dl,
    val_dl,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
):
    """Train TFT using PyTorch Lightning Trainer.

    Returns the trained model and trainer.
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=patience,
        verbose=True,
        mode="min",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(MODELS_DIR),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    logger.info("Starting TFT training …")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Load best checkpoint
    best_model_path = checkpoint_cb.best_model_path
    if best_model_path:
        from pytorch_forecasting import TemporalFusionTransformer
        model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        logger.info("Loaded best TFT checkpoint: %s", best_model_path)

    return model, trainer


# ──────────────────────────────────────────────
# Evaluation & Interpretation
# ──────────────────────────────────────────────

def evaluate_tft(
    model,
    test_dl,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate TFT on the test set.

    Returns (actuals, median_predictions, metrics_dict).
    """
    predictions = model.predict(test_dl, return_y=True)

    # predictions.output is (N, horizon, n_quantiles) or (N, horizon) for median
    # predictions.y is (N, horizon) tuple
    actuals = predictions.y[0].cpu().numpy().ravel() if isinstance(predictions.y, tuple) else predictions.y.cpu().numpy().ravel()

    # Median is the middle quantile index
    if predictions.output.ndim == 3:
        median_idx = len(TFT_QUANTILES) // 2
        preds = predictions.output[:, :, median_idx].cpu().numpy().ravel()
    else:
        preds = predictions.output.cpu().numpy().ravel()

    metrics = compute_all_metrics(actuals, preds)
    logger.info("TFT Test  → %s", metrics)
    return actuals, preds, metrics


def get_attention_weights(model, test_dl) -> Dict:
    """Extract attention weights and variable importance from TFT.

    Returns a dict with:
    - 'attention': average temporal attention weights
    - 'variable_importance': encoder variable importance dict
    """
    interpretation = model.interpret_output(
        model.predict(test_dl, mode="raw", return_x=True),
        reduction="mean",
    )

    result = {}

    if "attention" in interpretation:
        result["attention"] = interpretation["attention"].cpu().numpy()

    # Variable importance
    try:
        importance_raw = model.interpret_output(
            model.predict(test_dl, mode="raw", return_x=True),
            reduction="mean",
        )
        if "encoder_variables" in importance_raw:
            result["variable_importance"] = {
                name: float(val)
                for name, val in zip(
                    model.encoder_variables,
                    importance_raw["encoder_variables"].cpu().numpy(),
                )
            }
    except Exception as e:
        logger.warning("Could not extract variable importance: %s", e)

    return result


def get_quantile_predictions(
    model,
    test_dl,
) -> Dict[float, np.ndarray]:
    """Extract per-quantile predictions from TFT.

    Returns ``{quantile_value: predictions_array}``.
    """
    raw_predictions = model.predict(test_dl, mode="prediction", return_x=False)

    quantile_preds = {}
    if raw_predictions.ndim == 3:
        for i, q in enumerate(TFT_QUANTILES):
            quantile_preds[q] = raw_predictions[:, :, i].cpu().numpy().ravel()
    else:
        quantile_preds[0.5] = raw_predictions.cpu().numpy().ravel()

    return quantile_preds
