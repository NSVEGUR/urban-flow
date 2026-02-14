"""
UrbanFlow – Univariate GRU
============================
Single-junction GRU with residual connections, layer normalization,
AdamW optimizer, and OneCycleLR scheduler.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from app.config import (
    DEVICE,
    DROPOUT,
    EPOCHS,
    FORECAST_HORIZON,
    GRU_HIDDEN_SIZE,
    GRU_NUM_LAYERS,
    LEARNING_RATE,
    MODELS_DIR,
    PATIENCE,
    WEIGHT_DECAY,
)
from app.evaluation import compute_all_metrics
from app.utils import count_parameters

logger = logging.getLogger(__name__)


class UniGRU(nn.Module):
    """Multi-feature, single-junction GRU with residual connection.

    Architecture
    ------------
    Input  → GRU (2 layers, bidirectional=False)
           → LayerNorm
           → Dropout
           → FC (hidden → horizon)

    A residual connection adds the mean of the last ``horizon`` input steps
    to the output (when input_size == 1, i.e. the target channel).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = GRU_HIDDEN_SIZE,
        num_layers: int = GRU_NUM_LAYERS,
        dropout: float = DROPOUT,
        horizon: int = FORECAST_HORIZON,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        Tensor of shape (batch, horizon)
        """
        # GRU pass
        out, _ = self.gru(x)            # (batch, seq_len, hidden)
        out = out[:, -1, :]              # last time-step → (batch, hidden)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)               # (batch, horizon)
        return out


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────

def train_uni_gru(
    model: UniGRU,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    device: torch.device = DEVICE,
    checkpoint_name: str = "uni_gru",
) -> Dict[str, list]:
    """Train the UniGRU model with early stopping.

    Returns
    -------
    dict with 'train_loss' and 'val_loss' history lists.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr * 10,
        steps_per_epoch=len(train_dl), epochs=epochs,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    logger.info(
        "Training UniGRU  |  params=%s  |  device=%s",
        f"{count_parameters(model):,}", device,
    )

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d/%d  │  train_loss=%.6f  val_loss=%.6f",
                epoch, epochs, avg_train, avg_val,
            )

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            ckpt_path = MODELS_DIR / f"{checkpoint_name}.pt"
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (best val_loss=%.6f)", epoch, best_val_loss)
                break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f"{checkpoint_name}.pt", weights_only=True))
    model.eval()
    return history


# ──────────────────────────────────────────────
# Evaluation Helper
# ──────────────────────────────────────────────

def evaluate_uni_gru(
    model: UniGRU,
    test_dl: DataLoader,
    pipeline,
    junction_id: int,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run inference on test set and compute metrics.

    Returns (actuals, predictions, metrics_dict).
    """
    model = model.to(device)
    model.eval()

    all_preds, all_actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_actuals.append(y_batch.numpy())

    preds_flat = np.concatenate(all_preds).ravel()
    actuals_flat = np.concatenate(all_actuals).ravel()

    preds_inv = pipeline.inverse_transform_target(preds_flat, junction_id)
    actuals_inv = pipeline.inverse_transform_target(actuals_flat, junction_id)

    metrics = compute_all_metrics(actuals_inv, preds_inv)
    logger.info("UniGRU Test  → %s", metrics)
    return actuals_flat, preds_flat, metrics
