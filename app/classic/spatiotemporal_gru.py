"""
UrbanFlow – Spatio-Temporal GRU
=================================
Cross-junction GRU that treats all junctions as input channels to capture
inter-junction dependencies.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

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
    JUNCTION_IDS,
    LEARNING_RATE,
    MODELS_DIR,
    PATIENCE,
    WEIGHT_DECAY,
)
from app.evaluation import compute_all_metrics
from app.utils import count_parameters

logger = logging.getLogger(__name__)


class SpatioTemporalGRU(nn.Module):
    """GRU that ingests all junctions simultaneously.

    Input shape: (batch, seq_len, n_junctions, n_features)
    The junction and feature dimensions are flattened → (batch, seq_len, n_junctions * n_features)
    before being fed to the GRU.

    Output shape: (batch, horizon, n_junctions) — one prediction per junction.

    Architecture
    ------------
    Flatten(junctions × features) → GRU → LayerNorm → Dropout → FC → Reshape
    """

    def __init__(
        self,
        n_junctions: int = len(JUNCTION_IDS),
        n_features: int = 1,  # set at runtime
        hidden_size: int = GRU_HIDDEN_SIZE,
        num_layers: int = GRU_NUM_LAYERS,
        dropout: float = DROPOUT,
        horizon: int = FORECAST_HORIZON,
    ):
        super().__init__()
        self.n_junctions = n_junctions
        self.horizon = horizon
        input_dim = n_junctions * n_features

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon * n_junctions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_junctions, n_features)

        Returns
        -------
        (batch, horizon, n_junctions)
        """
        batch, seq_len, n_j, n_f = x.shape
        x = x.view(batch, seq_len, n_j * n_f)   # flatten spatial + features

        out, _ = self.gru(x)                      # (batch, seq_len, hidden)
        out = out[:, -1, :]                       # last step → (batch, hidden)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)                        # (batch, horizon * n_junctions)
        out = out.view(batch, self.horizon, self.n_junctions)
        return out


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────

def train_spatiotemporal_gru(
    model: SpatioTemporalGRU,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    device: torch.device = DEVICE,
    checkpoint_name: str = "spatiotemporal_gru",
) -> Dict[str, list]:
    """Train the spatio-temporal GRU with early stopping.

    Returns dict with 'train_loss' and 'val_loss' history.
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
        "Training SpatioTemporalGRU  |  params=%s  |  device=%s",
        f"{count_parameters(model):,}", device,
    )

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)        # (batch, horizon, n_junctions)
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
                logger.info("Early stopping at epoch %d (best val=%.6f)", epoch, best_val_loss)
                break

    model.load_state_dict(torch.load(MODELS_DIR / f"{checkpoint_name}.pt", weights_only=True))
    model.eval()
    return history


# ──────────────────────────────────────────────
# Evaluation Helper
# ──────────────────────────────────────────────

def evaluate_spatiotemporal_gru(
    model: SpatioTemporalGRU,
    test_dl: DataLoader,
    pipeline,
    junction_ids: list = JUNCTION_IDS,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run inference and compute *aggregate* metrics across all junctions (inverse-transformed).

    Returns (actuals, predictions, metrics_dict).
    Shapes: actuals and predictions are flattened across horizon×junctions.
    """
    model = model.to(device)
    model.eval()

    all_preds, all_actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()   # (batch, horizon, n_j)
            all_preds.append(preds)
            all_actuals.append(y_batch.numpy())

    preds_all = np.concatenate(all_preds)       # (N, horizon, n_junctions)
    actuals_all = np.concatenate(all_actuals)

    # Inverse transform per junction
    final_preds = []
    final_actuals = []

    for j_idx, jid in enumerate(junction_ids):
        # Extract for this junction
        p = preds_all[:, :, j_idx].ravel()
        a = actuals_all[:, :, j_idx].ravel()
        
        # Inverse transform
        p_inv = pipeline.inverse_transform_target(p, jid)
        a_inv = pipeline.inverse_transform_target(a, jid)
        
        final_preds.append(p_inv)
        final_actuals.append(a_inv)

    # Concatenate all junctions back into one long array for aggregate metrics
    preds_flat = np.concatenate(final_preds)
    actuals_flat = np.concatenate(final_actuals)

    metrics = compute_all_metrics(actuals_flat, preds_flat)
    logger.info("SpatioTemporalGRU Test (Scaled Back) → %s", metrics)
    return actuals_flat, preds_flat, metrics


def evaluate_per_junction(
    model: SpatioTemporalGRU,
    test_dl: DataLoader,
    pipeline,
    junction_ids: list = JUNCTION_IDS,
    device: torch.device = DEVICE,
) -> Dict[int, Dict[str, float]]:
    """Per-junction metrics from the spatio-temporal model.

    Returns ``{junction_id: {RMSE: ..., MAE: ..., MAPE: ...}}``.
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

    preds_all = np.concatenate(all_preds)       # (N, horizon, n_junctions)
    actuals_all = np.concatenate(all_actuals)

    results = {}
    for j_idx, jid in enumerate(junction_ids):
        p = preds_all[:, :, j_idx].ravel()
        a = actuals_all[:, :, j_idx].ravel()
        p_inv = pipeline.inverse_transform_target(p, jid)
        a_inv = pipeline.inverse_transform_target(a, jid)
        results[jid] = compute_all_metrics(a_inv, p_inv)
        logger.info("  Junction %d  → %s", jid, results[jid])

    return results
