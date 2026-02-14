"""
UrbanFlow – Data Pipeline
==========================
Loading, feature engineering, splitting, scaling, and PyTorch DataLoader creation
for the traffic forecasting dataset.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from app.config import (
    ALL_FEATURES,
    AUGMENTED_DATA_PATH,
    BATCH_SIZE,
    CYCLICAL_FEATURES,
    DATETIME_COL,
    FORECAST_HORIZON,
    JUNCTION_COL,
    JUNCTION_IDS,
    LAG_HOURS,
    RAW_DATA_PATH,
    ROLLING_WINDOWS,
    SEQ_LEN,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine encoded time features (hour, day-of-week, month)."""
    dt = df[DATETIME_COL]

    # Hour of day  (period = 24)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24) # type: ignore
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)  # type: ignore

    # Day of week  (period = 7)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)  # type: ignore
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)  # type: ignore

    # Month        (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)  # type: ignore
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)  # type: ignore

    # Weekend flag
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32)  # type: ignore

    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features per junction (t-1, t-24, t-168)."""
    for lag in LAG_HOURS:
        df[f"lag_{lag}"] = df.groupby(JUNCTION_COL)[TARGET_COL].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and std per junction."""
    for window in ROLLING_WINDOWS:
        grouped = df.groupby(JUNCTION_COL)[TARGET_COL]
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
    return df


def load_and_engineer_features(save: bool = True) -> pd.DataFrame:
    """Load raw CSV, apply feature engineering, optionally save augmented data.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe sorted by (DateTime, Junction) with NaN rows
        from lag creation dropped.
    """
    logger.info("Loading raw data from %s", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.sort_values([DATETIME_COL, JUNCTION_COL]).reset_index(drop=True)

    logger.info("Engineering features …")
    df = _add_cyclical_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)

    # Drop rows with NaN introduced by lags (first 168 hours per junction)
    df = df.dropna().reset_index(drop=True)

    if save:
        df.to_csv(AUGMENTED_DATA_PATH, index=False)
        logger.info("Saved augmented data → %s  (%d rows)", AUGMENTED_DATA_PATH, len(df))

    return df


# ──────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    junction_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train / val / test.

    If *junction_id* is given, filter to that junction first.
    """
    if junction_id is not None:
        df = df[df[JUNCTION_COL] == junction_id].copy()

    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        "Split (junction=%s): train=%d, val=%d, test=%d",
        junction_id or "all", len(train), len(val), len(test),
    )
    return train, val, test


# ──────────────────────────────────────────────
# Scaling
# ──────────────────────────────────────────────

def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit separate scalers for features and target on training data only.

    Returns (feature_scaler, target_scaler).
    """
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[feature_cols].values)

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[target_col]].values)

    return feature_scaler, target_scaler


def scale_dataframe(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Apply fitted scalers to a dataframe (in-place copy)."""
    df = df.copy()
    df[feature_cols] = feature_scaler.transform(df[feature_cols].values)
    df[target_col] = target_scaler.transform(df[[target_col]].values)
    return df


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────

class TrafficSequenceDataset(Dataset):
    """Sliding-window dataset that produces (X, y) sequences.

    Parameters
    ----------
    data : np.ndarray of shape (T, n_features)
        Scaled feature matrix (including target as last column or separate).
    targets : np.ndarray of shape (T,)
        Scaled target values.
    seq_len : int
        Number of past time-steps to include in each sample.
    horizon : int
        Number of future time-steps to predict.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        seq_len: int = SEQ_LEN,
        horizon: int = FORECAST_HORIZON,
    ):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]                     # (seq_len, n_features)
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.horizon]  # (horizon,)
        return x, y


class SpatioTemporalDataset(Dataset):
    """Dataset for the spatio-temporal model: all junctions stacked as channels.

    Each sample contains the sequences for *all* junctions at aligned time-steps.

    Parameters
    ----------
    junction_data : dict[int, np.ndarray]
        {junction_id: feature_matrix} – all must have the same length.
    junction_targets : dict[int, np.ndarray]
        {junction_id: target_array}
    seq_len, horizon : int
    """

    def __init__(
        self,
        junction_data: Dict[int, np.ndarray],
        junction_targets: Dict[int, np.ndarray],
        seq_len: int = SEQ_LEN,
        horizon: int = FORECAST_HORIZON,
    ):
        self.junctions = sorted(junction_data.keys())
        # Stack: (T, n_junctions, n_features) and (T, n_junctions)
        self.data = torch.FloatTensor(
            np.stack([junction_data[j] for j in self.junctions], axis=1)
        )
        self.targets = torch.FloatTensor(
            np.stack([junction_targets[j] for j in self.junctions], axis=1)
        )
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]                        # (seq_len, n_junctions, n_feat)
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.horizon]  # (horizon, n_junctions)
        return x, y


# ──────────────────────────────────────────────
# High-Level Pipeline
# ──────────────────────────────────────────────

class TrafficDataPipeline:
    """End-to-end pipeline: load → engineer → split → scale → DataLoaders.

    Usage::

        pipeline = TrafficDataPipeline()
        pipeline.prepare()

        # Per-junction DataLoaders (for univariate GRU)
        train_dl, val_dl, test_dl = pipeline.get_junction_dataloaders(junction_id=1)

        # Spatio-temporal DataLoaders (for cross-junction GRU)
        train_dl, val_dl, test_dl = pipeline.get_spatiotemporal_dataloaders()
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        seq_len: int = SEQ_LEN,
        horizon: int = FORECAST_HORIZON,
        batch_size: int = BATCH_SIZE,
    ):
        self.feature_cols = feature_cols or ALL_FEATURES
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size

        # Populated by prepare()
        self.df: Optional[pd.DataFrame] = None
        self.scalers: Dict[int, Tuple[MinMaxScaler, MinMaxScaler]] = {}
        self.splits: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    def prepare(self) -> None:
        """Run the full pipeline: load, engineer, split, scale."""
        self.df = load_and_engineer_features(save=True)

        for jid in JUNCTION_IDS:
            train, val, test = chronological_split(self.df, junction_id=jid)
            feat_scaler, tgt_scaler = fit_scalers(train, self.feature_cols)

            train = scale_dataframe(train, feat_scaler, tgt_scaler, self.feature_cols)
            val = scale_dataframe(val, feat_scaler, tgt_scaler, self.feature_cols)
            test = scale_dataframe(test, feat_scaler, tgt_scaler, self.feature_cols)

            self.scalers[jid] = (feat_scaler, tgt_scaler)
            self.splits[jid] = (train, val, test)

        logger.info("Pipeline prepared for %d junctions.", len(JUNCTION_IDS))

    # ── Per-junction loaders ──

    def _make_loader(
        self, df: pd.DataFrame, junction_id: int, shuffle: bool
    ) -> DataLoader:
        features = df[self.feature_cols].values
        targets = df[TARGET_COL].values
        ds = TrafficSequenceDataset(features, np.array(targets), self.seq_len, self.horizon)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def get_junction_dataloaders(
        self, junction_id: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_dl, val_dl, test_dl) for a single junction."""
        train, val, test = self.splits[junction_id]
        return (
            self._make_loader(train, junction_id, shuffle=True),
            self._make_loader(val, junction_id, shuffle=False),
            self._make_loader(test, junction_id, shuffle=False),
        )

    # ── Spatio-temporal loaders ──

    def get_spatiotemporal_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_dl, val_dl, test_dl) with all junctions stacked."""
        loaders = []
        for split_idx, shuffle in [(0, True), (1, False), (2, False)]:
            junction_data = {}
            junction_targets = {}
            # Find the minimum length across junctions for this split
            min_len = min(
                len(self.splits[jid][split_idx]) for jid in JUNCTION_IDS
            )
            for jid in JUNCTION_IDS:
                split_df = self.splits[jid][split_idx].iloc[:min_len]
                junction_data[jid] = split_df[self.feature_cols].values
                junction_targets[jid] = split_df[TARGET_COL].values

            ds = SpatioTemporalDataset(
                junction_data, junction_targets, self.seq_len, self.horizon
            )
            loaders.append(
                DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)
            )

        return tuple(loaders)  # type: ignore[return-value]

    # ── Raw data access for baselines ──

    def get_junction_arrays(
        self, junction_id: int, scaled: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_targets, val_targets, test_targets) as 1-D arrays.

        If *scaled=False*, returns in original scale.
        """
        arrays = []
        for split_df in self.splits[junction_id]:
            vals = split_df[TARGET_COL].values
            if not scaled:
                _, tgt_scaler = self.scalers[junction_id]
                vals = tgt_scaler.inverse_transform(np.array(vals).reshape(-1, 1)).ravel()
            arrays.append(vals)
        return tuple(arrays)  # type: ignore[return-value]

    def get_junction_dataframes(
        self, junction_id: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train_df, val_df, test_df) for a junction (scaled)."""
        return self.splits[junction_id]

    def inverse_transform_target(
        self, values: np.ndarray, junction_id: int
    ) -> np.ndarray:
        """Convert scaled target values back to original scale."""
        _, tgt_scaler = self.scalers[junction_id]
        return tgt_scaler.inverse_transform(values.reshape(-1, 1)).ravel()
