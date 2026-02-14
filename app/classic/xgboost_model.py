"""
UrbanFlow – XGBoost Model
=========================
Identified as the strongest baseline for our traffic data.
Extracted into its own module for priority.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.config import (
    ALL_FEATURES,
    TARGET_COL,
)
from app.evaluation import compute_all_metrics

logger = logging.getLogger(__name__)


class XGBoostBaseline:
    """XGBoost on tabular features (no sequence structure)."""

    def __init__(self, **kwargs):
        defaults = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=30,
            verbosity=0,
        )
        defaults.update(kwargs)
        self.model = XGBRegressor(**defaults)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols=None,
    ) -> "XGBoostBaseline":
        feature_cols = feature_cols or ALL_FEATURES
        self.feature_cols = feature_cols

        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_COL].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET_COL].values

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("XGBoost fitted (best iteration: %s)", self.model.best_iteration)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.feature_cols].values)

    def evaluate(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pipeline,
        junction_id: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        self.fit(train_df, val_df)
        preds = self.predict(test_df)
        preds = pipeline.inverse_transform_target(preds, junction_id)
        actual = pipeline.inverse_transform_target(
            test_df[TARGET_COL].values,
            junction_id
        )
        metrics = compute_all_metrics(actual, preds)
        logger.info("  Junction %d [XGBoost] → %s", junction_id, metrics)
        return preds, metrics
