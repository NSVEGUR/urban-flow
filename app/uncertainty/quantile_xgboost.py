"""
UrbanFlow – XGBoost Quantile Regression
=========================================
Implements probabilistic forecasting using multiple XGBoost regressors
trained with quantile loss objectives.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.config import (
    ALL_FEATURES,
    CONFIDENCE_LEVEL,
    TARGET_COL,
)
from app.evaluation import calibration_score, compute_all_metrics

logger = logging.getLogger(__name__)


class XGBoostQuantile:
    """Probabilistic XGBoost using Quantile Regression.

    Trains three separate models:
    - Lower quantile (e.g., 0.05)
    - Median (0.50)
    - Upper quantile (e.g., 0.95)
    """

    def __init__(
        self,
        confidence_level: float = CONFIDENCE_LEVEL,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        **kwargs,
    ):
        self.confidence_level = confidence_level
        self.alpha = (1 - confidence_level) / 2
        self.quantiles = [self.alpha, 0.5, 1 - self.alpha]
        
        common_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=30,
            verbosity=0,
            objective='reg:quantileerror',
            **kwargs,
        )

        self.models = {}
        for q in self.quantiles:
            # Create a localized copy of params for each quantile
            params = common_params.copy()
            # quantitative_alpha is the parameter for reg:quantileerror in standard XGBoost
            # but in the sklearn wrapper it's often passed via kwargs or set_params.
            # We will pass it in kwargs for safety.
            params['quantile_alpha'] = q
            self.models[q] = XGBRegressor(**params)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list = None,
    ) -> "XGBoostQuantile":
        feature_cols = feature_cols or ALL_FEATURES
        self.feature_cols = feature_cols
        
        # Check if features exist
        missing = [c for c in feature_cols if c not in train_df.columns]
        if missing:
            logger.error("Missing features for XGBoost: %s", missing)
            raise ValueError(f"Missing features: {missing}")

        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_COL].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET_COL].values

        for q, model in self.models.items():
            # logger.info("Training XGBoost Quantile: q=%.2f", q)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        
        return self

    def predict(self, df: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Return predictions for all quantiles."""
        X = df[self.feature_cols].values
        preds = {}
        for q, model in self.models.items():
            preds[q] = model.predict(X)
        return preds

    def evaluate(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pipeline,
        junction_id: int,
    ) -> Dict:
        """Evaluate point forecast and uncertainty."""
        self.fit(train_df, val_df)
        
        raw_preds = self.predict(test_df)
        
        # Inverse transform
        preds_inv = {}
        for q, p in raw_preds.items():
            preds_inv[q] = pipeline.inverse_transform_target(p, junction_id)
            
        actual = pipeline.inverse_transform_target(
            test_df[TARGET_COL].values,
            junction_id
        )

        # Point metrics (Median)
        median_pred = preds_inv[0.5]
        point_metrics = compute_all_metrics(actual, median_pred)

        # Calibration
        lower = preds_inv[self.quantiles[0]]
        upper = preds_inv[self.quantiles[-1]]
        
        cal = calibration_score(
            actual, lower, upper, 
            nominal_coverage=self.confidence_level
        )

        logger.info("  Junction %d [XGBoost Quantile] → Point: %s | Calibration: %s", 
                    junction_id, point_metrics, cal)

        return {
            "actuals": actual,
            "median": median_pred,
            "lower": lower,
            "upper": upper,
            "point_metrics": point_metrics,
            "calibration": cal,
        }
