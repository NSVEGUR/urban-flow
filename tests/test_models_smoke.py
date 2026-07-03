"""Smoke tests: each model family should train/predict without error on tiny
synthetic data and produce finite, sane outputs. Not meant to test forecast
quality — only that the pipeline runs end-to-end and companies can trust it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from torch.utils.data import DataLoader

from app.config import ALL_FEATURES, TARGET_COL
from app.data_pipeline import TrafficSequenceDataset


class _StubPipeline:
    """Minimal stand-in for TrafficDataPipeline exposing only what
    evaluate_* helpers need: inverse_transform_target() and .scalers.
    """

    def __init__(self, tgt_scaler, junction_id: int = 1):
        self.scalers = {junction_id: (None, tgt_scaler)}

    def inverse_transform_target(self, values: np.ndarray, junction_id: int) -> np.ndarray:
        _, tgt_scaler = self.scalers[junction_id]
        return tgt_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def _assert_all_finite(metrics: dict) -> None:
    for key, value in metrics.items():
        assert math.isfinite(value), f"{key}={value} is not finite"


def test_naive_seasonal_baseline_smoke() -> None:
    from app.classic.baselines import NaiveSeasonalBaseline

    rng = np.random.default_rng(0)
    train = rng.normal(20, 3, 200)
    test = rng.normal(20, 3, 50)

    baseline = NaiveSeasonalBaseline(seasonal_period=24)
    preds, metrics = baseline.evaluate(train, test)

    assert len(preds) == len(test)
    _assert_all_finite(metrics)


def test_xgboost_baseline_smoke(scaled_splits) -> None:
    from app.classic.xgboost_model import XGBoostBaseline

    train, val, test, tgt_scaler = scaled_splits
    pipeline = _StubPipeline(tgt_scaler)

    model = XGBoostBaseline(n_estimators=20, early_stopping_rounds=5)
    preds, metrics = model.evaluate(train, val, test, pipeline=pipeline, junction_id=1)

    assert len(preds) == len(test)
    _assert_all_finite(metrics)


def test_xgboost_quantile_smoke(scaled_splits) -> None:
    from app.uncertainty.quantile_xgboost import XGBoostQuantile

    train, val, test, tgt_scaler = scaled_splits
    pipeline = _StubPipeline(tgt_scaler)

    model = XGBoostQuantile(confidence_level=0.9, n_estimators=20)
    result = model.evaluate(train, val, test, pipeline, junction_id=1)

    assert len(result["median"]) == len(test)
    assert (result["lower"] <= result["upper"]).all()
    _assert_all_finite(result["point_metrics"])
    assert 0.0 <= result["calibration"]["empirical_coverage"] <= 1.0


@pytest.fixture
def tiny_gru_loaders(scaled_splits):
    train, val, test, _ = scaled_splits
    seq_len, horizon = 4, 2

    def make_loader(df, shuffle):
        features = df[ALL_FEATURES].values
        targets = df[TARGET_COL].values
        ds = TrafficSequenceDataset(features, targets, seq_len=seq_len, horizon=horizon)
        return DataLoader(ds, batch_size=8, shuffle=shuffle)

    return make_loader(train, True), make_loader(val, False), make_loader(test, False), horizon


def test_uni_gru_train_and_evaluate_smoke(
    tiny_gru_loaders, scaled_splits, tmp_path, monkeypatch
) -> None:
    from app.classic import univariate_gru as ugru

    monkeypatch.setattr(ugru, "MODELS_DIR", tmp_path)

    train_dl, val_dl, test_dl, horizon = tiny_gru_loaders
    _, _, _, tgt_scaler = scaled_splits
    pipeline = _StubPipeline(tgt_scaler)

    model = ugru.UniGRU(input_size=len(ALL_FEATURES), hidden_size=8, num_layers=1, horizon=horizon)
    history = ugru.train_uni_gru(
        model, train_dl, val_dl, epochs=2, patience=2, checkpoint_name="test_uni_gru"
    )

    assert len(history["train_loss"]) == 2
    assert all(math.isfinite(v) for v in history["train_loss"])
    assert (tmp_path / "test_uni_gru.pt").exists()

    actuals, preds, metrics = ugru.evaluate_uni_gru(
        model, test_dl, pipeline=pipeline, junction_id=1
    )
    assert len(actuals) == len(preds)
    _assert_all_finite(metrics)


def test_mc_dropout_gru_uncertainty_smoke(
    tiny_gru_loaders, scaled_splits, tmp_path, monkeypatch
) -> None:
    from app.classic import univariate_gru as ugru
    from app.uncertainty import mc_dropout_gru as mcd

    monkeypatch.setattr(ugru, "MODELS_DIR", tmp_path)

    train_dl, val_dl, test_dl, horizon = tiny_gru_loaders
    _, _, _, tgt_scaler = scaled_splits
    pipeline = _StubPipeline(tgt_scaler)

    model = mcd.MCDropoutGRU(
        input_size=len(ALL_FEATURES), hidden_size=8, num_layers=1, horizon=horizon
    )
    mcd.train_mc_gru(model, train_dl, val_dl, epochs=1, patience=1, checkpoint_name="test_mc_gru")

    result = mcd.evaluate_mc_gru(model, test_dl, pipeline=pipeline, junction_id=1, n_samples=5)

    assert (result["lower"] <= result["upper"]).all()
    assert (result["std"] >= 0).all()
    _assert_all_finite(result["point_metrics"])
    assert 0.0 <= result["calibration"]["empirical_coverage"] <= 1.0
