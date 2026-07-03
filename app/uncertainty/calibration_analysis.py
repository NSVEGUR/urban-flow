"""
UrbanFlow – Calibration Analysis
====================================
Builds a reliability diagram comparing empirical vs. nominal coverage across
several confidence levels for all three uncertainty-quantification methods
(XGBoost Quantile, MC Dropout GRU, Quantile TFT). Reuses already-trained
checkpoints where available rather than retraining GRUs from scratch.

Usage::

    uv run app/uncertainty/calibration_analysis.py

Requires Pipeline 3 (uncertainty/main.py) to have been run first, so that
MC Dropout GRU checkpoints and a TFT checkpoint exist in app/models/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ALL_FEATURES, DEVICE, JUNCTION_IDS, MC_SAMPLES, MODELS_DIR
from app.config import UNCERTAINTY_RESULTS_DIR as RESULTS_DIR
from app.data_pipeline import TrafficDataPipeline, load_and_engineer_features
from app.uncertainty.mc_dropout_gru import MCDropoutGRU
from app.uncertainty.quantile_xgboost import XGBoostQuantile
from app.utils import seed_everything, setup_logging, timer
from app.visualization import plot_calibration_diagram

logger = logging.getLogger(__name__)

CONFIDENCE_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]


def xgboost_reliability(pipeline: TrafficDataPipeline) -> tuple[list[float], list[float]]:
    """Empirical coverage of XGBoost Quantile at several nominal levels,
    averaged across junctions. Coverage is scale-invariant, so this is
    computed directly from XGBoostQuantile's own (already inverse-scaled)
    calibration output.
    """
    coverages = []
    for level in CONFIDENCE_LEVELS:
        junction_coverages = []
        for jid in JUNCTION_IDS:
            train_df, val_df, test_df = pipeline.get_junction_dataframes(jid)
            model = XGBoostQuantile(confidence_level=level)
            result = model.evaluate(train_df, val_df, test_df, pipeline, jid)
            junction_coverages.append(result["calibration"]["empirical_coverage"])
        coverage = float(np.mean(junction_coverages))
        coverages.append(coverage)
        logger.info("XGBoost Quantile @ %.0f%% nominal → %.3f empirical", level * 100, coverage)
    return CONFIDENCE_LEVELS, coverages


def mc_dropout_reliability(pipeline: TrafficDataPipeline) -> tuple[list[float], list[float]]:
    """Empirical coverage of MC Dropout GRU at several nominal levels, reusing
    already-trained per-junction checkpoints. Coverage is scale-invariant, so
    inverse-transforming back to vehicle counts is unnecessary here.
    """
    models = {}
    for jid in JUNCTION_IDS:
        model = MCDropoutGRU(input_size=len(ALL_FEATURES))
        ckpt_path = MODELS_DIR / f"mc_dropout_gru_j{jid}.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        models[jid] = model.to(DEVICE)

    coverages = []
    for level in CONFIDENCE_LEVELS:
        junction_coverages = []
        for jid in JUNCTION_IDS:
            _, _, test_dl = pipeline.get_junction_dataloaders(jid)
            model = models[jid]
            all_actuals, all_lowers, all_uppers = [], [], []
            for x_batch, y_batch in test_dl:
                x_batch = x_batch.to(DEVICE)
                _, _, lower, upper = model.predict_with_uncertainty(
                    x_batch, n_samples=MC_SAMPLES, confidence=level
                )
                all_lowers.append(lower)
                all_uppers.append(upper)
                all_actuals.append(y_batch.numpy())
            actuals = np.concatenate(all_actuals).ravel()
            lowers = np.concatenate(all_lowers).ravel()
            uppers = np.concatenate(all_uppers).ravel()
            inside = (actuals >= lowers) & (actuals <= uppers)
            junction_coverages.append(float(np.mean(inside)))
        coverage = float(np.mean(junction_coverages))
        coverages.append(coverage)
        logger.info("MC Dropout GRU @ %.0f%% nominal → %.3f empirical", level * 100, coverage)
    return CONFIDENCE_LEVELS, coverages


def tft_reliability(pipeline: TrafficDataPipeline) -> tuple[list[float], list[float]] | None:
    """Empirical coverage of Quantile TFT at the two nominal levels its fixed
    quantile set supports: 50% (0.25/0.75) and 80% (0.1/0.9).
    """
    tft_checkpoints = list(MODELS_DIR.glob("tft-*.ckpt"))
    if not tft_checkpoints:
        logger.warning("No TFT checkpoint found — skipping TFT in reliability diagram.")
        return None

    from pytorch_forecasting import TemporalFusionTransformer

    from app.sota.tft_model import build_tft_datasets, prepare_tft_dataframe

    df = load_and_engineer_features(save=False)
    tft_df = prepare_tft_dataframe(df)
    _, _, _, _, _, test_dl = build_tft_datasets(tft_df)

    best_ckpt = sorted(tft_checkpoints)[-1]
    model = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))

    predictions = model.predict(test_dl, return_y=True, mode="quantiles")
    actuals = predictions.y[0].cpu().numpy().ravel()
    output = predictions.output  # (N, horizon, n_quantiles)

    from app.config import TFT_QUANTILES

    quantile_preds = {q: output[:, :, i].cpu().numpy().ravel() for i, q in enumerate(TFT_QUANTILES)}

    levels, coverages = [], []
    for nominal, (lo_q, hi_q) in [(0.5, (0.25, 0.75)), (0.8, (0.1, 0.9))]:
        lower, upper = quantile_preds[lo_q], quantile_preds[hi_q]
        inside = (actuals >= lower) & (actuals <= upper)
        coverage = float(np.mean(inside))
        levels.append(nominal)
        coverages.append(coverage)
        logger.info("Quantile TFT @ %.0f%% nominal → %.3f empirical", nominal * 100, coverage)
    return levels, coverages


def main() -> None:
    setup_logging()
    seed_everything(42)

    logger.info("Preparing data pipeline …")
    pipeline = TrafficDataPipeline()
    pipeline.prepare()

    reliability = {}

    with timer("XGBoost Quantile reliability"):
        reliability["XGBoost Quantile"] = xgboost_reliability(pipeline)

    with timer("MC Dropout GRU reliability"):
        try:
            reliability["MC Dropout GRU"] = mc_dropout_reliability(pipeline)
        except FileNotFoundError:
            logger.warning("No MC Dropout GRU checkpoints found — run uncertainty/main.py first.")

    with timer("Quantile TFT reliability"):
        tft_result = tft_reliability(pipeline)
        if tft_result is not None:
            reliability["Quantile TFT"] = tft_result

    plot_calibration_diagram(
        reliability,
        save_path=RESULTS_DIR / "calibration_reliability_diagram.png",
    )
    logger.info(
        "Saved reliability diagram → %s", RESULTS_DIR / "calibration_reliability_diagram.png"
    )


if __name__ == "__main__":
    main()
