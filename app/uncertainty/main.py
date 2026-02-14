"""
UrbanFlow – Uncertainty Pipeline Orchestrator
=================================================
Trains MC Dropout GRU and evaluates Quantile TFT for probabilistic forecasting.
Compares both approaches on calibration and interval quality.

Usage::

    python -m  uncertainty.run_uncertainty
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from  app.config import (
    ALL_FEATURES,
    CONFIDENCE_LEVEL,
    DEVICE,
    JUNCTION_IDS,
    MC_SAMPLES,
    UNCERTAINTY_RESULTS_DIR as RESULTS_DIR,
    MODELS_DIR,
    TFT_QUANTILES,
)
from  app.data_pipeline import TrafficDataPipeline, load_and_engineer_features
from  app.evaluation import build_comparison_table, print_comparison
from  app.utils import seed_everything, setup_logging, timer
from  app.visualization import plot_fan_chart, plot_forecast

from  app.uncertainty.mc_dropout_gru import MCDropoutGRU, evaluate_mc_gru, train_mc_gru

logger = logging.getLogger(__name__)


def run_mc_dropout(pipeline: TrafficDataPipeline, train_model: bool = True) -> dict:
    """Train and evaluate MC Dropout GRU per junction."""
    all_results = {}

    for jid in JUNCTION_IDS:
        logger.info("═══ MC Dropout GRU – Junction %d ═══", jid)
        train_dl, val_dl, test_dl = pipeline.get_junction_dataloaders(jid)

        n_features = len(ALL_FEATURES)
        model = MCDropoutGRU(input_size=n_features)

        ckpt_path = MODELS_DIR / f"mc_dropout_gru_j{jid}.pt"

        if train_model or not ckpt_path.exists():
            with timer(f"MC Dropout Train – J{jid}"):
                history = train_mc_gru(
                    model, train_dl, val_dl,
                    checkpoint_name=f"mc_dropout_gru_j{jid}",
                )
        else:
            logger.info("Loading MC Dropout GRU checkpoint for Junction %d", jid)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        result = evaluate_mc_gru(model, test_dl, pipeline, jid, n_samples=MC_SAMPLES)
        all_results[jid] = result

        # Plot forecast with uncertainty
        n_plot = 200
        plot_forecast(
            result["actuals"][:n_plot],
            result["mean"][:n_plot],
            title=f"MC Dropout GRU – Junction {jid} (90% CI)",
            lower=result["lower"][:n_plot],
            upper=result["upper"][:n_plot],
            save_path=RESULTS_DIR / f"uncertainty_mc_gru_j{jid}.png",
        )

    return all_results


def run_quantile_tft(pipeline: TrafficDataPipeline) -> dict:
    """Evaluate quantile outputs from a pre-trained TFT model.

    Requires Pipeline 2 (SOTA) to have been run first.
    """
    logger.info("═══ Quantile TFT Evaluation ═══")

    # Check for saved TFT checkpoint
    from  app.config import MODELS_DIR
    tft_checkpoints = list(MODELS_DIR.glob("tft-*.ckpt"))

    if not tft_checkpoints:
        logger.warning(
            "No TFT checkpoint found in %s. Run Pipeline 2 (run_sota.py) first.",
            MODELS_DIR,
        )
        return {}

    try:
        from pytorch_forecasting import TemporalFusionTransformer
        from  app.sota.tft_model import build_tft_datasets, prepare_tft_dataframe
        from  app.uncertainty.quantile_tft import evaluate_quantile_tft

        df = load_and_engineer_features(save=False)
        tft_df = prepare_tft_dataframe(df)
        training_ds, _, test_ds, _, _, test_dl = build_tft_datasets(tft_df)

        best_ckpt = sorted(tft_checkpoints)[-1]
        model = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
        logger.info("Loaded TFT from %s", best_ckpt)

        result = evaluate_quantile_tft(model, test_dl)

        # Plot fan chart
        n_plot = 200
        quantiles_for_plot = {
            q: v[:n_plot] for q, v in result["quantile_preds"].items() if q != 0.5
        }
        plot_fan_chart(
            result["actuals"][:n_plot],
            result["median"][:n_plot],
            quantiles_for_plot,
            title="TFT Quantile Forecast (Fan Chart)",
            save_path=RESULTS_DIR / "fan_chart_quantile_tft.png",
        )

        return result

    except Exception as e:
        logger.error("Quantile TFT evaluation failed: %s", e)
        return {}


def run_quantile_xgboost(pipeline: TrafficDataPipeline) -> dict:
    """Train and evaluate XGBoost Quantile Regression per junction."""
    from app.uncertainty.quantile_xgboost import XGBoostQuantile

    all_results = {}
    for jid in JUNCTION_IDS:
        logger.info("═══ XGBoost Quantile – Junction %d ═══", jid)
        train_df, val_df, test_df = pipeline.get_junction_dataframes(jid)
        
        xgb_q = XGBoostQuantile()
        metrics = xgb_q.evaluate(train_df, val_df, test_df, pipeline, jid)
        all_results[jid] = metrics
        
        # Save Fan Chart
        plot_fan_chart(
            metrics["actuals"][:200], metrics["median"][:200],
            {xgb_q.quantiles[0]: metrics["lower"][:200], xgb_q.quantiles[-1]: metrics["upper"][:200]},
            title=f"XGBoost Quantile Forecast – Junction {jid}",
            save_path=RESULTS_DIR / f"fan_chart_xgb_j{jid}.png",
        )
    return all_results


def main(train_mc_dropout: bool = True) -> None:
    setup_logging()
    seed_everything(42)

    logger.info("Preparing data pipeline …")
    pipeline = TrafficDataPipeline()
    pipeline.prepare()

    # ── MC Dropout GRU ──
    logger.info("\n╔══════════════════════════════════╗")
    logger.info("║     MC DROPOUT GRU              ║")
    logger.info("╚══════════════════════════════════╝\n")
    mc_results = run_mc_dropout(pipeline, train_mc_dropout)

    # ── Quantile TFT ──
    logger.info("\n╔══════════════════════════════════╗")
    logger.info("║     QUANTILE TFT                ║")
    logger.info("╚══════════════════════════════════╝\n")
    tft_results = run_quantile_tft(pipeline)

    # ── XGBoost Quantile ──
    logger.info("\n╔══════════════════════════════════╗")
    logger.info("║     XGBOOST QUANTILE            ║")
    logger.info("╚══════════════════════════════════╝\n")
    xgb_results = run_quantile_xgboost(pipeline)

    # ── Comparison ──
    logger.info("\n╔══════════════════════════════════╗")
    logger.info("║   UNCERTAINTY COMPARISON         ║")
    logger.info("╚══════════════════════════════════╝\n")

    summary = {}

    # XGBoost Quantile
    if xgb_results:
        xgb_point = {}
        xgb_cal = {}
        for metric in ["RMSE", "MAE", "MAPE"]:
            xgb_point[metric] = float(np.mean([
                xgb_results[jid]["point_metrics"][metric] for jid in xgb_results
            ]))
        xgb_cal_vals = {
            key: float(np.mean([xgb_results[jid]["calibration"][key] for jid in xgb_results]))
            for key in ["empirical_coverage", "interval_width", "miscalibration"]
        }
        summary["XGBoost Quantile"] = {**xgb_point, **xgb_cal_vals}

    # MC Dropout – average across junctions
    if mc_results:
        mc_point = {}
        mc_cal = {}
        for metric in ["RMSE", "MAE", "MAPE"]:
            mc_point[metric] = float(np.mean([
                mc_results[jid]["point_metrics"][metric] for jid in mc_results
            ]))
        mc_cal_vals = {
            key: float(np.mean([mc_results[jid]["calibration"][key] for jid in mc_results]))
            for key in ["empirical_coverage", "interval_width", "miscalibration"]
        }
        summary["MC Dropout GRU"] = {**mc_point, **mc_cal_vals}

    # Quantile TFT
    if tft_results:
        summary["Quantile TFT"] = {
            **tft_results["point_metrics"],
            **tft_results["calibration"],
        }

    if summary:
        table = build_comparison_table(summary, sort_by="RMSE")
        print_comparison(table)
        table.to_csv(RESULTS_DIR / "uncertainty_comparison.csv")
    else:
        logger.warning("No results to compare.")

    logger.info("Uncertainty pipeline complete ✓")


if __name__ == "__main__":
    main(train_mc_dropout=True)
