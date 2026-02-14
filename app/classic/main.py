"""
UrbanFlow – Classic Pipeline Orchestrator
============================================
Trains all baselines and GRU variants, compares metrics, and saves results.

Usage::

    python -m  classic.run_classic
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from  app.config import ALL_FEATURES, DEVICE, CLASSIC_SEQ_LEN, JUNCTION_IDS, CLASSIC_RESULTS_DIR as RESULTS_DIR, MODELS_DIR
from  app.data_pipeline import TrafficDataPipeline
from  app.evaluation import build_comparison_table, print_comparison
from  app.utils import seed_everything, setup_logging, timer
from  app.visualization import plot_comparison_bar, plot_forecast
from  app.classic.baselines import ARIMABaseline, NaiveSeasonalBaseline
from  app.classic.xgboost_model import XGBoostBaseline
from  app.classic.univariate_gru import UniGRU, evaluate_uni_gru, train_uni_gru
from  app.classic.spatiotemporal_gru import (
    SpatioTemporalGRU,
    evaluate_per_junction,
    evaluate_spatiotemporal_gru,
    train_spatiotemporal_gru,
)

logger = logging.getLogger(__name__)


def run_baselines(pipeline: TrafficDataPipeline) -> dict:
    """Run Naive, ARIMA, and XGBoost baselines for each junction."""
    all_results = {}

    for jid in JUNCTION_IDS:
        logger.info("═══ Junction %d ═══", jid)
        train_arr, val_arr, test_arr = pipeline.get_junction_arrays(jid, scaled=False)
        train_df, val_df, test_df = pipeline.get_junction_dataframes(jid)

        jresults = {}

        # Naive
        with timer(f"Naive – J{jid}"):
            naive = NaiveSeasonalBaseline()
            naive_preds, naive_metrics = naive.evaluate(train_arr, test_arr)
            jresults["Naive Seasonal"] = naive_metrics

            # Save forecast plot
            plot_forecast(
                naive_preds[:200], test_arr[:200],
                title=f"Naive Seasonal Forecast – Junction {jid}",
                save_path=RESULTS_DIR / f"forecast_naive_j{jid}.png",
            )

        # ARIMA  (can be slow — run on a shorter series if needed)
        with timer(f"ARIMA – J{jid}"):
            try:
                arima = ARIMABaseline()
                arima_preds, arima_metrics = arima.evaluate(train_arr, test_arr)
                jresults["ARIMA"] = arima_metrics
                # Save forecast plot
                plot_forecast(
                    arima_preds[:200], test_arr[:200],
                    title=f"ARIMA Forecast – Junction {jid}",
                    save_path=RESULTS_DIR / f"forecast_arima_j{jid}.png",
                )
            except Exception as e:
                logger.warning("ARIMA failed for Junction %d: %s", jid, e)
                jresults["ARIMA"] = {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}

        # XGBoost
        with timer(f"XGBoost – J{jid}"):
            xgb = XGBoostBaseline()
            xgb_preds, xgb_metrics = xgb.evaluate(train_df, val_df, test_df, pipeline=pipeline, junction_id=jid)
            jresults["XGBoost"] = xgb_metrics

            # Save forecast plot
            plot_forecast(
                xgb_preds[:200], test_arr[:200],
                title=f"XGBoost Forecast – Junction {jid}",
                save_path=RESULTS_DIR / f"forecast_xgb_j{jid}.png",
            )


        all_results[jid] = jresults

    return all_results


def run_univariate_gru(pipeline: TrafficDataPipeline, train_model: bool = True) -> dict:
    """Train per-junction UniGRU models."""
    all_results = {}

    for jid in JUNCTION_IDS:
        logger.info("═══ UniGRU – Junction %d ═══", jid)
        train_dl, val_dl, test_dl = pipeline.get_junction_dataloaders(jid)

        n_features = len(ALL_FEATURES)
        model = UniGRU(input_size=n_features)

        ckpt_path = MODELS_DIR / f"uni_gru_j{jid}.pt"

        if train_model or not ckpt_path.exists():
            with timer(f"UniGRU Train – J{jid}"):
                history = train_uni_gru(
                    model, train_dl, val_dl,
                    checkpoint_name=f"uni_gru_j{jid}",
                )
        else:
            logger.info("Loading UniGRU checkpoint for Junction %d", jid)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))


        actuals, preds, metrics = evaluate_uni_gru(model, test_dl, pipeline=pipeline, junction_id=jid)
        all_results[jid] = metrics

        # Save forecast plot
        plot_forecast(
            actuals[:200], preds[:200],
            title=f"UniGRU Forecast – Junction {jid}",
            save_path=RESULTS_DIR / f"forecast_uni_gru_j{jid}.png",
        )

    return all_results


def run_spatiotemporal_gru(pipeline: TrafficDataPipeline, train_model: bool = True) -> dict:
    """Train a single SpatioTemporalGRU across all junctions."""
    logger.info("═══ SpatioTemporalGRU (all junctions) ═══")
    train_dl, val_dl, test_dl = pipeline.get_spatiotemporal_dataloaders()

    n_features = len(ALL_FEATURES)
    model = SpatioTemporalGRU(
        n_junctions=len(JUNCTION_IDS),
        n_features=n_features,
    )

    ckpt_path = MODELS_DIR / "spatiotemporal_gru.pt"

    if train_model or not ckpt_path.exists():
        with timer("SpatioTemporalGRU Train"):
            history = train_spatiotemporal_gru(
                model, train_dl, val_dl,
                checkpoint_name="spatiotemporal_gru",
            )
    else:
        logger.info("Loading SpatioTemporalGRU checkpoint")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))


    # Aggregate metrics
    actuals, preds, agg_metrics = evaluate_spatiotemporal_gru(
        model, test_dl, pipeline=pipeline, junction_ids=JUNCTION_IDS
    )

    # Per-junction metrics
    per_j = evaluate_per_junction(model, test_dl, pipeline=pipeline, junction_ids=JUNCTION_IDS)

    # Save forecast plot
    plot_forecast(
        actuals[:200], preds[:200],
        title="SpatioTemporalGRU Forecast – All Junctions (Median)",
        save_path=RESULTS_DIR / "forecast_spatiotemporal_gru.png",
    )

    return {"aggregate": agg_metrics, "per_junction": per_j}


def main(run_baseline: bool = True, run_uni_gru: bool = True, run_st_gru: bool = True, train_uni_gru_model: bool = True, train_st_gru_model: bool = True) -> None:
    setup_logging()
    seed_everything(42)

    logger.info("Preparing data pipeline …")
    pipeline = TrafficDataPipeline(seq_len=CLASSIC_SEQ_LEN)
    pipeline.prepare()

    # ── Baselines ──
    baseline_results = {}
    if run_baseline:
        logger.info("\n╔══════════════════════════════════╗")
        logger.info("║       BASELINE MODELS           ║")
        logger.info("╚══════════════════════════════════╝\n")
        baseline_results = run_baselines(pipeline)


    # ── UniGRU ──
    uni_gru_results = {}
    if run_uni_gru:
        logger.info("\n╔══════════════════════════════════╗")
        logger.info("║       UNIVARIATE GRU            ║")
        logger.info("╚══════════════════════════════════╝\n")
        uni_gru_results = run_univariate_gru(pipeline, train_model=train_uni_gru_model)

    # ── SpatioTemporal GRU ──
    st_results = {}
    if run_st_gru:
        logger.info("\n╔══════════════════════════════════╗")
        logger.info("║     SPATIO-TEMPORAL GRU         ║")
        logger.info("╚══════════════════════════════════╝\n")
        st_results = run_spatiotemporal_gru(pipeline, train_model=train_st_gru_model)

    if not (run_baseline or run_uni_gru or run_st_gru):
        logger.warning("No models were run. Please set at least one of run_baseline, run_uni_gru, or run_st_gru to True.")
        return

    # ── Summary Table ──
    logger.info("\n╔══════════════════════════════════╗")
    logger.info("║       FINAL COMPARISON          ║")
    logger.info("╚══════════════════════════════════╝\n")

    classic_csv = RESULTS_DIR / "classic_comparison.csv"

    # Load existing table if exists
    if classic_csv.exists():
        existing = pd.read_csv(classic_csv, index_col=0)
    else:
        existing = pd.DataFrame()

    summary = {}

    # ── Baselines ──
    if run_baseline:
        for model_name in ["Naive Seasonal", "ARIMA", "XGBoost"]:
            avg_metrics = {}
            for metric in ["RMSE", "MAE", "MAPE"]:
                vals = [baseline_results[jid][model_name][metric] for jid in JUNCTION_IDS]
                avg_metrics[metric] = float(np.nanmean(vals))
            summary[model_name] = avg_metrics

    # ── UniGRU ──
    if run_uni_gru:
        avg_uni = {}
        for metric in ["RMSE", "MAE", "MAPE"]:
            avg_uni[metric] = float(np.mean([uni_gru_results[jid][metric] for jid in JUNCTION_IDS]))
        summary["UniGRU"] = avg_uni

    # ── SpatioTemporal GRU ──
    if run_st_gru:
        summary["SpatioTemporalGRU"] = st_results["aggregate"]

    # Convert new summary to DataFrame
    new_table = build_comparison_table(summary)

    # Overwrite only relevant rows
    for model in new_table.index:
        existing.loc[model] = new_table.loc[model]

    # Sort by RMSE
    existing = existing.sort_values("RMSE")

    print_comparison(existing)

    # Save updated file
    existing.to_csv(classic_csv)

    logger.info("Results saved to %s", RESULTS_DIR)
    logger.info("Done! ✓")


if __name__ == "__main__":
    main(
        run_baseline=False,
        run_uni_gru=False,
        run_st_gru=True,
        train_uni_gru_model=False,
        train_st_gru_model=True
    ) 
