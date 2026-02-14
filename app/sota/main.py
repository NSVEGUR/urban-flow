"""
UrbanFlow – SOTA Pipeline Orchestrator
=========================================
Trains TFT, performs attention analysis, and compares with classic pipeline.

Usage::

    python -m  sota.run_sota
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from  app.config import SOTA_RESULTS_DIR as RESULTS_DIR, CLASSIC_RESULTS_DIR, TFT_QUANTILES, MODELS_DIR
from  app.data_pipeline import load_and_engineer_features
from  app.evaluation import build_comparison_table, print_comparison
from  app.utils import seed_everything, setup_logging, timer
from  app.visualization import (
    plot_attention_heatmap,
    plot_comparison_bar,
    plot_fan_chart,
    plot_forecast,
    plot_variable_importance,
)

from  app.sota.tft_model import (
    build_tft,
    build_tft_datasets,
    evaluate_tft,
    get_attention_weights,
    get_quantile_predictions,
    prepare_tft_dataframe,
    train_tft,
)

logger = logging.getLogger(__name__)


def main(train_model: bool = True) -> None:
    setup_logging()
    seed_everything(42)

    # ── Data Preparation ──
    logger.info("Loading and preparing data for TFT …")
    df = load_and_engineer_features(save=True)
    tft_df = prepare_tft_dataframe(df)

    # ── Build Datasets ──
    logger.info("Building pytorch-forecasting datasets …")
    training_ds, val_ds, test_ds, train_dl, val_dl, test_dl = build_tft_datasets(tft_df)

    model_ckpt_path = MODELS_DIR / "tft-best.ckpt"

    if train_model or not model_ckpt_path.exists():
        model = build_tft(training_ds)
        with timer("TFT Training"):
            model, trainer = train_tft(model, train_dl, val_dl)
        # Save best checkpoint path consistently
        best_ckpt = MODELS_DIR / f"tft-best.ckpt"
        if trainer.checkpoint_callback.best_model_path: # type: ignore
            Path(trainer.checkpoint_callback.best_model_path).rename(best_ckpt) # type: ignore
    else:
        from pytorch_forecasting import TemporalFusionTransformer
        model = TemporalFusionTransformer.load_from_checkpoint(model_ckpt_path)
        logger.info("Loaded TFT from saved checkpoint: %s", model_ckpt_path)


    # ── Evaluate ──
    logger.info("\n═══ TFT Evaluation ═══")
    actuals, preds, metrics = evaluate_tft(model, test_dl)

    plot_forecast(
        actuals[:200], preds[:200],
        title="TFT Forecast – All Junctions (Median)",
        save_path=RESULTS_DIR / "forecast_tft.png",
    )

    # ── Quantile Outputs ──
    logger.info("Extracting quantile predictions …")
    quantile_preds = get_quantile_predictions(model, test_dl)

    if len(quantile_preds) > 1:
        # Use median as mean_pred
        median = quantile_preds.get(0.5, preds)
        plot_fan_chart(
            actuals[:200], median[:200],
            {q: v[:200] for q, v in quantile_preds.items() if q != 0.5},
            title="TFT Probabilistic Forecast (Fan Chart)",
            save_path=RESULTS_DIR / "fan_chart_tft.png",
        )

    # ── Attention Analysis ──
    logger.info("Extracting attention weights …")
    try:
        interp = get_attention_weights(model, test_dl)

        if "attention" in interp:
            plot_attention_heatmap(
                interp["attention"],
                title="TFT Temporal Attention Weights",
                save_path=RESULTS_DIR / "attention_heatmap_tft.png",
            )

        if "variable_importance" in interp:
            plot_variable_importance(
                interp["variable_importance"],
                title="TFT Variable Importance",
                save_path=RESULTS_DIR / "variable_importance_tft.png",
            )
    except Exception as e:
        logger.warning("Attention analysis failed: %s", e)

    # ── Comparison with Classic Results ──
    classic_csv = CLASSIC_RESULTS_DIR / "classic_comparison.csv"
    if classic_csv.exists():
        logger.info("Loading classic pipeline results for comparison …")
        classic_table = pd.read_csv(classic_csv, index_col=0)
        # Add TFT row
        tft_row = pd.DataFrame(metrics, index=["TFT"])
        combined = pd.concat([classic_table, tft_row])
        combined = combined.sort_values("RMSE")
        print_comparison(combined)
        combined.to_csv(RESULTS_DIR / "full_comparison.csv")
        plot_comparison_bar(
            combined, metric="RMSE",
            title="Full Model Comparison – RMSE",
            save_path=RESULTS_DIR / "full_comparison_rmse.png",
        )
    else:
        logger.info("No classic results found — run Pipeline 1 first for comparison.")
        summary = {"TFT": metrics}
        table = build_comparison_table(summary)
        print_comparison(table)
        table.to_csv(RESULTS_DIR / "tft_results.csv")

    logger.info("SOTA pipeline complete ✓")


if __name__ == "__main__":
    main(train_model=False)  # Set to True to retrain TFT, False to load existing checkpoint
