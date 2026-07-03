"""
UrbanFlow – Interactive Demo
==============================
Pick a junction, see a probabilistic traffic forecast (median + 90% prediction
interval) from the XGBoost Quantile model against held-out actuals.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.config import COLOR_PALETTE, CONFIDENCE_LEVEL, JUNCTION_IDS
from app.data_pipeline import TrafficDataPipeline
from app.uncertainty.quantile_xgboost import XGBoostQuantile

st.set_page_config(page_title="UrbanFlow – Traffic Forecast Demo", page_icon="🚦", layout="wide")


@st.cache_resource(show_spinner="Loading and preparing traffic data…")
def get_pipeline() -> TrafficDataPipeline:
    pipeline = TrafficDataPipeline()
    pipeline.prepare()
    return pipeline


@st.cache_resource(show_spinner="Training quantile model for this junction…")
def get_forecast(junction_id: int) -> dict:
    pipeline = get_pipeline()
    train_df, val_df, test_df = pipeline.get_junction_dataframes(junction_id)
    model = XGBoostQuantile(confidence_level=CONFIDENCE_LEVEL)
    return model.evaluate(train_df, val_df, test_df, pipeline, junction_id)


def main() -> None:
    st.title("🚦 UrbanFlow — Probabilistic Traffic Forecast")
    st.caption(
        "Spatio-temporal traffic forecasting with calibrated uncertainty. "
        "Pick a junction to see the model's forecast against held-out actual "
        "traffic volume, with a 90% prediction interval."
    )

    with st.sidebar:
        st.header("Controls")
        junction_id = st.selectbox("Junction", JUNCTION_IDS, index=0)
        st.markdown("---")
        st.markdown(
            "**Model:** XGBoost Quantile Regression — the most reliably "
            "calibrated of the three uncertainty-quantification methods "
            "benchmarked in this project (vs. MC Dropout GRU and Quantile TFT)."
        )
        st.markdown("[Source code →](https://github.com/NSVEGUR/urban-flow)")

    result = get_forecast(junction_id)
    actual = result["actuals"]
    median = result["median"]
    lower = result["lower"]
    upper = result["upper"]

    n_points = len(actual)
    default_window = min(240, n_points)
    window = st.slider(
        "Window size (hours)",
        min_value=24,
        max_value=n_points,
        value=default_window,
        step=24,
    )
    max_start = max(0, n_points - window)
    start = st.slider("Start hour into test period", 0, max_start, 0, step=24)
    sl = slice(start, start + window)

    color = COLOR_PALETTE[junction_id]
    x = np.arange(window)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(x, lower[sl], upper[sl], color=color, alpha=0.25, label="90% interval")
    ax.plot(x, actual[sl], color="#222222", linewidth=1.3, label="Actual")
    ax.plot(x, median[sl], color=color, linewidth=1.6, label="Median forecast")
    ax.set_xlabel("Hours into held-out test period")
    ax.set_ylabel("Vehicles")
    ax.set_title(f"Junction {junction_id} — held-out forecast")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"{result['point_metrics']['RMSE']:.2f}")
    col2.metric("MAE", f"{result['point_metrics']['MAE']:.2f}")
    col3.metric(
        "Coverage (90% target)",
        f"{result['calibration']['empirical_coverage']:.0%}",
    )
    col4.metric("Avg. interval width", f"{result['calibration']['interval_width']:.2f}")


if __name__ == "__main__":
    main()
