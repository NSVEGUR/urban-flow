"""
UrbanFlow – Exploratory Data Analysis
=========================================
Comprehensive EDA across all 4 junctions: trends, seasonality, stationarity,
cross-correlation, and distribution analysis.

Usage::

    python run_eda.py

All plots are saved to ``app/results/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    COLOR_PALETTE,
    DATETIME_COL,
    JUNCTION_COL,
    JUNCTION_IDS,
    RAW_DATA_PATH,
    EDA_RESULTS_DIR as RESULTS_DIR,
    TARGET_COL,
)
from app.utils import setup_logging
from app.visualization import (
    plot_acf_pacf,
    plot_cross_correlation,
    plot_hourly_boxplot,
    plot_seasonal_decomposition,
    plot_time_series_overview,
)

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load and minimally preprocess the raw traffic dataset."""
    df = pd.read_csv(RAW_DATA_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.drop(columns=["ID"], errors="ignore")
    df = df.sort_values([DATETIME_COL, JUNCTION_COL]).reset_index(drop=True)
    logger.info("Loaded %d rows across %d junctions.", len(df), df[JUNCTION_COL].nunique())
    return df


def run_stationarity_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Augmented Dickey-Fuller test on each junction's raw series.

    Returns a summary DataFrame.
    """
    results = []
    for jid in JUNCTION_IDS:
        series = df[df[JUNCTION_COL] == jid][TARGET_COL].dropna()
        adf_result = adfuller(series)
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        critical = adf_result[3]
        stationary = p_value < 0.05
        results.append({
            "Junction": jid,    
            "ADF Statistic": round(float(adf_stat), 4),
            "p-value": round(float(p_value), 6),
            "Critical 1%": round(critical, 4), # type: ignore
            "Critical 5%": round(critical, 4), # type: ignore
            "Stationary": "Yes" if stationary else "No",
        })
        logger.info(
            "  Junction %d  │  ADF=%.4f  p=%.6f  → %s",
            jid, adf_stat, p_value, "Stationary" if stationary else "Non-Stationary",
        )

    return pd.DataFrame(results)


def run_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-junction descriptive statistics."""
    stats = df.groupby(JUNCTION_COL)[TARGET_COL].describe()
    logger.info("Descriptive statistics:\n%s", stats.to_string())
    return stats


def plot_yearly_monthly_trends(df: pd.DataFrame) -> None:
    """Plot average traffic by month and year for each junction."""
    df = df.copy()
    dt_series = df[DATETIME_COL]
    df["Year"] = df[DATETIME_COL].dt.year  # type: ignore
    df["Month"] = df[DATETIME_COL].dt.month # type: ignore
    df["Hour"] = df[DATETIME_COL].dt.hour # type: ignore

    # Monthly trend
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    monthly = df.groupby([JUNCTION_COL, "Month"])[TARGET_COL].mean().reset_index()
    for jid in JUNCTION_IDS:
        subset = monthly[monthly[JUNCTION_COL] == jid]
        axes[0].plot(subset["Month"], subset[TARGET_COL],
                     marker="o", color=COLOR_PALETTE[jid], label=f"J{jid}")
    axes[0].set_title("Average Vehicles by Month")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Avg Vehicles")
    axes[0].legend()
    axes[0].set_xticks(range(1, 13))

    # Hourly trend
    hourly = df.groupby([JUNCTION_COL, "Hour"])[TARGET_COL].mean().reset_index()
    for jid in JUNCTION_IDS:
        subset = hourly[hourly[JUNCTION_COL] == jid]
        axes[1].plot(subset["Hour"], subset[TARGET_COL],
                     marker="o", color=COLOR_PALETTE[jid], label=f"J{jid}")
    axes[1].set_title("Average Vehicles by Hour of Day")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Avg Vehicles")
    axes[1].legend()
    axes[1].set_xticks(range(0, 24))

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_monthly_hourly_trends.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved monthly/hourly trends → eda_monthly_hourly_trends.png")


def plot_day_of_week_trend(df: pd.DataFrame) -> None:
    """Box plot of traffic by day of week."""
    df = df.copy()
    df["DayOfWeek"] = df[DATETIME_COL].dt.day_name() # type: ignore
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for idx, jid in enumerate(JUNCTION_IDS):
        ax = axes[idx // 2][idx % 2]
        subset = df[df[JUNCTION_COL] == jid]
        sns.boxplot(
            data=subset, x="DayOfWeek", y=TARGET_COL,
            order=day_order, color=COLOR_PALETTE[jid], ax=ax,
        )
        ax.set_title(f"Junction {jid}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Traffic Distribution by Day of Week", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_day_of_week.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved day-of-week trends → eda_day_of_week.png")


def run_seasonal_decomposition(df: pd.DataFrame) -> None:
    """Seasonal decomposition for each junction (daily period=24)."""
    for jid in JUNCTION_IDS:
        series = df[df[JUNCTION_COL] == jid].set_index(DATETIME_COL)[TARGET_COL]
        series = series.sort_index().dropna()

        if len(series) < 48:
            logger.warning("Junction %d: too few data points for decomposition.", jid)
            continue

        result = seasonal_decompose(series, model="additive", period=24)
        plot_seasonal_decomposition(
            result, junction_id=jid,
            save_path=RESULTS_DIR / f"eda_seasonal_decomp_j{jid}.png",
        )
        logger.info("Saved seasonal decomposition for Junction %d", jid)


def run_acf_pacf(df: pd.DataFrame) -> None:
    """ACF/PACF plots for each junction."""
    for jid in JUNCTION_IDS:
        series = df[df[JUNCTION_COL] == jid][TARGET_COL].dropna()
        plot_acf_pacf(
            series, junction_id=jid, lags=72,
            save_path=RESULTS_DIR / f"eda_acf_pacf_j{jid}.png",
        )
        logger.info("Saved ACF/PACF for Junction %d", jid)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers using IQR method per junction."""
    outlier_counts = []
    for jid in JUNCTION_IDS:
        series = df[df[JUNCTION_COL] == jid][TARGET_COL]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((series < lower) | (series > upper)).sum()
        pct = n_outliers / len(series) * 100
        outlier_counts.append({
            "Junction": jid,
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "Outliers": n_outliers,
            "Outlier %": round(pct, 2),
        })
        logger.info("  Junction %d: %d outliers (%.1f%%)", jid, n_outliers, pct)

    return pd.DataFrame(outlier_counts)


def main() -> None:
    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("╔══════════════════════════════════╗")
    logger.info("║   EXPLORATORY DATA ANALYSIS     ║")
    logger.info("╚══════════════════════════════════╝\n")

    df = load_raw_data()

    # 1. Descriptive Stats
    logger.info("─── Descriptive Statistics ───")
    stats = run_descriptive_stats(df)
    stats.to_csv(RESULTS_DIR / "eda_descriptive_stats.csv")

    # 2. Missing values
    logger.info("─── Missing Values ───")
    missing = df.isnull().sum()
    logger.info("Missing values:\n%s", missing.to_string())

    # 3. Overview time series
    logger.info("─── Time Series Overview ───")
    plot_time_series_overview(
        df, save_path=RESULTS_DIR / "eda_time_series_overview.png",
    )

    # 4. Monthly & Hourly trends
    logger.info("─── Trend Analysis ───")
    plot_yearly_monthly_trends(df)
    plot_day_of_week_trend(df)

    # 5. Hourly box plots per junction
    logger.info("─── Hourly Distributions ───")
    for jid in JUNCTION_IDS:
        plot_hourly_boxplot(df, jid, save_path=RESULTS_DIR / f"eda_hourly_boxplot_j{jid}.png")

    # 6. Cross-correlation
    logger.info("─── Cross-Correlation ───")
    plot_cross_correlation(df, save_path=RESULTS_DIR / "eda_cross_correlation.png")

    # 7. Seasonal decomposition
    logger.info("─── Seasonal Decomposition ───")
    run_seasonal_decomposition(df)

    # 8. ACF / PACF
    logger.info("─── ACF / PACF ───")
    run_acf_pacf(df)

    # 9. Stationarity tests
    logger.info("─── Stationarity Tests (ADF) ───")
    adf_results = run_stationarity_tests(df)
    adf_results.to_csv(RESULTS_DIR / "eda_stationarity_tests.csv", index=False)
    print("\n" + adf_results.to_string(index=False))

    # 10. Outlier detection
    logger.info("─── Outlier Detection (IQR) ───")
    outliers = detect_outliers(df)
    outliers.to_csv(RESULTS_DIR / "eda_outliers.csv", index=False)
    print("\n" + outliers.to_string(index=False))

    logger.info("\n✓ EDA complete! All plots saved to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
