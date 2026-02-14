"""
UrbanFlow – Visualization Helpers
====================================
Publication-quality plots for forecasts, attention, uncertainty, and EDA.

All ``plot_*`` functions accept an optional *save_path*; when provided the
figure is saved to disk and ``plt.close()`` is called to free memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from app.config import COLOR_PALETTE, FIGURE_DPI, JUNCTION_IDS, RESULTS_DIR

# ── Global style ──
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (14, 5),
})


def _save_or_show(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ──────────────────────────────────────────────
# EDA Plots
# ──────────────────────────────────────────────

def plot_time_series_overview(
    df: pd.DataFrame,
    datetime_col: str = "DateTime",
    target_col: str = "Vehicles",
    junction_col: str = "Junction",
    save_path: Optional[str | Path] = None,
) -> None:
    """Line plot of vehicles over time, colored by junction."""
    fig, ax = plt.subplots(figsize=(18, 5))
    for jid in JUNCTION_IDS:
        subset = df[df[junction_col] == jid]
        ax.plot(
            pd.to_datetime(subset[datetime_col]),
            subset[target_col],
            color=COLOR_PALETTE[jid],
            label=f"Junction {jid}",
            alpha=0.7,
            linewidth=0.6,
        )
    ax.set_title("Traffic Volume Across Junctions Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vehicles / hour")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_or_show(fig, save_path)


def plot_seasonal_decomposition(
    result,  # statsmodels DecomposeResult
    junction_id: int,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot the four components from seasonal_decompose."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = ["observed", "trend", "seasonal", "resid"]
    titles = ["Observed", "Trend", "Seasonal", "Residual"]
    color = COLOR_PALETTE[junction_id]

    for ax, comp, title in zip(axes, components, titles):
        data = getattr(result, comp)
        ax.plot(data, color=color, linewidth=0.8)
        ax.set_ylabel(title)
    axes[0].set_title(f"Seasonal Decomposition – Junction {junction_id}")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_acf_pacf(
    series: pd.Series,
    junction_id: int,
    lags: int = 72,
    save_path: Optional[str | Path] = None,
) -> None:
    """ACF and PACF side-by-side."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=ax1, color=COLOR_PALETTE[junction_id])
    ax1.set_title(f"ACF – Junction {junction_id}")
    plot_pacf(series.dropna(), lags=lags, ax=ax2, color=COLOR_PALETTE[junction_id])
    ax2.set_title(f"PACF – Junction {junction_id}")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_hourly_boxplot(
    df: pd.DataFrame,
    junction_id: int,
    save_path: Optional[str | Path] = None,
) -> None:
    """Box plot of vehicle counts by hour of day for one junction."""
    subset = df[df["Junction"] == junction_id].copy()
    subset["hour"] = pd.to_datetime(subset["DateTime"]).dt.hour
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=subset, x="hour", y="Vehicles", color=COLOR_PALETTE[junction_id], ax=ax)
    ax.set_title(f"Hourly Distribution – Junction {junction_id}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Vehicles")
    _save_or_show(fig, save_path)


def plot_cross_correlation(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> None:
    """Heatmap of cross-correlation between junctions."""
    pivoted = df.pivot_table(
        index="DateTime", columns="Junction", values="Vehicles"
    )
    corr = pivoted.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="Blues",
        square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Cross-Correlation Between Junctions")
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Forecast Plots
# ──────────────────────────────────────────────

def plot_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Forecast vs Actual",
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    dates: Optional[Sequence] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Overlay plot of actual vs predicted, with optional confidence band."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = dates if dates is not None else np.arange(len(actual))

    ax.plot(x, actual, color="#152F67", label="Actual", alpha=0.8, linewidth=1.0)
    ax.plot(x, predicted, color="#E74C3C", label="Predicted", alpha=0.8, linewidth=1.0)

    if lower is not None and upper is not None:
        ax.fill_between(
            x, lower, upper,
            color="#E74C3C", alpha=0.15, label="90% CI",
        )

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Vehicles")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_fan_chart(
    actual: np.ndarray,
    mean_pred: np.ndarray,
    quantiles: Dict[float, np.ndarray],
    title: str = "Probabilistic Forecast (Fan Chart)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Fan chart showing multiple confidence bands.

    Parameters
    ----------
    quantiles : dict
        e.g. ``{0.1: lower_10, 0.25: lower_25, 0.75: upper_75, 0.9: upper_90}``
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(actual))

    ax.plot(x, actual, color="#152F67", label="Actual", linewidth=1.0)
    ax.plot(x, mean_pred, color="#E74C3C", label="Median / Mean", linewidth=1.0)

    # Band pairs: (0.1, 0.9), (0.25, 0.75)
    alphas = [0.12, 0.22]
    pairs = [(0.1, 0.9), (0.25, 0.75)]
    for (lo_q, hi_q), alpha in zip(pairs, alphas):
        if lo_q in quantiles and hi_q in quantiles:
            ax.fill_between(
                x, quantiles[lo_q], quantiles[hi_q],
                color="#E74C3C", alpha=alpha,
                label=f"{int(lo_q*100)}–{int(hi_q*100)}%",
            )

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Vehicles")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Comparison Bar Chart
# ──────────────────────────────────────────────

def plot_comparison_bar(
    table: pd.DataFrame,
    metric: str = "RMSE",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Grouped bar chart of a single metric across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = table.index.tolist()
    values = table[metric].values
    colors = sns.color_palette("Blues_d", len(models))

    bars = ax.barh(models, values, color=colors, edgecolor="white")
    ax.set_xlabel(metric)
    ax.set_title(title or f"Model Comparison – {metric}")
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01 * max(values),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", fontsize=10,
        )

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────
# Attention Heatmap (for TFT)
# ──────────────────────────────────────────────

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    time_labels: Optional[List[str]] = None,
    title: str = "Temporal Attention Weights",
    save_path: Optional[str | Path] = None,
) -> None:
    """Heatmap of TFT temporal attention weights.

    Parameters
    ----------
    attention_weights : ndarray of shape (n_heads, seq_len) or (seq_len,)
    """
    if attention_weights.ndim == 1:
        attention_weights = attention_weights.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(16, 3 * attention_weights.shape[0]))
    sns.heatmap(
        attention_weights, cmap="YlOrRd", ax=ax,
        xticklabels=time_labels or False,
        yticklabels=[f"Head {i+1}" for i in range(attention_weights.shape[0])],
    )
    ax.set_title(title)
    ax.set_xlabel("Time Step (lookback)")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_variable_importance(
    importance: Dict[str, float],
    title: str = "Variable Importance (TFT)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Horizontal bar chart of feature importances from TFT."""
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(names, values, color=sns.color_palette("Blues_r", len(names)))
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    _save_or_show(fig, save_path)
