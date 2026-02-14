"""
UrbanFlow – Global Configuration
=================================
Central place for hyperparameters, paths, feature lists, and device setup.
"""

from pathlib import Path
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "traffic.csv"
AUGMENTED_DATA_PATH = DATA_DIR / "traffic_augmented.csv"
MODELS_DIR = PROJECT_ROOT / "app" / "models"
RESULTS_DIR = PROJECT_ROOT / "app" / "results"
EDA_RESULTS_DIR = RESULTS_DIR / "eda"
CLASSIC_RESULTS_DIR = RESULTS_DIR / "classic"
SOTA_RESULTS_DIR = RESULTS_DIR / "sota"
UNCERTAINTY_RESULTS_DIR = RESULTS_DIR / "uncertainty"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLASSIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SOTA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UNCERTAINTY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
JUNCTION_IDS = [1, 2, 3, 4]
TARGET_COL = "Vehicles"
DATETIME_COL = "DateTime"
JUNCTION_COL = "Junction"

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
LAG_HOURS = [1, 24, 168]                # t-1, same hour yesterday, same hour last week
ROLLING_WINDOWS = [24]                   # 24-hour rolling statistics
CYCLICAL_FEATURES = ["hour", "day_of_week", "month"]

# Columns produced by feature engineering (populated at runtime by data_pipeline)
TIME_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
]
LAG_FEATURES = [f"lag_{h}" for h in LAG_HOURS]
ROLLING_FEATURES = [f"rolling_mean_{w}" for w in ROLLING_WINDOWS] + \
                   [f"rolling_std_{w}" for w in ROLLING_WINDOWS]

ALL_FEATURES = TIME_FEATURES + LAG_FEATURES + ROLLING_FEATURES

# ──────────────────────────────────────────────
# Sequence / Training
# ──────────────────────────────────────────────
SEQ_LEN = 24  # Use past 24 hours to predict next 24 hours
CLASSIC_SEQ_LEN = 168
SOTA_SEQ_LEN = 168  

FORECAST_HORIZON = 24   # Predict next 24 hours
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15           # Early-stopping patience
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2

# ──────────────────────────────────────────────
# Train / Val / Test split ratios (chronological)
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# GRU Architecture
# ──────────────────────────────────────────────
GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS = 2

# ──────────────────────────────────────────────
# TFT
# ──────────────────────────────────────────────
TFT_HIDDEN_SIZE = 64
TFT_ATTENTION_HEAD_SIZE = 4
TFT_DROPOUT = 0.1
TFT_LEARNING_RATE = 1e-3
TFT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# ──────────────────────────────────────────────
# MC Dropout (Probabilistic)
# ──────────────────────────────────────────────
MC_SAMPLES = 100        # Number of forward passes for uncertainty
MC_DROPOUT = 0.4
CONFIDENCE_LEVEL = 0.90 # 90% confidence intervals

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
COLOR_PALETTE = {
    1: "#152F67",
    2: "#6D79BA",
    3: "#98A2E7",
    4: "#DDE6FF",
}
JUNCTION_COLORS = [COLOR_PALETTE[j] for j in JUNCTION_IDS]
FIGURE_DPI = 150
