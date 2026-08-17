"""
config.py

Single source of truth for paths, hyperparameters, and constants.
Every other script imports from here instead of hard-coding values —
change a path or a hyperparameter once, everywhere picks it up.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "input")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Raw processed source file (the single CSV the original notebook read).
RAW_DATA_FILE = os.path.join(INPUT_DIR, "Sales_Processed.csv")

# Chronological splits produced by create_folds.py
TRAINING_FILE = os.path.join(INPUT_DIR, "train.csv")   # train + validation rows, flagged by `split`
TEST_FILE = os.path.join(INPUT_DIR, "test.csv")        # held-out final test period

# Artifacts produced by train.py
MODEL_PATH = os.path.join(MODEL_DIR, "model_lstm.keras")
FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.bin")
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.bin")
BEST_EPOCH_PATH = os.path.join(MODEL_DIR, "best_epoch.bin")

# Artifacts produced by the final refit + inference.py
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "model_lstm_final.keras")
FINAL_FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, "final_feature_scaler.bin")
FINAL_TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "final_target_scaler.bin")

DAILY_FORECAST_FILE = os.path.join(OUTPUT_DIR, "Sales_Forecast_2017_Daily.csv")
MONTHLY_FORECAST_FILE = os.path.join(OUTPUT_DIR, "Sales_Forecast_2017_Monthly.csv")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Feature engineering / windowing
# ---------------------------------------------------------------------------
DATE_COL = "Date"
TARGET_COL = "Sales"

FEATURE_COLS = [
    "Sales",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]

TIMESTEPS = 7  # 7-day lookback window

# Chronological split ratios (no shuffling — time series).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85  # cumulative — train ends at 0.70, val ends at 0.85, rest is test

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LSTM_UNITS = 64
DROPOUT_RATE = 0.20
DENSE_UNITS = 32
MAX_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# ---------------------------------------------------------------------------
# Forecast horizon
# ---------------------------------------------------------------------------
FORECAST_START = "2017-01-01"
FORECAST_END = "2017-12-31"
