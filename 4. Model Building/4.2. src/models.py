"""
models.py

Model architecture definitions only — no data loading, no training
loop. Anything that builds a compiled model or shapes data for one
lives here so train.py and inference.py both call the same code
instead of drifting apart.
"""

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

import config


def build_lstm_model(timesteps: int = config.TIMESTEPS,
                      n_features: int = len(config.FEATURE_COLS)):
    """The architecture used for both the validation-stage model and
    the final full-history refit — identical on purpose, so metrics
    from the validation run are a fair estimate of the final model."""
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(config.LSTM_UNITS),
        Dropout(config.DROPOUT_RATE),
        Dense(config.DENSE_UNITS, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


def create_sequences(features: np.ndarray, target: np.ndarray,
                      timesteps: int = config.TIMESTEPS):
    """Slide a `timesteps`-day window across features/target to build
    LSTM-ready (samples, timesteps, n_features) / (samples, 1) arrays."""
    X, y = [], []

    for i in range(timesteps, len(features)):
        X.append(features[i - timesteps:i])
        y.append(target[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )
