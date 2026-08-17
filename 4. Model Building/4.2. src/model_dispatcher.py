"""
model_dispatcher.py

Maps a model name to the callable that produces its predictions.
train.py picks a name off the command line (default "lstm") and
looks it up here instead of branching on if/elif — add a new model
by adding one entry.

Includes the naive baselines and ARIMA/SARIMAX from the notebook's
model-comparison section, since they're forecasters just like the
LSTM, only simpler ones.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import models


def naive_previous_day(df_model: pd.DataFrame, val_end: int, n: int) -> np.ndarray:
    """Tomorrow = today's Sales."""
    return df_model["Sales"].iloc[val_end - 1: n - 1].to_numpy()


def seasonal_naive_previous_week(df_model: pd.DataFrame, val_end: int, n: int) -> np.ndarray:
    """Tomorrow = same weekday, one week earlier."""
    return df_model["Sales"].iloc[val_end - 7: n - 7].to_numpy()


def forecast_arima(train_series: pd.Series, steps: int, order=(5, 1, 0)) -> np.ndarray:
    fit = ARIMA(train_series, order=order).fit()
    return fit.forecast(steps=steps)


def forecast_sarimax(train_series: pd.Series, steps: int,
                      order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)) -> np.ndarray:
    fit = SARIMAX(train_series, order=order, seasonal_order=seasonal_order).fit()
    return fit.forecast(steps=steps)


# Name -> model-builder callable. "lstm" is the only trainable Keras
# model here; the baselines above are called directly since they
# don't share the build/fit/predict shape of a Keras model.
MODELS = {
    "lstm": models.build_lstm_model,
}
