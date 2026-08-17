"""
inference.py

The production stage (notebook sections 15-25). Everything here
assumes train.py has already run once, so config.BEST_EPOCH_PATH
exists — that epoch count is what "the best number of epochs
selected from the validation experiment" means below.

Steps:
    1. Load ALL historical rows (train.csv + test.csv) and best_epoch
    2. Refit scalers + model on the full history (no held-out split —
       every observation is used for the production model)
    3. Build 2017 calendar features (known in advance, so this is safe)
    4. Recursively forecast one day at a time: predict day t, append
       it to the rolling 7-day window, predict day t+1, ...
    5. Export daily + monthly forecast CSVs and save the final model

Run directly (after train.py):
    python -m src.inference
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
import models


def load_full_history() -> pd.DataFrame:
    train_valid_df = pd.read_csv(config.TRAINING_FILE, parse_dates=[config.DATE_COL]).drop(columns="split")
    test_df = pd.read_csv(config.TEST_FILE, parse_dates=[config.DATE_COL])
    df_model = pd.concat([train_valid_df, test_df], ignore_index=True)
    df_model = df_model.sort_values(config.DATE_COL).reset_index(drop=True)
    return df_model


def load_best_epoch() -> int:
    with open(config.BEST_EPOCH_PATH, "rb") as f:
        return pickle.load(f)


def refit_on_full_history(df_model: pd.DataFrame, best_epoch: int):
    final_feature_scaler = MinMaxScaler()
    final_target_scaler = MinMaxScaler()

    all_features = final_feature_scaler.fit_transform(df_model[config.FEATURE_COLS]).astype(np.float32)
    all_target = final_target_scaler.fit_transform(df_model[[config.TARGET_COL]]).astype(np.float32)

    X_all, y_all = models.create_sequences(all_features, all_target, config.TIMESTEPS)

    final_model = models.build_lstm_model()
    final_model.fit(
        X_all, y_all,
        epochs=best_epoch,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        verbose=1,
    )

    return final_model, final_feature_scaler, final_target_scaler


def build_future_calendar(start=config.FORECAST_START, end=config.FORECAST_END) -> pd.DataFrame:
    future_dates = pd.date_range(start=start, end=end, freq="D")
    future = pd.DataFrame({config.DATE_COL: future_dates})

    future["day_of_week"] = future[config.DATE_COL].dt.dayofweek
    future["day_of_year"] = future[config.DATE_COL].dt.dayofyear

    future["dow_sin"] = np.sin(2 * np.pi * future["day_of_week"] / 7)
    future["dow_cos"] = np.cos(2 * np.pi * future["day_of_week"] / 7)

    future["doy_sin"] = np.sin(2 * np.pi * (future["day_of_year"] - 1) / 365.25)
    future["doy_cos"] = np.cos(2 * np.pi * (future["day_of_year"] - 1) / 365.25)

    return future


def recursive_forecast(final_model, final_feature_scaler, final_target_scaler,
                        df_model: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    history_window_original = df_model[config.FEATURE_COLS].tail(config.TIMESTEPS).copy()
    history_window_scaled = final_feature_scaler.transform(history_window_original).astype(np.float32)

    future_predictions_scaled = []

    for _, row in future.iterrows():
        X_future = history_window_scaled.reshape(1, config.TIMESTEPS, len(config.FEATURE_COLS))

        next_sales_scaled = final_model.predict(X_future, verbose=0)[0, 0]
        future_predictions_scaled.append(next_sales_scaled)

        next_sales_original = final_target_scaler.inverse_transform(
            np.array([[next_sales_scaled]])
        )[0, 0]

        next_row_original = pd.DataFrame({
            "Sales": [next_sales_original],
            "dow_sin": [row["dow_sin"]],
            "dow_cos": [row["dow_cos"]],
            "doy_sin": [row["doy_sin"]],
            "doy_cos": [row["doy_cos"]],
        })

        next_row_scaled = final_feature_scaler.transform(next_row_original[config.FEATURE_COLS]).astype(np.float32)

        history_window_scaled = np.vstack([history_window_scaled[1:], next_row_scaled])

    future_sales = final_target_scaler.inverse_transform(
        np.array(future_predictions_scaled).reshape(-1, 1)
    ).ravel()
    future_sales = np.maximum(future_sales, 0)  # Sales cannot be negative

    return pd.DataFrame({config.DATE_COL: future[config.DATE_COL], "Predicted Sales": future_sales})


def summarize_and_export(forecast: pd.DataFrame):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    total = forecast["Predicted Sales"].sum()
    average = forecast["Predicted Sales"].mean()
    highest = forecast.loc[forecast["Predicted Sales"].idxmax()]
    lowest = forecast.loc[forecast["Predicted Sales"].idxmin()]

    print(f"Projected 2017 Sales : {total:,.2f}")
    print(f"Average Daily Sales  : {average:,.2f}")
    print(f"Highest Forecast Day : {highest[config.DATE_COL].date()} | {highest['Predicted Sales']:,.2f}")
    print(f"Lowest Forecast Day  : {lowest[config.DATE_COL].date()} | {lowest['Predicted Sales']:,.2f}")

    forecast[[config.DATE_COL, "Predicted Sales"]].to_csv(config.DAILY_FORECAST_FILE, index=False)

    forecast = forecast.copy()
    forecast["Month"] = forecast[config.DATE_COL].dt.month
    forecast["Month Name"] = forecast[config.DATE_COL].dt.strftime("%B")
    monthly = (
        forecast.groupby(["Month", "Month Name"], as_index=False)["Predicted Sales"]
        .sum()
        .sort_values("Month")
    )
    monthly.to_csv(config.MONTHLY_FORECAST_FILE, index=False)

    print(f"\nWrote {config.DAILY_FORECAST_FILE}")
    print(f"Wrote {config.MONTHLY_FORECAST_FILE}")

    return monthly


def run():
    df_model = load_full_history()
    best_epoch = load_best_epoch()

    final_model, final_feature_scaler, final_target_scaler = refit_on_full_history(df_model, best_epoch)

    future = build_future_calendar()
    forecast = recursive_forecast(final_model, final_feature_scaler, final_target_scaler, df_model, future)

    summarize_and_export(forecast)

    final_model.save(config.FINAL_MODEL_PATH)
    with open(config.FINAL_FEATURE_SCALER_PATH, "wb") as f:
        pickle.dump(final_feature_scaler, f)
    with open(config.FINAL_TARGET_SCALER_PATH, "wb") as f:
        pickle.dump(final_target_scaler, f)

    print(f"\nSaved final model -> {config.FINAL_MODEL_PATH}")


if __name__ == "__main__":
    run()
