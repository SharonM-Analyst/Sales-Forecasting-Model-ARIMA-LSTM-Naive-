# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 2017 Sales Forecasting — LSTM Neural Network
# MAGIC
# MAGIC ## Dataset
# MAGIC
# MAGIC This notebook uses the provided processed dataset:
# MAGIC
# MAGIC `Sales_Processed.csv`
# MAGIC
# MAGIC The dataset contains **1,053 daily observations** from **2013-12-30 to 2016-11-16**.
# MAGIC
# MAGIC The objective is to build and evaluate an LSTM forecasting model and then generate a **daily Sales forecast for 2017**.
# MAGIC
# MAGIC ### Important ML design decision
# MAGIC
# MAGIC For the 2017 forecast, future values such as:
# MAGIC
# MAGIC - Cost Of Sales
# MAGIC - Quantity Sold
# MAGIC - Promotion Day
# MAGIC - Promotion Group
# MAGIC - Gross Profit
# MAGIC - Rolling price metrics
# MAGIC
# MAGIC are not known.
# MAGIC
# MAGIC Therefore, the production forecast model uses:
# MAGIC
# MAGIC 1. Historical Sales
# MAGIC 2. Calendar variables that are known in advance
# MAGIC
# MAGIC The model uses a **7-day lookback window** and recursively forecasts one day at a time.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Install Parckages

# COMMAND ----------

# MAGIC %pip install tensorflow  openpyxl pydot statsmodels

# COMMAND ----------

# MAGIC %md
# MAGIC ## Restart Kernel

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Import Libraries

# COMMAND ----------


import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Load Dataset

# COMMAND ----------

file_path = "/Workspace/Repos/sshanay92@gmail.com/Sales-Forecasting-Model-ARIMA-LSTM-Naive-/1. Raw Data & Project Description/Sales_Processed.csv"

df = pd.read_csv(file_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Data Qulity Checks

# COMMAND ----------


# Convert Date to datetime.
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Sort chronologically.
df = df.sort_values("Date").reset_index(drop=True)

# Check missing values.
missing = df.isna().sum().sort_values(ascending=False)

print("Missing values:")
print(missing[missing > 0])

print("\nDuplicate dates:", df["Date"].duplicated().sum())

print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())

# Check that observations are consecutive daily observations.
date_diff = df["Date"].diff().dropna()

print("\nNon-daily gaps:")
print(date_diff[date_diff != pd.Timedelta(days=1)].value_counts())

print("\nSales summary:")
print(df["Sales"].describe())


# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Visualize Historical Sales
# MAGIC

# COMMAND ----------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Sales"],
        mode="lines",
        name="Historical Sales"
    )
)

fig.update_layout(
    title="Historical Daily Sales",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    height=600
)

fig.show()


# COMMAND ----------

# Sales distribution and skewness

cols = ["Sales", "Cost Of Sales", "Quantity Sold", "Gross Profit"]

print("Skewness:")
display(df[cols].skew())

for col in cols:
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        marginal="box",
        title=f"{col} Distribution | Skewness = {df[col].skew():.2f}"
    )
    fig.update_layout(template="plotly_white", height=500)
    fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Create Forecasting Features

# COMMAND ----------


# We use Sales plus calendar variables that are known for future dates.
# We intentionally do NOT use future-dependent business variables
# such as Quantity Sold, Promotion Day, Gross Profit, or Cost Of Sales.

df_model = df[["Date", "Sales"]].copy()

# Day of week: Monday=0 ... Sunday=6
df_model["day_of_week"] = df_model["Date"].dt.dayofweek

# Day of year: 1 ... 365/366
df_model["day_of_year"] = df_model["Date"].dt.dayofyear

# Cyclical weekly encoding.
df_model["dow_sin"] = np.sin(2 * np.pi * df_model["day_of_week"] / 7)
df_model["dow_cos"] = np.cos(2 * np.pi * df_model["day_of_week"] / 7)

# Cyclical annual encoding.
df_model["doy_sin"] = np.sin(2 * np.pi * (df_model["day_of_year"] - 1) / 365.25)
df_model["doy_cos"] = np.cos(2 * np.pi * (df_model["day_of_year"] - 1) / 365.25)

df_model.head()


# COMMAND ----------

# Save the model dataset to CSV
df_model.to_csv("/Workspace/Repos/sshanay92@gmail.com/Sales-Forecasting-Model-ARIMA-LSTM-Naive-/4. Model Building/model dataset/Sales_Model_Features.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why these features are safe for 2017
# MAGIC
# MAGIC Calendar variables are known before the forecast is made.
# MAGIC
# MAGIC For example, before 1 January 2017 we already know:
# MAGIC
# MAGIC - its day of week
# MAGIC - its position in the year
# MAGIC - its seasonal/cyclical position
# MAGIC
# MAGIC By contrast, we do **not** know 1 January 2017's:
# MAGIC
# MAGIC - Quantity Sold
# MAGIC - Cost Of Sales
# MAGIC - Promotion Day
# MAGIC
# MAGIC unless those variables are separately forecast or provided by a business plan.
# MAGIC
# MAGIC This prevents future-data leakage.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Chronological Train/ Validation/ Test Split

# COMMAND ----------


# 70% train, 15% validation, 15% test.
n = len(df_model)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df_model.iloc[:train_end].copy()
val_df = df_model.iloc[train_end:val_end].copy()
test_df = df_model.iloc[val_end:].copy()

print("TRAIN:")
print(train_df["Date"].min(), "to", train_df["Date"].max())

print("\nVALIDATION:")
print(val_df["Date"].min(), "to", val_df["Date"].max())

print("\nTEST:")
print(test_df["Date"].min(), "to", test_df["Date"].max())


# COMMAND ----------

# Save the Splitted dataset to CSV
train_df.to_csv("/Workspace/Repos/sshanay92@gmail.com/Sales-Forecasting-Model-ARIMA-LSTM-Naive-/4. Model Building/4.1 Input/train.csv", index=False)

val_df.to_csv("/Workspace/Repos/sshanay92@gmail.com/Sales-Forecasting-Model-ARIMA-LSTM-Naive-/4. Model Building/4.1 Input/val.csv", index=False)

test_df.to_csv("/Workspace/Repos/sshanay92@gmail.com/Sales-Forecasting-Model-ARIMA-LSTM-Naive-/4. Model Building/4.1 Input/test.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time-series rule
# MAGIC
# MAGIC We do **not** randomly shuffle the observations.
# MAGIC
# MAGIC The model learns:
# MAGIC
# MAGIC `past → future`
# MAGIC
# MAGIC and the test set represents a later historical period that was not used to fit the final training parameters.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Scale Features Without Leakage

# COMMAND ----------

feature_cols = [
    "Sales",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos"
]

target_col = "Sales"

# Fit feature scaler ONLY on training data.
feature_scaler = MinMaxScaler()
train_features = feature_scaler.fit_transform(train_df[feature_cols]).astype(np.float32)

# Transform validation/test using the training scaler.
val_features = feature_scaler.transform(val_df[feature_cols]).astype(np.float32)
test_features = feature_scaler.transform(test_df[feature_cols]).astype(np.float32)

# Separate target scaler so predictions can be returned to Sales units.
target_scaler = MinMaxScaler()

train_target = target_scaler.fit_transform(train_df[[target_col]]).astype(np.float32)
val_target = target_scaler.transform(val_df[[target_col]]).astype(np.float32)
test_target = target_scaler.transform(test_df[[target_col]]).astype(np.float32)

print("Train features:", train_features.shape)
print("Train target:", train_target.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Create 7-Day LSTM Sequences

# COMMAND ----------

TIMESTEPS = 7

def create_sequences(features, target, timesteps=7):
    X = []
    y = []

    for i in range(timesteps, len(features)):
        X.append(features[i - timesteps:i])
        y.append(target[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


# Training sequences.
X_train, y_train = create_sequences(
    train_features,
    train_target,
    TIMESTEPS
)

# For validation and test, include the previous 7 observations
# as context so the first validation/test day has a valid window.
val_features_with_context = np.vstack([train_features[-TIMESTEPS:],val_features])
val_target_with_context = np.vstack([train_target[-TIMESTEPS:],val_target])

test_features_with_context = np.vstack([val_features[-TIMESTEPS:],test_features])
test_target_with_context = np.vstack([val_target[-TIMESTEPS:],test_target])

X_val, y_val = create_sequences(
    val_features_with_context,
    val_target_with_context,
    TIMESTEPS
)

X_test, y_test = create_sequences(
    test_features_with_context,
    test_target_with_context,
    TIMESTEPS
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC ### LSTM input shape
# MAGIC
# MAGIC The model receives:
# MAGIC
# MAGIC `(samples, 7, 5)`
# MAGIC
# MAGIC where:
# MAGIC
# MAGIC - `samples` = number of training sequences
# MAGIC - `7` = previous seven days
# MAGIC - `5` = Sales + four calendar features
# MAGIC
# MAGIC The target is:
# MAGIC
# MAGIC `next day's Sales`
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 10. Build The LSTM Model

# COMMAND ----------

model = Sequential([
    Input(shape=(TIMESTEPS, len(feature_cols))),

    LSTM(64),

    Dropout(0.20),

    Dense(32, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# COMMAND ----------

# MAGIC %md
# MAGIC # 11. Train The LSTM

# COMMAND ----------


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# COMMAND ----------

# MAGIC %md
# MAGIC # 12. TRAINING / VALIDATION LOSS

# COMMAND ----------


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=history.history["loss"],
        mode="lines",
        name="Training Loss"
    )
)

fig.add_trace(
    go.Scatter(
        y=history.history["val_loss"],
        mode="lines",
        name="Validation Loss"
    )
)

fig.update_layout(
    title="LSTM Training and Validation Loss",
    xaxis_title="Epoch",
    yaxis_title="MSE",
    template="plotly_white",
    height=550
)

fig.show()

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

print("Best validation epoch:", best_epoch)


# COMMAND ----------

# MAGIC %md
# MAGIC # 13. EVALUATE ON VALIDATION AND TEST DATA

# COMMAND ----------

def evaluate_model(model, X, y, scaler, label):
    pred_scaled = model.predict(X, verbose=0)

    actual = scaler.inverse_transform(y).ravel()
    pred = scaler.inverse_transform(pred_scaled).ravel()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)

    print(f"--- {label} ---")
    print(f"MAE :  {mae:,.2f}")
    print(f"RMSE:  {rmse:,.2f}")
    print(f"R²  :  {r2:.4f}")
    print()

    return actual, pred, mae, rmse, r2


val_actual, val_pred, val_mae, val_rmse, val_r2 = evaluate_model(
    model,
    X_val,
    y_val,
    target_scaler,
    "VALIDATION"
)

test_actual, test_pred, test_mae, test_rmse, test_r2 = evaluate_model(
    model,
    X_test,
    y_test,
    target_scaler,
    "TEST"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison LSTM, ARIMA, NAIVE BASELINE, SARIMAX

# COMMAND ----------

# MAGIC %md
# MAGIC # 14. NAIVE BASELINES

# COMMAND ----------

# MAGIC %md
# MAGIC - Sets a score to beat.
# MAGIC - Proves if a complex system adds real value.
# MAGIC - Catches basic errors in data setup.
# MAGIC - Offers a fast sanity check.

# COMMAND ----------


# Baseline 1:
# Tomorrow = today's Sales

test_actual_original = test_df["Sales"].to_numpy()

naive_1_pred = df_model["Sales"].iloc[
    val_end - 1 : n - 1
].to_numpy()

naive_1_mae = mean_absolute_error(
    test_actual_original,
    naive_1_pred
)

naive_1_rmse = np.sqrt(
    mean_squared_error(
        test_actual_original,
        naive_1_pred
    )
)

naive_1_r2 = r2_score(
    test_actual_original,
    naive_1_pred
)


# Baseline 2:
# Tomorrow = Sales from the same weekday one week earlier.

# Because the data is daily and continuous, this is a useful
# seasonal-naive benchmark for a 7-day forecasting problem.

seasonal_naive_pred = df_model["Sales"].iloc[
    val_end - 7 : n - 7
].to_numpy()

seasonal_naive_mae = mean_absolute_error(
    test_actual_original,
    seasonal_naive_pred
)

seasonal_naive_rmse = np.sqrt(
    mean_squared_error(
        test_actual_original,
        seasonal_naive_pred
    )
)

seasonal_naive_r2 = r2_score(
    test_actual_original,
    seasonal_naive_pred
)


comparison = pd.DataFrame({
    "Model": [
        "Naive - Previous Day",
        "Seasonal Naive - Previous Week",
        "LSTM"
    ],
    "MAE": [
        naive_1_mae,
        seasonal_naive_mae,
        test_mae
    ],
    "RMSE": [
        naive_1_rmse,
        seasonal_naive_rmse,
        test_rmse
    ],
    "R2": [
        naive_1_r2,
        seasonal_naive_r2,
        test_r2
    ]
})

comparison


# COMMAND ----------

# MAGIC %md
# MAGIC - LSTM PERFORMED BETTER THAN Naive baseline 

# COMMAND ----------

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prepare training series for ARIMA/SARIMAX
train_series = train_df["Sales"]

# ARIMA example (order=(p,d,q))
arima_model = ARIMA(train_series, order=(5,1,0))
arima_fit = arima_model.fit()

# SARIMAX example (order=(p,d,q), seasonal_order=(P,D,Q,s))
sarimax_model = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,7))
sarimax_fit = sarimax_model.fit()

# Forecast for test period
arima_forecast = arima_fit.forecast(steps=len(test_df))
sarimax_forecast = sarimax_fit.forecast(steps=len(test_df))

# Evaluate ARIMA
arima_mae = mean_absolute_error(test_df["Sales"], arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_df["Sales"], arima_forecast))
arima_r2 = r2_score(test_df["Sales"], arima_forecast)

# Evaluate SARIMAX
sarimax_mae = mean_absolute_error(test_df["Sales"], sarimax_forecast)
sarimax_rmse = np.sqrt(mean_squared_error(test_df["Sales"], sarimax_forecast))
sarimax_r2 = r2_score(test_df["Sales"], sarimax_forecast)

# Comparison DataFrame
arima_comparison = pd.DataFrame({
    "Model": ["ARIMA", "SARIMAX"],
    "MAE": [arima_mae, sarimax_mae],
    "RMSE": [arima_rmse, sarimax_rmse],
    "R2": [arima_r2, sarimax_r2]
})

display(arima_comparison)

# COMMAND ----------

comparison_metrics = pd.DataFrame({
    "Model": [
        "Naive - Previous Day",
        "Seasonal Naive - Previous Week",
        "LSTM",
        "ARIMA",
        "SARIMAX"
    ],
    "MAE": [
        round(naive_1_mae, 3),
        round(seasonal_naive_mae, 3),
        round(test_mae, 3),
        round(arima_mae, 3),
        round(sarimax_mae, 3)
    ],
    "RMSE": [
        round(naive_1_rmse, 3),
        round(seasonal_naive_rmse, 3),
        round(test_rmse, 3),
        round(arima_rmse, 3),
        round(sarimax_rmse, 3)
    ],
    "R2": [
        round(naive_1_r2, 3),
        round(seasonal_naive_r2, 3),
        round(test_r2, 3),
        round(arima_r2, 3),
        round(sarimax_r2, 3)
    ]
})

display(comparison_metrics)

# COMMAND ----------

fig_mae_rmse = go.Figure()

fig_mae_rmse.add_trace(
    go.Bar(
        x=comparison_metrics["Model"],
        y=comparison_metrics["MAE"],
        name="MAE"
    )
)

fig_mae_rmse.add_trace(
    go.Bar(
        x=comparison_metrics["Model"],
        y=comparison_metrics["RMSE"],
        name="RMSE"
    )
)

fig_mae_rmse.update_layout(
    barmode="stack",
    title="Stacked MAE & RMSE by Model",
    xaxis_title="Model",
    yaxis_title="Metric Value",
    template="plotly_white",
    height=500
)

fig_mae_rmse.show()

fig_r2 = go.Figure()

fig_r2.add_trace(
    go.Bar(
        x=comparison_metrics["Model"],
        y=comparison_metrics["R2"],
        name="R2"
    )
)

fig_r2.update_layout(
    title="R2 Comparison by Model",
    xaxis_title="Model",
    yaxis_title="R2",
    template="plotly_white",
    height=500
)

fig_r2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 15. ACTUAL VS PREDICTED TEST SALES

# COMMAND ----------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=test_df["Date"],
        y=test_actual,
        mode="lines",
        name="Actual Sales"
    )
)

fig.add_trace(
    go.Scatter(
        x=test_df["Date"],
        y=test_pred,
        mode="lines",
        name="LSTM Prediction"
    )
)

fig.add_trace(
    go.Scatter(
        x=test_df["Date"],
        y=naive_1_pred,
        mode="lines",
        name="Naive - Previous Day"
    )
)

fig.add_trace(
    go.Scatter(
        x=test_df["Date"],
        y=seasonal_naive_pred,
        mode="lines",
        name="Seasonal Naive - Previous Week"
    )
)

fig.update_layout(
    title="Test Period — Actual vs LSTM vs Naive Baselines",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    height=600
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Final 2017 Forecast
# MAGIC
# MAGIC The model has now been evaluated on a historical test period.
# MAGIC
# MAGIC For the final forecast, we:
# MAGIC
# MAGIC 1. Use all historical observations available in the CSV.
# MAGIC 2. Refit the scalers using all historical data.
# MAGIC 3. Rebuild the same LSTM architecture.
# MAGIC 4. Train for the best number of epochs selected from the validation experiment.
# MAGIC 5. Generate every date in 2017.
# MAGIC 6. Forecast recursively, one day at a time.
# MAGIC
# MAGIC The final forecast does not use future `Quantity Sold`, `Cost Of Sales`, promotions, or other unknown business variables.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 15. PREPARE ALL HISTORICAL DATA FOR FINAL MODEL

# COMMAND ----------


final_feature_scaler = MinMaxScaler()
final_target_scaler = MinMaxScaler()

all_features = final_feature_scaler.fit_transform(
    df_model[feature_cols]
).astype(np.float32)

all_target = final_target_scaler.fit_transform(
    df_model[[target_col]]
).astype(np.float32)

X_all, y_all = create_sequences(
    all_features,
    all_target,
    TIMESTEPS
)

print("Final X shape:", X_all.shape)
print("Final y shape:", y_all.shape)


# COMMAND ----------

# MAGIC %md
# MAGIC # 17. BUILD FINAL LSTM

# COMMAND ----------


final_model = Sequential([
    Input(shape=(TIMESTEPS, len(feature_cols))),
    LSTM(64),
    Dropout(0.20),
    Dense(32, activation="relu"),
    Dense(1)
])

final_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

final_model.summary()


# COMMAND ----------

# MAGIC %md
# MAGIC # 18. TRAIN FINAL MODEL

# COMMAND ----------


# Use the best epoch identified during the model-selection stage.
# We do not use a random split here.

final_model.fit(
    X_all,
    y_all,
    epochs=best_epoch,
    batch_size=32,
    shuffle=False,
    verbose=1
)


# COMMAND ----------

# MAGIC %md
# MAGIC # 19. CREATE 2017 CALENDAR FEATURES

# COMMAND ----------


future_dates = pd.date_range(
    start="2017-01-01",
    end="2017-12-31",
    freq="D"
)

future = pd.DataFrame({
    "Date": future_dates
})

future["day_of_week"] = future["Date"].dt.dayofweek
future["day_of_year"] = future["Date"].dt.dayofyear

future["dow_sin"] = np.sin(2 * np.pi * future["day_of_week"] / 7)
future["dow_cos"] = np.cos(2 * np.pi * future["day_of_week"] / 7)

future["doy_sin"] = np.sin(2 * np.pi * (future["day_of_year"] - 1) / 365.25)
future["doy_cos"] = np.cos(2 * np.pi * (future["day_of_year"] - 1) / 365.25)

future.head()


# COMMAND ----------

# MAGIC %md
# MAGIC # 20. RECURSIVE 2017 FORECAST

# COMMAND ----------


# Start with the final 7 historical observations in ORIGINAL units.
history_window_original = df_model[
    feature_cols
].tail(TIMESTEPS).copy()

# Scale the initial window using the final feature scaler.
history_window_scaled = final_feature_scaler.transform(
    history_window_original
).astype(np.float32)

future_predictions_scaled = []

for i, row in future.iterrows():

    # Current 7-day window.
    X_future = history_window_scaled.reshape(
        1,
        TIMESTEPS,
        len(feature_cols)
    )

    # Predict next day's Sales in SCALED units.
    next_sales_scaled = final_model.predict(
        X_future,
        verbose=0
    )[0, 0]

    future_predictions_scaled.append(
        next_sales_scaled
    )

    # Convert the predicted Sales back to ORIGINAL units.
    next_sales_original = final_target_scaler.inverse_transform(
        np.array([[next_sales_scaled]])
    )[0, 0]

    # Calendar features are known for the future date.
    next_row_original = pd.DataFrame({
        "Sales": [next_sales_original],
        "dow_sin": [row["dow_sin"]],
        "dow_cos": [row["dow_cos"]],
        "doy_sin": [row["doy_sin"]],
        "doy_cos": [row["doy_cos"]]
    })

    # Scale the complete next-day feature row correctly.
    next_row_scaled = final_feature_scaler.transform(
        next_row_original[feature_cols]
    ).astype(np.float32)

    # Remove the oldest day and append the new predicted day.
    history_window_scaled = np.vstack([
        history_window_scaled[1:],
        next_row_scaled
    ])


# Convert all predictions back to original Sales units.
future_sales = final_target_scaler.inverse_transform(
    np.array(future_predictions_scaled).reshape(-1, 1)
).ravel()

# Sales cannot be negative.
future_sales = np.maximum(future_sales, 0)

forecast_2017 = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_sales
})

forecast_2017.head(10)


# COMMAND ----------

# MAGIC %md
# MAGIC # 21. PLOT HISTORICAL SALES + 2017 FORECAST

# COMMAND ----------


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_model["Date"],
        y=df_model["Sales"],
        mode="lines",
        name="Historical Sales"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_2017["Date"],
        y=forecast_2017["Predicted Sales"],
        mode="lines",
        name="2017 LSTM Forecast"
    )
)

fig.update_layout(
    title="Historical Sales and 2017 LSTM Forecast",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    height=650
)

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # 22. Forecast Monthly Sales 2017

# COMMAND ----------


forecast_2017["Month"] = forecast_2017["Date"].dt.month
forecast_2017["Month Name"] = forecast_2017["Date"].dt.strftime("%B")

monthly_forecast = (
    forecast_2017
    .groupby(
        ["Month", "Month Name"],
        as_index=False
    )["Predicted Sales"]
    .sum()
    .sort_values("Month")
)

monthly_forecast


# COMMAND ----------

# Monthly Plotly visualization

fig = px.bar(
    monthly_forecast,
    x="Month Name",
    y="Predicted Sales",
    title="Projected Monthly Sales — 2017"
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Month",
    yaxis_title="Predicted Sales",
    height=550
)

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # 23. 2017 FORECAST SUMMARY

# COMMAND ----------


total_2017_sales = forecast_2017["Predicted Sales"].sum()
average_daily_sales = forecast_2017["Predicted Sales"].mean()

highest_day = forecast_2017.loc[
    forecast_2017["Predicted Sales"].idxmax()
]

lowest_day = forecast_2017.loc[
    forecast_2017["Predicted Sales"].idxmin()
]

print(f"Projected 2017 Sales : {total_2017_sales:,.2f}")
print(f"Average Daily Sales   : {average_daily_sales:,.2f}")

print(
    f"Highest Forecast Day  : "
    f"{highest_day['Date'].date()} | "
    f"{highest_day['Predicted Sales']:,.2f}"
)

print(
    f"Lowest Forecast Day   : "
    f"{lowest_day['Date'].date()} | "
    f"{lowest_day['Predicted Sales']:,.2f}"
)


# COMMAND ----------

# MAGIC %md
# MAGIC # 24. EXPORT DAILY AND MONTHLY FORECASTS

# COMMAND ----------


daily_output = forecast_2017[
    ["Date", "Predicted Sales"]
].copy()

daily_output.to_csv(
    "Sales_Forecast_2017_Daily.csv",
    index=False
)

monthly_forecast.to_csv(
    "Sales_Forecast_2017_Monthly.csv",
    index=False
)

print("Created:")
print("- Sales_Forecast_2017_Daily.csv")
print("- Sales_Forecast_2017_Monthly.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC # 25. SAVE THE FINAL MODEL

# COMMAND ----------


final_model.save(
    "Sales_LSTM_2017.keras"
)

print("Saved model: Sales_LSTM_2017.keras")

# COMMAND ----------

# MAGIC %md
# MAGIC # Saving Models

# COMMAND ----------

model_save_path = "/Workspace/Repos/sshanay92@gmail.com/Sales-Forecasting-Model-ARIMA-LSTM-Naive-/4. Model Building/4.3 models"

final_model.save(f"{model_save_path}/Sales_LSTM_2017.keras")

import joblib
joblib.dump(arima_model, f"{model_save_path}/Sales_ARIMA_2017.pkl")
joblib.dump(sarimax_model, f"{model_save_path}/Sales_SARIMAX_2017.pkl")

print("Saved all models to:")
print(model_save_path)