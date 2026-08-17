# Databricks notebook source
# MAGIC %md
# MAGIC # Training Code. Build The LSTM Model

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
# MAGIC # ARIMA & SARIMA MODELS

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

# MAGIC %md
# MAGIC # Naive Baseline Model

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
