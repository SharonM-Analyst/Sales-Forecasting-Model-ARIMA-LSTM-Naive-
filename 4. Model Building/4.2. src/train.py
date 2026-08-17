"""
train.py

The model-selection stage (notebook sections 8-15):
    1. Load train.csv / test.csv from create_folds.py, re-split train
       into train/valid using the `split` flag
    2. Scale features/target — scaler fit on TRAIN ONLY (no leakage)
    3. Build 7-day LSTM sequences
    4. Train with early stopping + LR reduction on plateau
    5. Evaluate on validation + test, compare against naive/seasonal-
       naive/ARIMA/SARIMAX baselines
    6. Save the model, scalers, and the best epoch (needed later by
       inference.py for the full-history refit)

Run directly:
    python -m src.train --model lstm
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
import model_dispatcher
import models


def load_split_data():
    train_valid_df = pd.read_csv(config.TRAINING_FILE, parse_dates=[config.DATE_COL])
    test_df = pd.read_csv(config.TEST_FILE, parse_dates=[config.DATE_COL])

    train_df = train_valid_df[train_valid_df["split"] == "train"].drop(columns="split")
    val_df = train_valid_df[train_valid_df["split"] == "valid"].drop(columns="split")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


def scale_without_leakage(train_df, val_df, test_df):
    feature_scaler = MinMaxScaler()
    train_features = feature_scaler.fit_transform(train_df[config.FEATURE_COLS]).astype(np.float32)
    val_features = feature_scaler.transform(val_df[config.FEATURE_COLS]).astype(np.float32)
    test_features = feature_scaler.transform(test_df[config.FEATURE_COLS]).astype(np.float32)

    target_scaler = MinMaxScaler()
    train_target = target_scaler.fit_transform(train_df[[config.TARGET_COL]]).astype(np.float32)
    val_target = target_scaler.transform(val_df[[config.TARGET_COL]]).astype(np.float32)
    test_target = target_scaler.transform(test_df[[config.TARGET_COL]]).astype(np.float32)

    return (feature_scaler, target_scaler,
            train_features, train_target,
            val_features, val_target,
            test_features, test_target)


def build_sequences_with_context(train_features, train_target,
                                  val_features, val_target,
                                  test_features, test_target):
    """Validation/test each borrow the previous TIMESTEPS rows from the
    prior split so their first window is complete, exactly like the
    notebook's val_features_with_context / test_features_with_context."""
    X_train, y_train = models.create_sequences(train_features, train_target, config.TIMESTEPS)

    val_features_ctx = np.vstack([train_features[-config.TIMESTEPS:], val_features])
    val_target_ctx = np.vstack([train_target[-config.TIMESTEPS:], val_target])
    X_val, y_val = models.create_sequences(val_features_ctx, val_target_ctx, config.TIMESTEPS)

    test_features_ctx = np.vstack([val_features[-config.TIMESTEPS:], test_features])
    test_target_ctx = np.vstack([val_target[-config.TIMESTEPS:], test_target])
    X_test, y_test = models.create_sequences(test_features_ctx, test_target_ctx, config.TIMESTEPS)

    return X_train, y_train, X_val, y_val, X_test, y_test


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
    print(f"R2  :  {r2:.4f}\n")

    return actual, pred, mae, rmse, r2


def compare_against_baselines(train_df, test_df, test_mae, test_rmse, test_r2):
    val_end = len(train_df)  # train_df here is train+valid combined length
    n = val_end + len(test_df)
    # Rebuild the combined chronological frame naive baselines index into.
    df_model = pd.concat([train_df, test_df], ignore_index=True)

    naive_pred = model_dispatcher.naive_previous_day(df_model, val_end, n)
    seasonal_pred = model_dispatcher.seasonal_naive_previous_week(df_model, val_end, n)
    test_actual = test_df[config.TARGET_COL].to_numpy()

    rows = [
        ("Naive - Previous Day", naive_pred),
        ("Seasonal Naive - Previous Week", seasonal_pred),
    ]

    results = {"Model": ["LSTM"], "MAE": [test_mae], "RMSE": [test_rmse], "R2": [test_r2]}
    for name, pred in rows:
        results["Model"].append(name)
        results["MAE"].append(mean_absolute_error(test_actual, pred))
        results["RMSE"].append(np.sqrt(mean_squared_error(test_actual, pred)))
        results["R2"].append(r2_score(test_actual, pred))

    try:
        train_series = train_df[config.TARGET_COL]
        arima_pred = model_dispatcher.forecast_arima(train_series, steps=len(test_df))
        sarimax_pred = model_dispatcher.forecast_sarimax(train_series, steps=len(test_df))

        for name, pred in [("ARIMA", arima_pred), ("SARIMAX", sarimax_pred)]:
            results["Model"].append(name)
            results["MAE"].append(mean_absolute_error(test_actual, pred))
            results["RMSE"].append(np.sqrt(mean_squared_error(test_actual, pred)))
            results["R2"].append(r2_score(test_actual, pred))
    except Exception as exc:  # ARIMA/SARIMAX can fail to converge on small data
        print(f"ARIMA/SARIMAX skipped: {exc}")

    comparison = pd.DataFrame(results)
    print(comparison)
    return comparison


def run(model_name: str = "lstm"):
    train_df, val_df, test_df = load_split_data()

    (feature_scaler, target_scaler,
     train_features, train_target,
     val_features, val_target,
     test_features, test_target) = scale_without_leakage(train_df, val_df, test_df)

    X_train, y_train, X_val, y_val, X_test, y_test = build_sequences_with_context(
        train_features, train_target, val_features, val_target, test_features, test_target
    )

    build_fn = model_dispatcher.MODELS[model_name]
    model = build_fn()
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE, min_lr=config.MIN_LR, verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print("Best validation epoch:", best_epoch)

    evaluate_model(model, X_val, y_val, target_scaler, "VALIDATION")
    _, _, test_mae, test_rmse, test_r2 = evaluate_model(model, X_test, y_test, target_scaler, "TEST")

    compare_against_baselines(pd.concat([train_df, val_df], ignore_index=True), test_df,
                               test_mae, test_rmse, test_r2)

    model.save(config.MODEL_PATH)
    with open(config.FEATURE_SCALER_PATH, "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(config.TARGET_SCALER_PATH, "wb") as f:
        pickle.dump(target_scaler, f)
    with open(config.BEST_EPOCH_PATH, "wb") as f:
        pickle.dump(best_epoch, f)

    print(f"\nSaved model -> {config.MODEL_PATH}")
    print(f"Saved scalers -> {config.FEATURE_SCALER_PATH}, {config.TARGET_SCALER_PATH}")
    print(f"Saved best_epoch -> {config.BEST_EPOCH_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", choices=list(model_dispatcher.MODELS.keys()))
    args = parser.parse_args()
    run(args.model)
