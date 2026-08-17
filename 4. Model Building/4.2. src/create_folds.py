"""
create_folds.py

Turns the raw processed sales CSV into the two files train.py and
inference.py actually consume: input/train.csv and input/test.csv.

Responsibilities (pulled from notebook sections 3-7):
    1. Load Sales_Processed.csv
    2. Data-quality checks (missing values, duplicate/non-daily dates)
    3. Build calendar features (day-of-week / day-of-year, cyclically encoded)
    4. Chronological 70/15/15 split — NO shuffling, this is time series

train.csv contains the train + validation rows, distinguished by a
`split` column ("train" / "valid"), so train.py can filter it in one
place instead of re-deriving the split boundaries itself.
test.csv is the untouched final holdout period.

Run directly:
    python -m src.create_folds
"""

import numpy as np
import pandas as pd

import config


def load_raw(path: str = config.RAW_DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")
    df = df.sort_values(config.DATE_COL).reset_index(drop=True)
    return df


def run_data_quality_checks(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    print("Missing values:")
    print(missing[missing > 0])

    print("\nDuplicate dates:", df[config.DATE_COL].duplicated().sum())
    print("Start date:", df[config.DATE_COL].min())
    print("End date:", df[config.DATE_COL].max())

    date_diff = df[config.DATE_COL].diff().dropna()
    print("\nNon-daily gaps:")
    print(date_diff[date_diff != pd.Timedelta(days=1)].value_counts())

    print("\nSales summary:")
    print(df[config.TARGET_COL].describe())


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sales + calendar-only features — safe for forecasting because
    calendar position is known in advance, unlike Quantity Sold,
    Promotion Day, Cost Of Sales, etc."""
    df_model = df[[config.DATE_COL, config.TARGET_COL]].copy()

    df_model["day_of_week"] = df_model[config.DATE_COL].dt.dayofweek
    df_model["day_of_year"] = df_model[config.DATE_COL].dt.dayofyear

    df_model["dow_sin"] = np.sin(2 * np.pi * df_model["day_of_week"] / 7)
    df_model["dow_cos"] = np.cos(2 * np.pi * df_model["day_of_week"] / 7)

    df_model["doy_sin"] = np.sin(2 * np.pi * (df_model["day_of_year"] - 1) / 365.25)
    df_model["doy_cos"] = np.cos(2 * np.pi * (df_model["day_of_year"] - 1) / 365.25)

    return df_model


def chronological_split(df_model: pd.DataFrame):
    """70/15/15 chronological split. No shuffling — the model learns
    past -> future, and test must be a later, untouched period."""
    n = len(df_model)
    train_end = int(n * config.TRAIN_FRAC)
    val_end = int(n * config.VAL_FRAC)

    train_df = df_model.iloc[:train_end].copy()
    val_df = df_model.iloc[train_end:val_end].copy()
    test_df = df_model.iloc[val_end:].copy()

    train_df["split"] = "train"
    val_df["split"] = "valid"

    print("TRAIN:", train_df[config.DATE_COL].min(), "to", train_df[config.DATE_COL].max())
    print("VALID:", val_df[config.DATE_COL].min(), "to", val_df[config.DATE_COL].max())
    print("TEST :", test_df[config.DATE_COL].min(), "to", test_df[config.DATE_COL].max())

    return train_df, val_df, test_df


def main():
    df = load_raw()
    run_data_quality_checks(df)

    df_model = add_calendar_features(df)

    train_df, val_df, test_df = chronological_split(df_model)

    train_valid_df = pd.concat([train_df, val_df], ignore_index=True)
    train_valid_df.to_csv(config.TRAINING_FILE, index=False)
    test_df.to_csv(config.TEST_FILE, index=False)

    print(f"\nWrote {config.TRAINING_FILE} ({len(train_valid_df)} rows, train+valid)")
    print(f"Wrote {config.TEST_FILE} ({len(test_df)} rows, holdout)")


if __name__ == "__main__":
    main()
