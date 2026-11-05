"""Expanding-window preprocessing for the unemployment quantile replication."""
import pandas as pd


def load_replication_data(path="data/replication_dataset.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    target = "y_unrate_change_1m_ahead"
    return df.drop(columns=[target]), df[target]


def standardize_train_forecast(X_train_raw, X_forecast_raw):
    """Use only the expanding training window's columnwise mean/std."""
    train_mean = X_train_raw.mean(axis=0)
    train_std = X_train_raw.std(axis=0, ddof=1)  # torch.std default in fred.py
    if train_std.isna().any() or (train_std == 0).any():
        raise ValueError("Zero/undefined predictor std in expanding window")
    X_train = X_train_raw.sub(train_mean, axis=1).div(train_std, axis=1)
    X_forecast = X_forecast_raw.sub(train_mean, axis=1).div(train_std, axis=1)
    return X_train, X_forecast


def standardize_target_train_forecast(y_train_raw, y_actual_raw):
    """Match fred.py: recursively demean y, without dividing by its std."""
    y_mean = float(y_train_raw.mean())
    y_std = 1.0
    y_train_std = y_train_raw - y_mean
    y_actual_std = y_actual_raw - y_mean
    return y_train_std, y_actual_std, y_mean, y_std
