import pandas as pd


def load_replication_data(path="data/replication_dataset.csv"):
    data = pd.read_csv(path, index_col=0, parse_dates=True)

    target = "y_unrate_change_1m_ahead"

    drop_columns = [
        target,
        "TOTRESNS",
        "NONBORRES",
    ]

    X = data.drop(columns=drop_columns, errors="ignore")
    y = data[target]

    return X, y

def standardize_train_forecast(X_train_raw, X_forecast_raw):
    """
    Standardize each predictor using expanding-window training statistics.

    For each predictor column, calculate its mean and standard deviation
    using only the training sample available before the forecast date.

    The forecast observation is standardized using those SAME training
    statistics, avoiding look-ahead bias.
    """

    # Mean and std through time for each predictor
    train_mean = X_train_raw.mean(axis=0)
    train_std = X_train_raw.std(axis=0)

    # Prevent division by zero / near-zero variance
    train_std = train_std.mask(train_std < 1e-6, 1.0)

    # Standardize training observations
    X_train = X_train_raw.sub(train_mean, axis=1)
    X_train = X_train.div(train_std, axis=1)

    # Standardize forecast observation using TRAINING statistics
    X_forecast = X_forecast_raw.sub(train_mean, axis=1)
    X_forecast = X_forecast.div(train_std, axis=1)

    return X_train, X_forecast

def standardize_target_train_forecast(y_train_raw, y_actual_raw):
    """
    Standardize the target using only the expanding training sample.

    Returns:
        y_train_std
        y_actual_std
        y_mean
        y_std
    """

    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std()

    if pd.isna(y_std) or y_std < 1e-6:
        y_std = 1.0

    y_train_std = (y_train_raw - y_mean) / y_std
    y_actual_std = (y_actual_raw - y_mean) / y_std

    return y_train_std, y_actual_std, y_mean, y_std