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
    Cross-sectionally standardize each month across predictors.

    Each row/month has approximately:
        mean = 0
        standard deviation = 1
    """

    # Mean and std across predictors within each training month
    train_row_mean = X_train_raw.mean(axis=1)
    train_row_std = X_train_raw.std(axis=1)

    # Prevent division by zero or near-zero values
    train_row_std = train_row_std.mask(train_row_std < 1e-6, 1.0)

    # axis=0 tells pandas to match the Series to DataFrame rows
    X_train = X_train_raw.sub(train_row_mean, axis=0)
    X_train = X_train.div(train_row_std, axis=0)

    # Normalize the forecast month using its own cross-sectional statistics
    forecast_row_mean = X_forecast_raw.mean(axis=1)
    forecast_row_std = X_forecast_raw.std(axis=1)
    forecast_row_std = forecast_row_std.mask(
        forecast_row_std < 1e-6,
        1.0
    )

    X_forecast = X_forecast_raw.sub(forecast_row_mean, axis=0)
    X_forecast = X_forecast.div(forecast_row_std, axis=0)

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