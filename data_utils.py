import pandas as pd


def load_replication_data(path="data/replication_dataset.csv"):
    data = pd.read_csv(path, index_col=0, parse_dates=True)

    target = "y_unrate_change_1m_ahead"

    X = data.drop(columns=[target])
    y = data[target]

    return X, y


def standardize_train_forecast(X_train_raw, X_forecast_raw):
    train_mean = X_train_raw.mean()
    train_std = X_train_raw.std()

    # Prevent division by zero or near-zero std
    train_std = train_std.replace(0, 1)
    train_std = train_std.mask(train_std < 1e-6, 1)

    X_train = (X_train_raw - train_mean) / train_std
    X_forecast = (X_forecast_raw - train_mean) / train_std

    return X_train, X_forecast