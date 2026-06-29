import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

data = pd.read_csv(
    "replication_dataset.csv",
    index_col=0,
    parse_dates=True
)

target = "y_unrate_change_1m_ahead"

X = data.drop(columns=[target])
y = data[target]

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

tau = 0.50

validation_start = "1980-01-01"
validation_end = "1999-12-01"
test_start = "2000-01-01"

# Start with a small lambda grid for speed/debugging
lambda_grid = [0.1, 1.0, 10.0]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def pinball_loss(y_true, y_pred, tau):
    error = y_true - y_pred
    return np.where(error >= 0, tau * error, (tau - 1) * error)


def objective(params, X_train, y_train, tau, lam):
    intercept = params[0]
    beta = params[1:]

    y_pred = intercept + X_train @ beta

    loss = pinball_loss(y_train, y_pred, tau).mean()

    # Do not penalize intercept
    ridge_penalty = lam * np.sum(beta ** 2)

    return loss + ridge_penalty


def standardize_train_forecast(X_train_raw, X_forecast_raw):
    train_mean = X_train_raw.mean()
    train_std = X_train_raw.std()

    # Avoid division by zero
    train_std = train_std.replace(0, 1)

    X_train = (X_train_raw - train_mean) / train_std
    X_forecast = (X_forecast_raw - train_mean) / train_std

    return X_train, X_forecast


def fit_and_forecast(date, lam, tau):
    """
    For a given forecast date:
    - train on all observations strictly before date
    - forecast at date
    """

    X_train_raw = X.loc[:date].iloc[:-1]
    y_train = y.loc[:date].iloc[:-1]

    X_forecast_raw = X.loc[[date]]

    X_train_std, X_forecast_std = standardize_train_forecast(
        X_train_raw,
        X_forecast_raw
    )

    X_train_np = X_train_std.to_numpy()
    y_train_np = y_train.to_numpy()
    X_forecast_np = X_forecast_std.to_numpy()

    initial_params = np.zeros(X_train_np.shape[1] + 1)

    # Start intercept at unconditional historical quantile
    initial_params[0] = np.quantile(y_train_np, tau)

    result = minimize(
    objective,
    initial_params,
    args=(X_train_np, y_train_np, tau, lam),
    method="Powell",
    options={
        "maxiter": 2000,
        "disp": False
    }
)

    if not result.success:
        print(f"Warning: optimization failed on {date.date()} with lambda={lam}")

    params_hat = result.x
    intercept_hat = params_hat[0]
    beta_hat = params_hat[1:]

    forecast = intercept_hat + X_forecast_np @ beta_hat

    return forecast[0], result.success


def recursive_forecasts(start_date, end_date, lam, tau):
    forecast_dates = y.loc[start_date:end_date].index

    rows = []

    for i, date in enumerate(forecast_dates):
        forecast, success = fit_and_forecast(date, lam, tau)

        rows.append({
            "date": date,
            f"q{tau:.2f}": forecast,
            "actual": y.loc[date],
            "lambda": lam,
            "success": success
        })

        if i % 25 == 0:
            print(f"lambda={lam}: completed {i}/{len(forecast_dates)} forecasts")

    return pd.DataFrame(rows).set_index("date")


def average_pinball_loss(forecast_df, tau):
    forecast_col = f"q{tau:.2f}"

    losses = pinball_loss(
        forecast_df["actual"].to_numpy(),
        forecast_df[forecast_col].to_numpy(),
        tau
    )

    return losses.mean()


# ------------------------------------------------------------
# Validation: choose lambda
# ------------------------------------------------------------

validation_results = []

for lam in lambda_grid:
    print("\n" + "-" * 60)
    print(f"Running validation for lambda={lam}")
    print("-" * 60)

    val_forecasts = recursive_forecasts(
        validation_start,
        validation_end,
        lam,
        tau
    )

    val_loss = average_pinball_loss(val_forecasts, tau)

    validation_results.append({
        "lambda": lam,
        "validation_loss": val_loss
    })

    val_forecasts.to_csv(f"linear_q{tau:.2f}_validation_lambda_{lam}.csv")

    print(f"lambda={lam}, validation loss={val_loss}")


validation_results_df = pd.DataFrame(validation_results)
validation_results_df.to_csv(
    f"linear_q{tau:.2f}_validation_results.csv",
    index=False
)

best_lambda = validation_results_df.loc[
    validation_results_df["validation_loss"].idxmin(),
    "lambda"
]

print("\nValidation results:")
print(validation_results_df)

print(f"\nBest lambda selected by validation: {best_lambda}")


# ------------------------------------------------------------
# Test: evaluate selected lambda
# ------------------------------------------------------------

print("\n" + "=" * 60)
print(f"Running test period with best lambda={best_lambda}")
print("=" * 60)

test_forecasts = recursive_forecasts(
    test_start,
    None,
    best_lambda,
    tau
)

test_loss = average_pinball_loss(test_forecasts, tau)

test_forecasts.to_csv(f"linear_q{tau:.2f}_test_forecasts.csv")

test_results = pd.DataFrame([{
    "tau": tau,
    "best_lambda": best_lambda,
    "test_loss": test_loss
}])

test_results.to_csv(f"linear_q{tau:.2f}_test_results.csv", index=False)

print("\nTest results:")
print(test_results)

print("\nSaved:")
print(f"linear_q{tau:.2f}_validation_results.csv")
print(f"linear_q{tau:.2f}_test_forecasts.csv")
print(f"linear_q{tau:.2f}_test_results.csv")