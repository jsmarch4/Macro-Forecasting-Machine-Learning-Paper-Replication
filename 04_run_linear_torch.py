import numpy as np
import pandas as pd
import torch

from models import LinearQuantileModel
from data_utils import load_replication_data, standardize_train_forecast
from train_utils import train_model, average_pinball_loss

torch.manual_seed(123)
np.random.seed(123)

X, y = load_replication_data()

tau = 0.90

validation_start = "1980-01-01"
validation_end = "1999-12-01"
test_start = "2000-01-01"

lambda_grid = [0.001, 0.01, 0.1, 1.0, 10.0]

epochs_initial = 500
epochs_update = 100
learning_rate = 0.01


def recursive_forecasts(start_date, end_date, lam, tau):
    if end_date is None:
        forecast_dates = y.loc[start_date:].index
    else:
        forecast_dates = y.loc[start_date:end_date].index

    rows = []
    model = None

    for i, date in enumerate(forecast_dates):
        X_train_raw = X.loc[:date].iloc[:-1]
        y_train = y.loc[:date].iloc[:-1]
        X_forecast_raw = X.loc[[date]]

        X_train_std, X_forecast_std = standardize_train_forecast(
            X_train_raw,
            X_forecast_raw
        )

        X_train_tensor = torch.tensor(X_train_std.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        X_forecast_tensor = torch.tensor(X_forecast_std.to_numpy(), dtype=torch.float32)

        n_features = X_train_tensor.shape[1]

        if model is None:
            model = LinearQuantileModel(n_features)

            with torch.no_grad():
                model.linear.bias.fill_(float(np.quantile(y_train.to_numpy(), tau)))
                model.linear.weight.zero_()

            epochs = epochs_initial
        else:
            epochs = epochs_update

        model = train_model(
            model,
            X_train_tensor,
            y_train_tensor,
            tau,
            lam,
            epochs,
            learning_rate
        )

        with torch.no_grad():
            forecast = model(X_forecast_tensor).item()

        rows.append({
            "date": date,
            f"q{tau:.2f}": forecast,
            "actual": y.loc[date],
            "lambda": lam
        })

        if i % 25 == 0:
            print(f"lambda={lam}: completed {i}/{len(forecast_dates)} forecasts")

    return pd.DataFrame(rows).set_index("date")


validation_results = []

for lam in lambda_grid:
    print(f"\nRunning validation for lambda={lam}")

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

    print(f"lambda={lam}, validation loss={val_loss}")


validation_results_df = pd.DataFrame(validation_results)

best_lambda = validation_results_df.loc[
    validation_results_df["validation_loss"].idxmin(),
    "lambda"
]

print("\nValidation results:")
print(validation_results_df)
print(f"\nBest lambda: {best_lambda}")

test_forecasts = recursive_forecasts(
    test_start,
    None,
    best_lambda,
    tau
)

test_loss = average_pinball_loss(test_forecasts, tau)

test_forecasts.to_csv(f"Results/torch_linear_q{tau:.2f}_test_forecasts.csv")

test_results = pd.DataFrame([{
    "tau": tau,
    "best_lambda": best_lambda,
    "test_loss": test_loss
}])

test_results.to_csv(
    f"Results/torch_linear_q{tau:.2f}_test_results.csv",
    index=False
)

print("\nTest results:")
print(test_results)