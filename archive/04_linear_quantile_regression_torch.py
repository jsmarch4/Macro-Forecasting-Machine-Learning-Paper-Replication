import pandas as pd
import numpy as np
import torch
from torch import nn

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

torch.manual_seed(123)
np.random.seed(123)

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

lambda_grid = np.exp(np.linspace(np.log(0.2), np.log(10), 40))

epochs_initial = 500
epochs_update = 100
learning_rate = 0.01

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def pinball_loss_torch(y_true, y_pred, tau):
    error = y_true - y_pred
    return torch.mean(torch.maximum(tau * error, (tau - 1) * error))


def pinball_loss_numpy(y_true, y_pred, tau):
    error = y_true - y_pred
    return np.where(error >= 0, tau * error, (tau - 1) * error)


class LinearQuantileModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze()


def standardize_train_forecast(X_train_raw, X_forecast_raw):
    train_mean = X_train_raw.mean()
    train_std = X_train_raw.std()
    train_std = train_std.replace(0, 1)

    X_train = (X_train_raw - train_mean) / train_std
    X_forecast = (X_forecast_raw - train_mean) / train_std

    return X_train, X_forecast


def train_model(model, X_train_tensor, y_train_tensor, tau, lam, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = model(X_train_tensor)

        loss = pinball_loss_torch(y_train_tensor, y_pred, tau)

        # L2 penalty on slope coefficients only, not intercept
        beta = model.linear.weight.squeeze()
        ridge_penalty = lam * torch.sum(beta ** 2)

        total_loss = loss + ridge_penalty

        total_loss.backward()
        optimizer.step()

    return model


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

        X_train_tensor = torch.tensor(
            X_train_std.to_numpy(),
            dtype=torch.float32
        )

        y_train_tensor = torch.tensor(
            y_train.to_numpy(),
            dtype=torch.float32
        )

        X_forecast_tensor = torch.tensor(
            X_forecast_std.to_numpy(),
            dtype=torch.float32
        )

        n_features = X_train_tensor.shape[1]

        # First date: initialize model
        if model is None:
            model = LinearQuantileModel(n_features)

            # Start intercept near unconditional quantile
            with torch.no_grad():
                model.linear.bias.fill_(float(np.quantile(y_train.to_numpy(), tau)))
                model.linear.weight.zero_()

            epochs = epochs_initial

        # Later dates: warm start previous model
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


def average_pinball_loss(forecast_df, tau):
    forecast_col = f"q{tau:.2f}"

    losses = pinball_loss_numpy(
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

    val_forecasts.to_csv(f"torch_linear_q{tau:.2f}_validation_lambda_{lam}.csv")

    print(f"lambda={lam}, validation loss={val_loss}")


validation_results_df = pd.DataFrame(validation_results)

validation_results_df.to_csv(
    f"torch_linear_q{tau:.2f}_validation_results.csv",
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

test_forecasts.to_csv(f"torch_linear_q{tau:.2f}_test_forecasts.csv")

test_results = pd.DataFrame([{
    "tau": tau,
    "best_lambda": best_lambda,
    "test_loss": test_loss
}])

test_results.to_csv(
    f"torch_linear_q{tau:.2f}_test_results.csv",
    index=False
)

print("\nTest results:")
print(test_results)

print("\nSaved:")
print(f"torch_linear_q{tau:.2f}_validation_results.csv")
print(f"torch_linear_q{tau:.2f}_test_forecasts.csv")
print(f"torch_linear_q{tau:.2f}_test_results.csv")