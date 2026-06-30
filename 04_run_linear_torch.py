import numpy as np
import pandas as pd
import torch

from models import QuantileNetwork
from data_utils import load_replication_data, standardize_train_forecast
from train_utils import train_model, average_pinball_loss

torch.manual_seed(123)
np.random.seed(123)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X, y = load_replication_data()

# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------

quantiles = [0.50]

validation_start = "1980-01-01"
validation_end = "1999-12-01"

test_start = "2000-01-01"
test_end = "2024-01-01"

# np.exp(np.linspace(np.log(0.2), np.log(10), 40))
lambda_grid = [0.1, 1.0, 10.0]

# Linear activation case: alpha = 1.0
# hidden_layers = 0 means no hidden layer, so hidden_dim is irrelevant.
architecture_grid = [
    {"hidden_layers": 0, "hidden_dim": 0, "alpha": 1.0},

    {"hidden_layers": 1, "hidden_dim": 2, "alpha": 1.0},
    {"hidden_layers": 1, "hidden_dim": 4, "alpha": 1.0},
    {"hidden_layers": 1, "hidden_dim": 8, "alpha": 1.0},

    {"hidden_layers": 2, "hidden_dim": 2, "alpha": 1.0},
    {"hidden_layers": 2, "hidden_dim": 4, "alpha": 1.0},
    {"hidden_layers": 2, "hidden_dim": 8, "alpha": 1.0},
]

epochs_initial = 500
epochs_update = 100
learning_rate = 0.001


# ------------------------------------------------------------
# Recursive forecasting
# ------------------------------------------------------------

def recursive_forecasts(
    start_date,
    end_date,
    lam,
    tau,
    hidden_layers,
    hidden_dim,
    alpha
):
    if end_date is None:
        forecast_dates = y.loc[start_date:].index
    else:
        forecast_dates = y.loc[start_date:end_date].index

    rows = []
    model = None

    for i, date in enumerate(forecast_dates):
        # Train on all observations strictly before forecast date
        X_train_raw = X.loc[:date].iloc[:-1]
        y_train = y.loc[:date].iloc[:-1]

        # Forecast at date
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

        if model is None:
            model = QuantileNetwork(
                n_features=n_features,
                hidden_layers=hidden_layers,
                hidden_dim=hidden_dim,
                alpha=alpha
            )

            # Initialize output level near historical quantile only for no-hidden-layer model.
            # For networks, use small random initialization.
            with torch.no_grad():
                for param in model.parameters():
                    param.normal_(mean=0.0, std=0.01)

            epochs = epochs_initial

        else:
            epochs = epochs_update

        model = train_model(
            model=model,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            tau=tau,
            lam=lam,
            epochs=epochs,
            lr=learning_rate
        )

        with torch.no_grad():
            forecast = model(X_forecast_tensor).item()

        rows.append({
            "date": date,
            f"q{tau:.2f}": forecast,
            "actual": y.loc[date],
            "lambda": lam,
            "hidden_layers": hidden_layers,
            "hidden_dim": hidden_dim,
            "alpha": alpha
        })

        if i % 25 == 0:
            print(
                f"tau={tau}, layers={hidden_layers}, dim={hidden_dim}, "
                f"alpha={alpha}, lambda={lam}: "
                f"completed {i}/{len(forecast_dates)} forecasts"
            )

    return pd.DataFrame(rows).set_index("date")


# ------------------------------------------------------------
# Validation and test loop
# ------------------------------------------------------------

all_test_results = []

for tau in quantiles:
    print("\n" + "=" * 80)
    print(f"Running paper-style LINEAR ACTIVATION network search for tau={tau}")
    print("=" * 80)

    validation_results = []

    for arch in architecture_grid:
        hidden_layers = arch["hidden_layers"]
        hidden_dim = arch["hidden_dim"]
        alpha = arch["alpha"]

        for lam in lambda_grid:
            print("\n" + "-" * 80)
            print(
                f"Validation: tau={tau}, layers={hidden_layers}, "
                f"dim={hidden_dim}, alpha={alpha}, lambda={lam}"
            )
            print("-" * 80)

            val_forecasts = recursive_forecasts(
                start_date=validation_start,
                end_date=validation_end,
                lam=lam,
                tau=tau,
                hidden_layers=hidden_layers,
                hidden_dim=hidden_dim,
                alpha=alpha
            )

            val_loss = average_pinball_loss(val_forecasts, tau)

            validation_results.append({
                "tau": tau,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "alpha": alpha,
                "lambda": lam,
                "validation_loss": val_loss
            })

            print(
                f"Validation loss: tau={tau}, layers={hidden_layers}, "
                f"dim={hidden_dim}, alpha={alpha}, lambda={lam}, "
                f"loss={val_loss}"
            )

    validation_results_df = pd.DataFrame(validation_results)

    validation_results_df.to_csv(
        f"results/linear_activation_q{tau:.2f}_validation_results.csv",
        index=False
    )

    best_row = validation_results_df.loc[
        validation_results_df["validation_loss"].idxmin()
    ]

    best_hidden_layers = int(best_row["hidden_layers"])
    best_hidden_dim = int(best_row["hidden_dim"])
    best_alpha = float(best_row["alpha"])
    best_lambda = float(best_row["lambda"])

    print("\nBest validation configuration:")
    print(best_row)

    print("\n" + "=" * 80)
    print(
        f"Testing best model: tau={tau}, layers={best_hidden_layers}, "
        f"dim={best_hidden_dim}, alpha={best_alpha}, lambda={best_lambda}"
    )
    print("=" * 80)

    test_forecasts = recursive_forecasts(
        start_date=test_start,
        end_date=test_end,
        lam=best_lambda,
        tau=tau,
        hidden_layers=best_hidden_layers,
        hidden_dim=best_hidden_dim,
        alpha=best_alpha
    )

    test_loss = average_pinball_loss(test_forecasts, tau)

    test_forecasts.to_csv(
        f"results/linear_activation_q{tau:.2f}_test_forecasts.csv"
    )

    all_test_results.append({
        "tau": tau,
        "best_hidden_layers": best_hidden_layers,
        "best_hidden_dim": best_hidden_dim,
        "best_alpha": best_alpha,
        "best_lambda": best_lambda,
        "test_loss": test_loss
    })

    print("\nTest result:")
    print(all_test_results[-1])


all_test_results_df = pd.DataFrame(all_test_results)

all_test_results_df.to_csv(
    "results/linear_activation_all_quantiles_test_results.csv",
    index=False
)

print("\nAll test results:")
print(all_test_results_df)