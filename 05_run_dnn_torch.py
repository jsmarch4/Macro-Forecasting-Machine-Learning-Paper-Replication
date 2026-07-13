import numpy as np
import pandas as pd
import torch

from models import QuantileNetwork
from data_utils import load_replication_data, standardize_train_forecast
from train_utils import train_model, average_pinball_loss

torch.manual_seed(123)
np.random.seed(123)

# Tried to use Apple mps GPU and was slower
device = torch.device("cpu")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X, y = load_replication_data()

# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------

quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

validation_start = "1980-01-01"
validation_end = "1999-12-01"

test_start = "2000-01-01"
test_end = "2024-01-01"

# Small grid for now. Paper-style full grid later:
# lambda_grid = np.exp(np.linspace(np.log(0.2), np.log(10), 40))
lambda_grid = [0.1, 1.0, 10.0]

architecture_grid = [
    {"nonlinear_layers": 1, "hidden_dim": 2, "alpha": 0.0},
    {"nonlinear_layers": 1, "hidden_dim": 2, "alpha": 0.5},
    {"nonlinear_layers": 1, "hidden_dim": 2, "alpha": 1.0},

    {"nonlinear_layers": 1, "hidden_dim": 4, "alpha": 0.0},
    {"nonlinear_layers": 1, "hidden_dim": 4, "alpha": 0.5},
    {"nonlinear_layers": 1, "hidden_dim": 4, "alpha": 1.0},

    {"nonlinear_layers": 1, "hidden_dim": 8, "alpha": 0.0},
    {"nonlinear_layers": 1, "hidden_dim": 8, "alpha": 0.5},
    {"nonlinear_layers": 1, "hidden_dim": 8, "alpha": 1.0},

    {"nonlinear_layers": 2, "hidden_dim": 2, "alpha": 0.0},
    {"nonlinear_layers": 2, "hidden_dim": 2, "alpha": 0.5},
    {"nonlinear_layers": 2, "hidden_dim": 2, "alpha": 1.0},

    {"nonlinear_layers": 2, "hidden_dim": 4, "alpha": 0.0},
    {"nonlinear_layers": 2, "hidden_dim": 4, "alpha": 0.5},
    {"nonlinear_layers": 2, "hidden_dim": 4, "alpha": 1.0},

    {"nonlinear_layers": 2, "hidden_dim": 8, "alpha": 0.0},
    {"nonlinear_layers": 2, "hidden_dim": 8, "alpha": 0.5},
    {"nonlinear_layers": 2, "hidden_dim": 8, "alpha": 1.0},
]


epochs_initial = 500
epochs_update = 100
learning_rate = 0.001


# ------------------------------------------------------------
# Recursive forecasting
# ------------------------------------------------------------

def initialize_model(model, y_train, tau):
    """
    Initialize network weights.

    The paper does not specify exact initialization.
    This uses small random weights and sets the final output bias
    to the historical tau-quantile of the target.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.normal_(mean=0.0, std=0.01)
            elif "bias" in name:
                param.zero_()

        final_layer = model.network[-1]
        final_layer.bias.fill_(float(np.quantile(y_train.to_numpy(), tau)))

def build_forecast_cache(start_date, end_date):
    """
    Precompute the standardized expanding-window datasets for each forecast date.

    This avoids repeatedly standardizing and converting to tensors for every
    architecture/lambda combination.
    """
    if end_date is None:
        forecast_dates = y.loc[start_date:].index
    else:
        forecast_dates = y.loc[start_date:end_date].index

    cache = []

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
            dtype=torch.float32,
            device=device
        )

        y_train_tensor = torch.tensor(
            y_train.to_numpy(),
            dtype=torch.float32,
            device=device
        )

        X_forecast_tensor = torch.tensor(
            X_forecast_std.to_numpy(),
            dtype=torch.float32,
            device=device
        )

        cache.append({
            "date": date,
            "X_train_tensor": X_train_tensor,
            "y_train_tensor": y_train_tensor,
            "X_forecast_tensor": X_forecast_tensor,
            "y_train_series": y_train,
            "actual": y.loc[date],
            "n_features": X_train_tensor.shape[1]
        })

        if i % 100 == 0:
            print(f"Cache: {i}/{len(forecast_dates)}")

    return cache


def recursive_forecasts(
    forecast_cache,
    lam,
    tau,
    nonlinear_layers,
    hidden_dim,
    alpha
):
    rows = []
    model = None

    for i, item in enumerate(forecast_cache):
        date = item["date"]
        X_train_tensor = item["X_train_tensor"]
        y_train_tensor = item["y_train_tensor"]
        X_forecast_tensor = item["X_forecast_tensor"]
        y_train_series = item["y_train_series"]
        actual = item["actual"]
        n_features = item["n_features"]

        if model is None:
            model = QuantileNetwork(
                n_features=n_features,
                nonlinear_layers=nonlinear_layers,
                hidden_dim=hidden_dim,
                alpha=alpha
            )
            model = model.to(device)

            initialize_model(model, y_train_series, tau)

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
            "actual": actual,
            "lambda": lam,
            "nonlinear_layers": nonlinear_layers,
            "hidden_dim": hidden_dim,
            "alpha": alpha
        })

        if i % 100 == 0:
            print(f"  Forecast {i}/{len(forecast_cache)}")       


    return pd.DataFrame(rows).set_index("date")


# ------------------------------------------------------------
# Validation and test loop
# ------------------------------------------------------------

print("\nBuilding validation cache...")
validation_cache = build_forecast_cache(
    validation_start,
    validation_end
)

print("\nBuilding test cache...")
test_cache = build_forecast_cache(
    test_start,
    test_end
)


all_test_results = []
all_validation_results = []

best_validation_loss = float("inf")
best_validation_forecasts = None

for tau in quantiles:
    print("\n" + "=" * 80)
    print(f"Running Deep Neural Network search for tau={tau}")
    print("=" * 80)

    validation_results = []
    best_validation_loss = float("inf")
    best_validation_forecasts = None

    total_models = len(architecture_grid) * len(lambda_grid)
    model_number = 0

    for arch in architecture_grid:
        nonlinear_layers = arch["nonlinear_layers"]
        hidden_dim = arch["hidden_dim"]
        alpha = arch["alpha"]

        for lam in lambda_grid:
            model_number += 1

            print(
                f"\nModel {model_number}/{total_models}: "
                f"tau={tau} | "
                f"layers={nonlinear_layers} | "
                f"dim={hidden_dim} | "
                f"alpha={alpha} | "
                f"lambda={lam}"
            )

            val_forecasts = recursive_forecasts(
                forecast_cache=validation_cache,
                lam=lam,
                tau=tau,
                nonlinear_layers=nonlinear_layers,
                hidden_dim=hidden_dim,
                alpha=alpha
            )
            

            val_loss = average_pinball_loss(val_forecasts, tau)

            validation_results.append({
                "model_family": "dnn",
                "tau": tau,
                "nonlinear_layers": nonlinear_layers,
                "hidden_dim": hidden_dim,
                "alpha": alpha,
                "lambda": lam,
                "validation_loss": val_loss
            })

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_validation_forecasts = val_forecasts.copy()



    validation_results_df = pd.DataFrame(validation_results)

    validation_results_df.to_csv(
        f"results/dnn_q{tau:.2f}_validation_results.csv",
        index=False
    )

    best_row = validation_results_df.loc[
        validation_results_df["validation_loss"].idxmin()
    ]

    best_nonlinear_layers = int(best_row["nonlinear_layers"])
    best_hidden_dim = int(best_row["hidden_dim"])
    best_alpha = float(best_row["alpha"])
    best_lambda = float(best_row["lambda"])


    print(
        f"\nBest validation model:"
        f"\n  layers = {best_nonlinear_layers}"
        f"\n  hidden dimension = {best_hidden_dim}"
        f"\n  alpha = {best_alpha}"
        f"\n  lambda = {best_lambda}"
        f"\n  validation loss = {best_row['validation_loss']:.6f}"
    )


    best_validation_forecasts.to_csv(
    f"results/dnn_q{tau:.2f}_validation_forecasts.csv"
        )
    
    
    print("\nRunning out-of-sample test...")

    test_forecasts = recursive_forecasts(
        forecast_cache=test_cache,
        lam=best_lambda,
        tau=tau,
        nonlinear_layers=best_nonlinear_layers,
        hidden_dim=best_hidden_dim,
        alpha=best_alpha
    )

    test_loss = average_pinball_loss(test_forecasts, tau)
    print(f"Test pinball loss = {test_loss:.6f}")

    test_forecasts.to_csv(
        f"results/dnn_q{tau:.2f}_test_forecasts.csv"
    )

    all_test_results.append({
        "tau": tau,
        "best_nonlinear_layers": best_nonlinear_layers,
        "best_hidden_dim": best_hidden_dim,
        "best_alpha": best_alpha,
        "best_lambda": best_lambda,
        "test_loss": test_loss
    })



all_test_results_df = pd.DataFrame(all_test_results)

all_test_results_df.to_csv(
    "results/dnn_all_quantiles_test_results.csv",
    index=False
)

all_validation_results_df = pd.DataFrame(all_validation_results)

all_validation_results_df.to_csv(
    "results/dnn_hyperparameter_search_results.csv",
    index=False
)

print("\nFinal Results")
print("-" * 60)
print(
    all_test_results_df[
        ["tau", "best_nonlinear_layers",
         "best_hidden_dim", "best_lambda",
         "test_loss"]
    ]
)