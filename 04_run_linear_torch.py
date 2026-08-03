from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from models import QuantileNetwork
from data_utils import (
    load_replication_data,
    standardize_train_forecast,
    standardize_target_train_forecast,
)
from train_utils import train_model, average_pinball_loss


# ---------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Organized output directories
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
FAMILY_DIR = RESULTS_DIR / "linear_activation"

SEARCH_DIR = FAMILY_DIR / "search"
VALIDATION_FORECAST_DIR = FAMILY_DIR / "validation" / "forecasts"
VALIDATION_RESULTS_DIR = FAMILY_DIR / "validation" / "results"
TEST_FORECAST_DIR = FAMILY_DIR / "test" / "forecasts"
TEST_SUMMARY_DIR = FAMILY_DIR / "test" / "summaries"

for directory in [
    SEARCH_DIR,
    VALIDATION_FORECAST_DIR,
    VALIDATION_RESULTS_DIR,
    TEST_FORECAST_DIR,
    TEST_SUMMARY_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Data and experiment settings
# ---------------------------------------------------------------------

X, y = load_replication_data()

ALL_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--quantile",
    type=float,
    choices=ALL_QUANTILES,
    required=True,
    help="Quantile to estimate.",
)
args = parser.parse_args()

tau = float(args.quantile)

VALIDATION_START = "1980-01-01"
VALIDATION_END = "1999-12-01"
TEST_START = "2000-01-01"
TEST_END = "2024-01-01"

# Paper-style 40-point log-spaced lambda grid.
LAMBDA_GRID = np.exp(
    np.linspace(np.log(0.2), np.log(10.0), 40)
)

ARCHITECTURE_GRID = [
    {"nonlinear_layers": 0, "hidden_dim": 0, "alpha": 1.0},
    {"nonlinear_layers": 1, "hidden_dim": 2, "alpha": 1.0},
    {"nonlinear_layers": 1, "hidden_dim": 4, "alpha": 1.0},
    {"nonlinear_layers": 1, "hidden_dim": 8, "alpha": 1.0},
    {"nonlinear_layers": 2, "hidden_dim": 2, "alpha": 1.0},
    {"nonlinear_layers": 2, "hidden_dim": 4, "alpha": 1.0},
    {"nonlinear_layers": 2, "hidden_dim": 8, "alpha": 1.0},
]

EPOCHS_INITIAL = 500
EPOCHS_UPDATE = 100
LEARNING_RATE = 0.001


# ---------------------------------------------------------------------
# Model and forecast helpers
# ---------------------------------------------------------------------

def initialize_model(
    model: QuantileNetwork,
    y_train: pd.Series,
    tau_value: float,
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "weight" in name:
                parameter.normal_(mean=0.0, std=0.01)
            elif "bias" in name:
                parameter.zero_()

        final_layer = model.network[-1]
        final_layer.bias.fill_(
            float(np.quantile(y_train.to_numpy(), tau_value))
        )


def build_forecast_cache(
    start_date: str,
    end_date: str | None,
) -> list[dict]:
    if end_date is None:
        forecast_dates = y.loc[start_date:].index
    else:
        forecast_dates = y.loc[start_date:end_date].index

    cache = []

    for i, date in enumerate(forecast_dates):
        X_train_raw = X.loc[:date].iloc[:-1]
        y_train_raw = y.loc[:date].iloc[:-1]
        actual_raw = float(y.loc[date])
        X_forecast_raw = X.loc[[date]]

        y_train_std, actual_std, y_mean, y_std = (
            standardize_target_train_forecast(
                y_train_raw,
                actual_raw,
            )
        )

        X_train_std, X_forecast_std = standardize_train_forecast(
            X_train_raw,
            X_forecast_raw,
        )

        cache.append({
            "date": date,
            "X_train_tensor": torch.tensor(
                X_train_std.to_numpy(),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "y_train_tensor": torch.tensor(
                y_train_std.to_numpy(),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "X_forecast_tensor": torch.tensor(
                X_forecast_std.to_numpy(),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "y_train_series": y_train_std,
            "actual": float(actual_std),
            "actual_raw": actual_raw,
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "n_features": X_train_std.shape[1],
        })

        if i % 100 == 0:
            print(f"Cache: {i}/{len(forecast_dates)}")

    return cache


def recursive_forecasts(
    forecast_cache: list[dict],
    lam: float,
    tau_value: float,
    nonlinear_layers: int,
    hidden_dim: int,
    alpha: float,
    lr: float,
) -> pd.DataFrame:
    rows = []
    model = None

    for i, item in enumerate(forecast_cache):
        if model is None:
            model = QuantileNetwork(
                n_features=item["n_features"],
                nonlinear_layers=nonlinear_layers,
                hidden_dim=hidden_dim,
                alpha=alpha,
            ).to(DEVICE)

            initialize_model(
                model,
                item["y_train_series"],
                tau_value,
            )
            epochs = EPOCHS_INITIAL
        else:
            epochs = EPOCHS_UPDATE

        model = train_model(
            model=model,
            X_train_tensor=item["X_train_tensor"],
            y_train_tensor=item["y_train_tensor"],
            tau=tau_value,
            lam=float(lam),
            epochs=epochs,
            lr=lr,
        )

        with torch.no_grad():
            forecast_std = float(
                model(item["X_forecast_tensor"]).item()
            )

        forecast_raw = (
            forecast_std * item["y_std"]
            + item["y_mean"]
        )

        rows.append({
            "date": item["date"],
            f"q{tau_value:.2f}": forecast_std,
            f"q{tau_value:.2f}_raw": forecast_raw,
            "actual": item["actual"],
            "actual_raw": item["actual_raw"],
            "lambda": float(lam),
            "nonlinear_layers": nonlinear_layers,
            "hidden_dim": hidden_dim,
            "alpha": alpha,
        })

        if i % 100 == 0:
            print(f"  Forecast {i}/{len(forecast_cache)}")

    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------
# Main search for one array quantile
# ---------------------------------------------------------------------

print("\nBuilding validation cache...")
validation_cache = build_forecast_cache(
    VALIDATION_START,
    VALIDATION_END,
)

print("\nBuilding test cache...")
test_cache = build_forecast_cache(
    TEST_START,
    TEST_END,
)

validation_results = []
best_validation_loss = float("inf")
best_validation_forecasts = None

total_models = len(ARCHITECTURE_GRID) * len(LAMBDA_GRID)
model_number = 0

print("\n" + "=" * 80)
print(f"Running linear-activation search for tau={tau:.2f}")
print(f"Lambda candidates: {len(LAMBDA_GRID)}")
print("=" * 80)

for architecture in ARCHITECTURE_GRID:
    nonlinear_layers = architecture["nonlinear_layers"]
    hidden_dim = architecture["hidden_dim"]
    alpha = architecture["alpha"]

    for lam in LAMBDA_GRID:
        model_number += 1

        print(
            f"\nModel {model_number}/{total_models}: "
            f"tau={tau:.2f} | "
            f"layers={nonlinear_layers} | "
            f"dim={hidden_dim} | "
            f"alpha={alpha} | "
            f"lambda={lam:.12g}"
        )

        validation_forecasts = recursive_forecasts(
            forecast_cache=validation_cache,
            lam=float(lam),
            tau_value=tau,
            nonlinear_layers=nonlinear_layers,
            hidden_dim=hidden_dim,
            alpha=alpha,
            lr=LEARNING_RATE,
        )

        validation_loss = average_pinball_loss(
            validation_forecasts,
            tau,
        )

        if not np.isfinite(validation_loss):
            retry_learning_rate = 0.0001
            print(
                "Non-finite validation loss; retrying candidate with "
                f"learning rate {retry_learning_rate}."
            )

            validation_forecasts = recursive_forecasts(
                forecast_cache=validation_cache,
                lam=float(lam),
                tau_value=tau,
                nonlinear_layers=nonlinear_layers,
                hidden_dim=hidden_dim,
                alpha=alpha,
                lr=retry_learning_rate,
            )

            validation_loss = average_pinball_loss(
                validation_forecasts,
                tau,
            )

        if not np.isfinite(validation_loss):
            raise RuntimeError(
                "Non-finite validation loss after retry for "
                f"tau={tau:.2f}, "
                f"layers={nonlinear_layers}, "
                f"hidden_dim={hidden_dim}, "
                f"alpha={alpha}, "
                f"lambda={lam}"
            )

        result = {
            "model_family": "linear_activation",
            "tau": tau,
            "nonlinear_layers": nonlinear_layers,
            "hidden_dim": hidden_dim,
            "alpha": alpha,
            "lambda": float(lam),
            "validation_loss": float(validation_loss),
        }
        validation_results.append(result)

        if validation_loss < best_validation_loss:
            best_validation_loss = float(validation_loss)
            best_validation_forecasts = validation_forecasts.copy()

validation_results_df = pd.DataFrame(validation_results)

# Full architecture × 40-lambda grid for this quantile.
validation_results_df.to_csv(
    VALIDATION_RESULTS_DIR / f"q{tau:.2f}.csv",
    index=False,
)
validation_results_df.to_csv(
    SEARCH_DIR / f"q{tau:.2f}_search.csv",
    index=False,
)

best_row = validation_results_df.loc[
    validation_results_df["validation_loss"].idxmin()
]

best_nonlinear_layers = int(best_row["nonlinear_layers"])
best_hidden_dim = int(best_row["hidden_dim"])
best_alpha = float(best_row["alpha"])
best_lambda = float(best_row["lambda"])

if best_validation_forecasts is None:
    raise RuntimeError(
        f"No validation forecasts were stored for tau={tau:.2f}."
    )

best_validation_forecasts.to_csv(
    VALIDATION_FORECAST_DIR / f"q{tau:.2f}.csv"
)

print(
    "\nBest validation model:"
    f"\n  layers = {best_nonlinear_layers}"
    f"\n  hidden dimension = {best_hidden_dim}"
    f"\n  alpha = {best_alpha}"
    f"\n  lambda = {best_lambda}"
    f"\n  validation loss = {best_validation_loss:.6f}"
)

print("\nRunning out-of-sample test...")

test_forecasts = recursive_forecasts(
    forecast_cache=test_cache,
    lam=best_lambda,
    tau_value=tau,
    nonlinear_layers=best_nonlinear_layers,
    hidden_dim=best_hidden_dim,
    alpha=best_alpha,
    lr=LEARNING_RATE,
)

test_loss = float(average_pinball_loss(test_forecasts, tau))

if not np.isfinite(test_loss):
    raise RuntimeError(
        f"Non-finite test loss for tau={tau:.2f}."
    )

test_forecasts.to_csv(
    TEST_FORECAST_DIR / f"q{tau:.2f}.csv"
)

test_summary = pd.DataFrame([{
    "tau": tau,
    "best_nonlinear_layers": best_nonlinear_layers,
    "best_hidden_dim": best_hidden_dim,
    "best_alpha": best_alpha,
    "best_lambda": best_lambda,
    "validation_loss": best_validation_loss,
    "test_loss": test_loss,
}])

test_summary.to_csv(
    TEST_SUMMARY_DIR / f"q{tau:.2f}.csv",
    index=False,
)

print("\nFinal result")
print("-" * 60)
print(test_summary.to_string(index=False))