from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_utils import load_replication_data, standardize_train_forecast
from losses import pinball_loss_torch
from models import QuantileNetwork


# ------------------------------------------------------------
# Reproducibility and configuration
# ------------------------------------------------------------

SEED = 123

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")

LEARNING_RATE = 0.0001
EPOCHS_INITIAL = 500
EPOCHS_UPDATE = 100

# Temporary debugging output only.
# Delete the entire convergence_debug/ folder when experiments are complete.
LR_LABEL = f"lr_{LEARNING_RATE:.0e}".replace("-", "m")
DEBUG_DIR = (
    PROJECT_ROOT
    / "debugging"
    / "outputs"
    / LR_LABEL
)

DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# Validation period only. No 2000-2024 test data are used.
VALIDATION_START = pd.Timestamp("1980-01-01")
VALIDATION_END = pd.Timestamp("1999-12-01")

DEBUG_DATES = [
    pd.Timestamp("1980-01-01"),
    pd.Timestamp("1985-01-01"),
    pd.Timestamp("1990-01-01"),
    pd.Timestamp("1995-01-01"),
    pd.Timestamp("1999-12-01"),
]


DIAGNOSTIC_CASES = [
    {
        "name": "median_linear",
        "tau": 0.50,
        "nonlinear_layers": 1,
        "hidden_dim": 4,
        "alpha": 1.0,
        "lambda": 1.0,
    },
    {
        "name": "upper_tail_leaky_relu",
        "tau": 0.90,
        "nonlinear_layers": 1,
        "hidden_dim": 4,
        "alpha": 0.5,
        "lambda": 1.0,
    },
    {
        "name": "extreme_upper_tail",
        "tau": 0.95,
        "nonlinear_layers": 2,
        "hidden_dim": 4,
        "alpha": 0.5,
        "lambda": 1.0,
    },
]


# Only these plots are useful for the current learning-rate comparison.
PLOTS_TO_GENERATE = {
    "median_linear": [
        "forecast",
        "pinball_mean",
        "gradient_norm",
    ],
    "upper_tail_leaky_relu": [
        "forecast",
        "pinball_mean",
        "gradient_norm",
    ],
    "extreme_upper_tail": [
        "pinball_mean",
    ],
}


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

X, y = load_replication_data(
    PROJECT_ROOT / "data" / "replication_dataset.csv"
)


def initialize_model(model, y_train, tau):
    """
    Current project initialization.

    The paper does not specify its exact initialization rule.
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "weight" in name:
                parameter.normal_(mean=0.0, std=0.01)
            elif "bias" in name:
                parameter.zero_()

        final_layer = model.network[-1]
        initial_quantile = float(np.quantile(y_train.to_numpy(), tau))
        final_layer.bias.fill_(initial_quantile)


def prepare_forecast_origin(date):
    """
    Construct one expanding-window training set using information
    available before the validation forecast date.
    """
    X_train_raw = X.loc[:date].iloc[:-1]
    y_train = y.loc[:date].iloc[:-1]
    X_forecast_raw = X.loc[[date]]

    if len(y_train) == 0:
        raise ValueError(f"No training data available before {date:%Y-%m}.")

    X_train_std, X_forecast_std = standardize_train_forecast(
        X_train_raw,
        X_forecast_raw,
    )

    return {
        "X_train_tensor": torch.tensor(
            X_train_std.to_numpy(),
            dtype=torch.float32,
            device=device,
        ),
        "y_train_tensor": torch.tensor(
            y_train.to_numpy(),
            dtype=torch.float32,
            device=device,
        ),
        "X_forecast_tensor": torch.tensor(
            X_forecast_std.to_numpy(),
            dtype=torch.float32,
            device=device,
        ),
        "y_train_series": y_train,
        "actual": float(y.loc[date]),
    }


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def global_gradient_norm(model):
    squared_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().pow(2).sum().item()

    return squared_norm ** 0.5


def train_with_diagnostics(
    model,
    X_train_tensor,
    y_train_tensor,
    X_forecast_tensor,
    tau,
    lam,
    epochs,
    learning_rate,
    date,
):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    history = []
    n_observations = len(y_train_tensor)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        predictions = model(X_train_tensor)

        pinball_sum = pinball_loss_torch(
            y_train_tensor,
            predictions,
            tau,
        )

        l2_penalty = model.l2_penalty()
        total_loss = pinball_sum + lam * l2_penalty

        total_loss.backward()
        gradient_norm = global_gradient_norm(model)

        optimizer.step()

        with torch.no_grad():
            forecast = float(model(X_forecast_tensor).item())

        history.append({
            "date": date,
            "epoch": epoch,
            "n_observations": n_observations,
            "learning_rate": learning_rate,
            "tau": tau,
            "lambda": lam,
            "pinball_mean": float(
                pinball_sum.detach().item() / n_observations
            ),
            "gradient_norm": gradient_norm,
            "forecast": forecast,
        })

    return model, pd.DataFrame(history)


def ordinary_sgd_update(
    model,
    X_train_tensor,
    y_train_tensor,
    tau,
    lam,
    epochs,
):
    """
    Warm-start update for dates whose epoch histories are not retained.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for _ in range(epochs):
        optimizer.zero_grad()

        predictions = model(X_train_tensor)
        pinball = pinball_loss_torch(
            y_train_tensor,
            predictions,
            tau,
        )

        total_loss = pinball + lam * model.l2_penalty()
        total_loss.backward()
        optimizer.step()

    return model


def plot_metric(history, metric, case_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    for date, date_history in history.groupby("date"):
        ax.plot(
            date_history["epoch"],
            date_history[metric],
            label=pd.Timestamp(date).strftime("%Y-%m"),
        )

    readable_metric = metric.replace("_", " ").title()

    ax.set_title(
        f"{case_name}: {readable_metric} "
        f"(lr={LEARNING_RATE:g})"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(readable_metric)
    ax.grid(alpha=0.25)
    ax.legend(title="Forecast origin")

    fig.tight_layout()

    output_file = (
        DEBUG_DIR
        / f"{case_name}_{metric}_lr_{LEARNING_RATE:.0e}.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {output_file}")


def run_diagnostic_case(case):
    case_name = case["name"]
    tau = case["tau"]
    lam = case["lambda"]
    nonlinear_layers = case["nonlinear_layers"]
    hidden_dim = case["hidden_dim"]
    alpha = case["alpha"]

    validation_dates = y.loc[
        VALIDATION_START:VALIDATION_END
    ].index

    debug_dates_available = [
        date for date in DEBUG_DATES if date in validation_dates
    ]

    if not debug_dates_available:
        raise ValueError("None of the selected debug dates are available.")

    model = None
    all_histories = []

    for forecast_number, date in enumerate(validation_dates):
        item = prepare_forecast_origin(date)

        if model is None:
            model = QuantileNetwork(
                n_features=item["X_train_tensor"].shape[1],
                nonlinear_layers=nonlinear_layers,
                hidden_dim=hidden_dim,
                alpha=alpha,
            ).to(device)

            initialize_model(
                model,
                item["y_train_series"],
                tau,
            )

            epochs = EPOCHS_INITIAL
        else:
            epochs = EPOCHS_UPDATE

        if date in debug_dates_available:
            print(
                f"{case_name}: diagnosing {date:%Y-%m} "
                f"for {epochs} epochs"
            )

            model, history = train_with_diagnostics(
                model=model,
                X_train_tensor=item["X_train_tensor"],
                y_train_tensor=item["y_train_tensor"],
                X_forecast_tensor=item["X_forecast_tensor"],
                tau=tau,
                lam=lam,
                epochs=epochs,
                learning_rate=LEARNING_RATE,
                date=date,
            )

            history["actual"] = item["actual"]
            all_histories.append(history)

        else:
            model = ordinary_sgd_update(
                model=model,
                X_train_tensor=item["X_train_tensor"],
                y_train_tensor=item["y_train_tensor"],
                tau=tau,
                lam=lam,
                epochs=epochs,
            )

        if forecast_number % 50 == 0:
            print(
                f"{case_name}: validation forecast "
                f"{forecast_number}/{len(validation_dates)}"
            )

    combined_history = pd.concat(
        all_histories,
        ignore_index=True,
    )

    # One compact CSV per case in the temporary debug directory.
    history_file = (
        DEBUG_DIR
        / f"{case_name}_history_lr_{LEARNING_RATE:.0e}.csv"
    )
    combined_history.to_csv(history_file, index=False)
    print(f"Saved {history_file}")

    for metric in PLOTS_TO_GENERATE[case_name]:
        plot_metric(
            combined_history,
            metric,
            case_name,
        )


def main():
    print(f"Temporary debug output directory: {DEBUG_DIR}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("Validation period only: 1980-01 through 1999-12")

    for case in DIAGNOSTIC_CASES:
        print("\n" + "=" * 80)
        print(f"Running convergence diagnostic: {case['name']}")
        print("=" * 80)

        # Reset seeds so each case remains reproducible.
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        run_diagnostic_case(case)


if __name__ == "__main__":
    main()