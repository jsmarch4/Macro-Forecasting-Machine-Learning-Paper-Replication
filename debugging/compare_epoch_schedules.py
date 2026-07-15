from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import load_replication_data, standardize_train_forecast
from losses import pinball_loss_torch, pinball_loss_numpy
from models import QuantileNetwork


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SEED = 123
device = torch.device("cpu")

VALIDATION_START = "1980-01-01"
VALIDATION_END = "1999-12-01"

LEARNING_RATE = 0.001

EPOCH_SCHEDULES = [
    {"epochs_initial": 500, "epochs_update": 100},
    {"epochs_initial": 750, "epochs_update": 150},
    {"epochs_initial": 1000, "epochs_update": 200},
]

MODEL_CASES = [
    {
        "model": "median_linear",
        "tau": 0.50,
        "nonlinear_layers": 1,
        "hidden_dim": 4,
        "alpha": 1.0,
        "lambda": 1.0,
    },
    {
        "model": "upper_tail_leaky_relu",
        "tau": 0.90,
        "nonlinear_layers": 1,
        "hidden_dim": 4,
        "alpha": 0.5,
        "lambda": 1.0,
    },
]

# Temporary debugging output only.
DEBUG_DIR = Path(__file__).resolve().parent / "outputs"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

X, y = load_replication_data(
    PROJECT_ROOT / "data" / "replication_dataset.csv"
)


def initialize_model(model, y_train, tau):
    """
    Match the current project initialization.
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "weight" in name:
                parameter.normal_(mean=0.0, std=0.01)
            elif "bias" in name:
                parameter.zero_()

        final_layer = model.network[-1]
        final_layer.bias.fill_(
            float(np.quantile(y_train.to_numpy(), tau))
        )


def prepare_validation_cache():
    """
    Precompute expanding-window datasets for the 1980-1999
    validation period only.
    """
    forecast_dates = y.loc[
        VALIDATION_START:VALIDATION_END
    ].index

    cache = []

    for i, date in enumerate(forecast_dates):
        X_train_raw = X.loc[:date].iloc[:-1]
        y_train = y.loc[:date].iloc[:-1]
        X_forecast_raw = X.loc[[date]]

        X_train_std, X_forecast_std = standardize_train_forecast(
            X_train_raw,
            X_forecast_raw,
        )

        cache.append({
            "date": date,
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
            "n_features": X_train_std.shape[1],
        })

        if i % 50 == 0:
            print(
                f"Built validation cache: "
                f"{i}/{len(forecast_dates)}"
            )

    return cache


def train_one_window(
    model,
    X_train_tensor,
    y_train_tensor,
    tau,
    lam,
    epochs,
):
    """
    Train one expanding-window model with plain SGD.
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


def recursive_validation_forecasts(
    validation_cache,
    case,
    epochs_initial,
    epochs_update,
):
    """
    Produce recursive forecasts over 1980-1999 only.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = None
    rows = []

    for i, item in enumerate(validation_cache):
        if model is None:
            model = QuantileNetwork(
                n_features=item["n_features"],
                nonlinear_layers=case["nonlinear_layers"],
                hidden_dim=case["hidden_dim"],
                alpha=case["alpha"],
            ).to(device)

            initialize_model(
                model,
                item["y_train_series"],
                case["tau"],
            )

            epochs = epochs_initial
        else:
            epochs = epochs_update

        model = train_one_window(
            model=model,
            X_train_tensor=item["X_train_tensor"],
            y_train_tensor=item["y_train_tensor"],
            tau=case["tau"],
            lam=case["lambda"],
            epochs=epochs,
        )

        with torch.no_grad():
            forecast = float(
                model(item["X_forecast_tensor"]).item()
            )

        rows.append({
            "date": item["date"],
            "actual": item["actual"],
            "forecast": forecast,
        })

        if i % 50 == 0:
            print(
                f"{case['model']} | "
                f"{epochs_initial}/{epochs_update} epochs | "
                f"forecast {i}/{len(validation_cache)}"
            )

    return pd.DataFrame(rows)


def validation_pinball_loss(forecasts, tau):
    losses = pinball_loss_numpy(
        forecasts["actual"].to_numpy(),
        forecasts["forecast"].to_numpy(),
        tau,
    )

    return float(losses.mean())


def main():
    print("Building validation-only cache for 1980-1999...")
    validation_cache = prepare_validation_cache()

    summary_rows = []

    for case in MODEL_CASES:
        for schedule in EPOCH_SCHEDULES:
            epochs_initial = schedule["epochs_initial"]
            epochs_update = schedule["epochs_update"]

            print("\n" + "=" * 80)
            print(
                f"Model: {case['model']} | "
                f"tau={case['tau']} | "
                f"lr={LEARNING_RATE} | "
                f"epochs={epochs_initial}/{epochs_update}"
            )
            print("=" * 80)

            forecasts = recursive_validation_forecasts(
                validation_cache=validation_cache,
                case=case,
                epochs_initial=epochs_initial,
                epochs_update=epochs_update,
            )

            validation_loss = validation_pinball_loss(
                forecasts,
                case["tau"],
            )

            summary_rows.append({
                "model": case["model"],
                "tau": case["tau"],
                "learning_rate": LEARNING_RATE,
                "epochs_initial": epochs_initial,
                "epochs_update": epochs_update,
                "validation_pinball_loss": validation_loss,
            })

            print(
                f"Validation pinball loss: "
                f"{validation_loss:.8f}"
            )

    summary = pd.DataFrame(summary_rows)

    summary["rank_within_model"] = (
        summary.groupby("model")["validation_pinball_loss"]
        .rank(method="min")
        .astype(int)
    )

    summary = summary.sort_values(
        ["model", "validation_pinball_loss"]
    ).reset_index(drop=True)

    output_file = (
        DEBUG_DIR
        / "epoch_schedule_validation_comparison.csv"
    )
    summary.to_csv(output_file, index=False)

    print("\nValidation-only epoch-schedule comparison")
    print(summary.to_string(index=False))
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()