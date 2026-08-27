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
from train_utils import train_model


SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

DEVICE = torch.device("cpu")

VALIDATION_START = pd.Timestamp("1980-01-01")

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

LAMBDA_GRID = np.exp(
    np.linspace(np.log(0.2), np.log(10.0), 40)
)

LEARNING_RATE = 0.0001

COMPLEXITY_EPOCHS = {
    "linear_activation": 5000,
    "dnn": 7500,
}

LINEAR_ARCHITECTURES = [
    {"nonlinear_layers": 0, "hidden_dim": 0, "alpha": 1.0},
    {"nonlinear_layers": 1, "hidden_dim": 2, "alpha": 1.0},
    {"nonlinear_layers": 1, "hidden_dim": 4, "alpha": 1.0},
    {"nonlinear_layers": 1, "hidden_dim": 8, "alpha": 1.0},
    {"nonlinear_layers": 2, "hidden_dim": 2, "alpha": 1.0},
    {"nonlinear_layers": 2, "hidden_dim": 4, "alpha": 1.0},
    {"nonlinear_layers": 2, "hidden_dim": 8, "alpha": 1.0},
]

DNN_ARCHITECTURES = [
    {
        "nonlinear_layers": layers,
        "hidden_dim": dim,
        "alpha": alpha,
    }
    for layers in [1, 2]
    for dim in [2, 4, 8]
    for alpha in [0.0, 0.5, 1.0]
]

ARCHITECTURES = {
    "linear_activation": LINEAR_ARCHITECTURES,
    "dnn": DNN_ARCHITECTURES,
}


parser = argparse.ArgumentParser()

parser.add_argument(
    "--family",
    choices=["linear_activation", "dnn"],
    required=True,
)

parser.add_argument(
    "--quantile",
    type=float,
    choices=QUANTILES,
    required=True,
)

parser.add_argument(
    "--architecture-index",
    type=int,
    required=True,
)

args = parser.parse_args()

family = args.family
tau = float(args.quantile)
architecture_index = int(args.architecture_index)

architecture_grid = ARCHITECTURES[family]

if not (0 <= architecture_index < len(architecture_grid)):
    raise ValueError(
        f"Architecture index {architecture_index} invalid for {family}; "
        f"expected 0 through {len(architecture_grid) - 1}."
    )

architecture = architecture_grid[architecture_index]

nonlinear_layers = int(architecture["nonlinear_layers"])
hidden_dim = int(architecture["hidden_dim"])
alpha = float(architecture["alpha"])


OUTPUT_DIR = (
    Path("results")
    / "complexity"
    / "variance_fits"
    / family
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"q{tau:.2f}_arch{architecture_index:02d}.csv"
)


X, y = load_replication_data()

mask = X.index < VALIDATION_START

X_raw = X.loc[mask].copy()
y_raw = y.loc[mask].copy()

if X_raw.empty:
    raise ValueError("No observations exist before 1980-01.")

X_std, _ = standardize_train_forecast(
    X_raw,
    X_raw,
)

y_std, _, _, _ = standardize_target_train_forecast(
    y_raw,
    float(y_raw.iloc[-1]),
)

X_tensor = torch.tensor(
    X_std.to_numpy(),
    dtype=torch.float32,
    device=DEVICE,
)

y_tensor = torch.tensor(
    y_std.to_numpy(),
    dtype=torch.float32,
    device=DEVICE,
)


def initialize_model(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "weight" in name:
                parameter.normal_(mean=0.0, std=0.01)
            elif "bias" in name:
                parameter.zero_()

        model.network[-1].bias.fill_(
            float(np.quantile(y_std.to_numpy(), tau))
        )


def fit_variance(lam):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = QuantileNetwork(
        n_features=X_tensor.shape[1],
        nonlinear_layers=nonlinear_layers,
        hidden_dim=hidden_dim,
        alpha=alpha,
    ).to(DEVICE)

    initialize_model(model)

    model = train_model(
        model=model,
        X_train_tensor=X_tensor,
        y_train_tensor=y_tensor,
        tau=tau,
        lam=float(lam),
        epochs=COMPLEXITY_EPOCHS[family],
        lr=LEARNING_RATE,
    )

    with torch.no_grad():
        fitted = (
            model(X_tensor)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    return float(np.var(fitted, ddof=0))


rows = []

all_lambdas = list(LAMBDA_GRID) + [0.0]

print("=" * 80)
print(
    f"Family: {family}\n"
    f"Tau: {tau:.2f}\n"
    f"Architecture index: {architecture_index}\n"
    f"Layers: {nonlinear_layers}\n"
    f"Hidden dimension: {hidden_dim}\n"
    f"Alpha: {alpha}\n"
    f"Epochs: {COMPLEXITY_EPOCHS[family]}\n"
    f"Learning rate: {LEARNING_RATE}"
)
print("=" * 80)

for i, lam in enumerate(all_lambdas):
    print(f"{i + 1}/{len(all_lambdas)} lambda={lam:.12g}")

    variance = fit_variance(lam)

    rows.append({
        "family": family,
        "tau": tau,
        "architecture_index": architecture_index,
        "nonlinear_layers": nonlinear_layers,
        "hidden_dim": hidden_dim,
        "alpha": alpha,
        "lambda": float(lam),
        "fitted_variance": variance,
    })

    print(f"    variance = {variance:.10f}")

results = pd.DataFrame(rows)

lambda_zero_variance = float(
    results.loc[
        np.isclose(results["lambda"], 0.0),
        "fitted_variance",
    ].iloc[0]
)

results["architecture_lambda_zero_fitted_variance"] = (
    lambda_zero_variance
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:")
print(OUTPUT_FILE)

print(
    "\nLambda=0 fitted variance: "
    f"{lambda_zero_variance:.10f}"
    )