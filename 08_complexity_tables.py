"""
08_complexity_tables.py

Build preliminary complexity-index mappings and paper-style validation/test
tables for BOTH:
    1. linear-activation networks
    2. deep neural networks

The candidate grid remains:
    lambda in {0.1, 1.0, 10.0}

The complexity reference is computed separately at lambda = 0:

    r(candidate)
        = fitted_variance(candidate)
          / fitted_variance(same architecture, lambda=0)

Fitted variance is computed from fitted values on the fixed initial
pre-validation sample (all observations before 1980-01). Lambda = 0 is not
added to the validation-selection grid.

Thus:
    r approximately 0 -> almost-flat fitted quantile
    r = 1             -> most variable candidate in that family/quantile grid

For every target r in {0.1, ..., 1.0}, the script selects the candidate with
the closest achieved r. Ties are broken by:
    1. lower validation loss
    2. smaller lambda
    3. fewer layers
    4. smaller hidden dimension
    5. smaller alpha

The r = 0 row is the recursive unconditional-quantile benchmark.

Outputs
-------
results/complexity/
    linear_activation_complexity_candidates.csv
    linear_activation_complexity_mapping.csv
    dnn_complexity_candidates.csv
    dnn_complexity_mapping.csv

results/complexity/forecasts/{validation,test}/
    forecast files for selected representative candidates

results/complexity/tables/
    table_validation_linear_activation_detailed.csv
    table_validation_linear_activation_formatted.csv
    table_test_linear_activation_detailed.csv
    table_test_linear_activation_formatted.csv
    table_validation_dnn_detailed.csv
    table_validation_dnn_formatted.csv
    table_test_dnn_detailed.csv
    table_test_dnn_formatted.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch

from data_utils import (
    load_replication_data,
    standardize_train_forecast,
    standardize_target_train_forecast,
)
from losses import pinball_loss_numpy
from models import QuantileNetwork
from train_utils import train_model


# ---------------------------------------------------------------------
# Reproducibility and runtime
# ---------------------------------------------------------------------

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

# These tiny networks are often faster with one CPU thread.
torch.set_num_threads(1)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "complexity"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
TABLE_DIR = OUTPUT_DIR / "tables"

for directory in [OUTPUT_DIR, FORECAST_DIR, TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

VALIDATION_START = pd.Timestamp("1980-01-01")
VALIDATION_END = pd.Timestamp("1999-12-01")
TEST_START = pd.Timestamp("2000-01-01")
TEST_END = None

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TARGET_R_GRID = [round(value, 1) for value in np.arange(0.1, 1.01, 0.1)]

THREE_LAMBDA_GRID = np.array([0.1, 1.0, 10.0], dtype=float)

LEARNING_RATE = 0.001
EPOCHS_INITIAL = 500
EPOCHS_UPDATE = 100

# The paper normalizes loss by tau * (1 - tau).
# Multiplying by 100 produces readable paper-style numbers.
DISPLAY_SCALE = 100.0

# The paper reports HAC standard errors but does not clearly specify bandwidth.
# None uses the common automatic Newey-West lag rule.
HAC_MAXLAGS = None

FAMILIES = {
    "linear_activation": {
        "search_file": RESULTS_DIR
        / "linear_activation_hyperparameter_search_results.csv",
    },
    "dnn": {
        "search_file": RESULTS_DIR / "dnn_hyperparameter_search_results.csv",
    },
}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

X, y = load_replication_data()


@dataclass(frozen=True)
class ModelSpec:
    family: str
    tau: float
    nonlinear_layers: int
    hidden_dim: int
    alpha: float
    lam: float

    @property
    def key(self) -> Tuple:
        return (
            self.family,
            round(self.tau, 8),
            self.nonlinear_layers,
            self.hidden_dim,
            round(self.alpha, 8),
            round(self.lam, 12),
        )


# ---------------------------------------------------------------------
# Shared model helpers
# ---------------------------------------------------------------------

def initialize_model(
    model: QuantileNetwork,
    y_train: pd.Series,
    tau: float,
) -> None:
    """
    Match the current project initialization:
      - all weights ~ Normal(0, 0.01)
      - all biases = 0
      - final bias = historical tau-quantile
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


def make_model(spec: ModelSpec, n_features: int) -> QuantileNetwork:
    model = QuantileNetwork(
        n_features=n_features,
        nonlinear_layers=spec.nonlinear_layers,
        hidden_dim=spec.hidden_dim,
        alpha=spec.alpha,
    ).to(DEVICE)
    return model


def build_forecast_cache(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None,
) -> list[dict]:
    """
    Precompute expanding-window standardized tensors.
    """
    if end_date is None:
        dates = y.loc[start_date:].index
    else:
        dates = y.loc[start_date:end_date].index

    cache = []

    for i, date in enumerate(dates):
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
            "actual_std": float(actual_std),
            "actual_raw": actual_raw,
            "y_mean": float(y_mean),
            "y_std": float(y_std),
            "n_features": X_train_std.shape[1],
        })

        if i % 100 == 0:
            print(f"  Cache {i}/{len(dates)}")

    return cache


def recursive_forecasts(
    cache: list[dict],
    spec: ModelSpec,
) -> pd.DataFrame:
    """
    Recursive expanding-window forecasts with warm-started model weights.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = None
    rows = []

    for i, item in enumerate(cache):
        if model is None:
            model = make_model(spec, item["n_features"])
            initialize_model(
                model=model,
                y_train=item["y_train_series"],
                tau=spec.tau,
            )
            epochs = EPOCHS_INITIAL
        else:
            epochs = EPOCHS_UPDATE

        model = train_model(
            model=model,
            X_train_tensor=item["X_train_tensor"],
            y_train_tensor=item["y_train_tensor"],
            tau=spec.tau,
            lam=spec.lam,
            epochs=epochs,
            lr=LEARNING_RATE,
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
            "actual": item["actual_raw"],
            f"q{spec.tau:.2f}": forecast_raw,
            "family": spec.family,
            "tau": spec.tau,
            "nonlinear_layers": spec.nonlinear_layers,
            "hidden_dim": spec.hidden_dim,
            "alpha": spec.alpha,
            "lambda": spec.lam,
        })

        if i % 100 == 0:
            print(
                f"    {spec.family}, tau={spec.tau:.2f}, "
                f"layers={spec.nonlinear_layers}, "
                f"dim={spec.hidden_dim}, alpha={spec.alpha}, "
                f"lambda={spec.lam}: {i}/{len(cache)}"
            )

    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------
# Complexity calculation
# ---------------------------------------------------------------------

def initial_sample_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
    pd.Series,
]:
    """
    Fixed initial estimation sample: all observations strictly before 1980-01.

    Standardization is computed using only this initial sample.
    """
    mask = X.index < VALIDATION_START

    X_initial_raw = X.loc[mask].copy()
    y_initial = y.loc[mask].copy()

    if X_initial_raw.empty:
        raise ValueError("No observations exist before 1980-01.")

    # Apply the same predictor normalization used by 04 and 05.
    X_initial_std, _ = standardize_train_forecast(
        X_initial_raw,
        X_initial_raw,
    )

    # Apply the same target standardization used by 04 and 05.
    y_initial_std, _, _, _ = standardize_target_train_forecast(
        y_initial,
        float(y_initial.iloc[-1]),
    )

    X_tensor = torch.tensor(
        X_initial_std.to_numpy(),
        dtype=torch.float32,
        device=DEVICE,
    )
    y_tensor = torch.tensor(
        y_initial_std.to_numpy(),
        dtype=torch.float32,
        device=DEVICE,
    )

    return X_tensor, y_tensor, y_initial_std


def fitted_variance_for_candidate(spec: ModelSpec) -> float:
    """
    Fit one candidate once on the fixed pre-1980 sample and compute
    the population variance (ddof=0) of its in-sample fitted quantiles.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_tensor, y_tensor, y_series = initial_sample_tensors()

    model = make_model(spec, X_tensor.shape[1])
    initialize_model(model, y_series, spec.tau)

    model = train_model(
        model=model,
        X_train_tensor=X_tensor,
        y_train_tensor=y_tensor,
        tau=spec.tau,
        lam=spec.lam,
        epochs=EPOCHS_INITIAL,
        lr=LEARNING_RATE,
    )

    with torch.no_grad():
        fitted = model(X_tensor).detach().cpu().numpy()

    return float(np.var(fitted, ddof=0))


def read_three_lambda_search(family: str) -> pd.DataFrame:
    """
    Read the stored hyperparameter-search results and retain only
    lambda in {0.1, 1, 10}.
    """
    path = FAMILIES[family]["search_file"]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing hyperparameter search file: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "tau",
        "nonlinear_layers",
        "hidden_dim",
        "alpha",
        "lambda",
        "validation_loss",
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    lambda_mask = np.zeros(len(df), dtype=bool)

    for lam in THREE_LAMBDA_GRID:
        lambda_mask |= np.isclose(
            df["lambda"].to_numpy(dtype=float),
            lam,
            rtol=0,
            atol=1e-10,
        )

    df = df.loc[lambda_mask].copy()
    df = df[df["tau"].isin(QUANTILES)].copy()

    if df.empty:
        raise ValueError(
            f"No three-grid lambda rows found in {path}."
        )

    # Remove accidental duplicate rows from repeated runs.
    df = (
        df.sort_values("validation_loss")
        .drop_duplicates(
            subset=[
                "tau",
                "nonlinear_layers",
                "hidden_dim",
                "alpha",
                "lambda",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    df["family"] = family

    return df


def build_complexity_candidates(family: str) -> pd.DataFrame:
    """
    Compute each candidate's fitted variance and normalize it by the
    fitted variance of the same architecture at lambda = 0.

    Lambda = 0 is used only as the complexity reference. It is not added
    to the validation-selection grid.
    """
    search = read_three_lambda_search(family)
    candidate_variances = []
    reference_variances = []

    # Avoid fitting the same lambda=0 reference once for every penalized lambda.
    zero_variance_cache: dict[tuple, float] = {}

    total = len(search)

    for index, row in search.iterrows():
        spec = ModelSpec(
            family=family,
            tau=float(row["tau"]),
            nonlinear_layers=int(row["nonlinear_layers"]),
            hidden_dim=int(row["hidden_dim"]),
            alpha=float(row["alpha"]),
            lam=float(row["lambda"]),
        )

        print(
            f"Complexity fit {index + 1}/{total}: "
            f"{family}, tau={spec.tau:.2f}, "
            f"layers={spec.nonlinear_layers}, "
            f"dim={spec.hidden_dim}, alpha={spec.alpha}, "
            f"lambda={spec.lam}"
        )

        candidate_variance = fitted_variance_for_candidate(spec)
        candidate_variances.append(candidate_variance)

        architecture_key = (
            spec.family,
            round(spec.tau, 8),
            spec.nonlinear_layers,
            spec.hidden_dim,
            round(spec.alpha, 8),
        )

        if architecture_key not in zero_variance_cache:
            zero_spec = ModelSpec(
                family=spec.family,
                tau=spec.tau,
                nonlinear_layers=spec.nonlinear_layers,
                hidden_dim=spec.hidden_dim,
                alpha=spec.alpha,
                lam=0.0,
            )

            print(
                "  Computing lambda=0 reference for "
                f"layers={spec.nonlinear_layers}, "
                f"dim={spec.hidden_dim}, alpha={spec.alpha}"
            )

            zero_variance_cache[architecture_key] = (
                fitted_variance_for_candidate(zero_spec)
            )

        reference_variances.append(
            zero_variance_cache[architecture_key]
        )

    search["initial_fitted_variance"] = candidate_variances
    search["lambda_zero_fitted_variance"] = reference_variances

    denominator = search["lambda_zero_fitted_variance"].to_numpy()
    numerator = search["initial_fitted_variance"].to_numpy()

    search["achieved_r"] = np.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )

    # The theoretical index lies in [0, 1]. Small overshoots can occur
    # because the neural-network fits are numerical rather than exact.
    search["achieved_r"] = search["achieved_r"].clip(0.0, 1.0)

    output_path = (
        OUTPUT_DIR / f"{family}_complexity_candidates.csv"
    )
    search.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return search


def map_target_complexities(
    family: str,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Choose one representative candidate nearest each target r.
    """
    rows = []

    for tau in QUANTILES:
        tau_candidates = candidates[
            np.isclose(candidates["tau"], tau)
        ].copy()

        if tau_candidates.empty:
            raise ValueError(
                f"No candidates for {family}, tau={tau:.2f}."
            )

        for target_r in TARGET_R_GRID:
            tau_candidates["distance_to_target_r"] = (
                tau_candidates["achieved_r"] - target_r
            ).abs()

            chosen = tau_candidates.sort_values(
                [
                    "distance_to_target_r",
                    "validation_loss",
                    "lambda",
                    "nonlinear_layers",
                    "hidden_dim",
                    "alpha",
                ],
                ascending=[True, True, True, True, True, True],
            ).iloc[0]

            rows.append({
                "family": family,
                "tau": tau,
                "target_r": target_r,
                "achieved_r": float(chosen["achieved_r"]),
                "distance_to_target_r": float(
                    chosen["distance_to_target_r"]
                ),
                "nonlinear_layers": int(
                    chosen["nonlinear_layers"]
                ),
                "hidden_dim": int(chosen["hidden_dim"]),
                "alpha": float(chosen["alpha"]),
                "lambda": float(chosen["lambda"]),
                "initial_fitted_variance": float(
                    chosen["initial_fitted_variance"]
                ),
                "lambda_zero_fitted_variance": float(
                    chosen["lambda_zero_fitted_variance"]
                ),
                "validation_loss": float(
                    chosen["validation_loss"]
                ),
            })

    mapping = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / f"{family}_complexity_mapping.csv"
    mapping.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return mapping


# ---------------------------------------------------------------------
# Naive benchmark and loss calculations
# ---------------------------------------------------------------------

def recursive_naive_forecasts(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """
    Expanding-window unconditional empirical quantile forecasts.
    """
    if end_date is None:
        dates = y.loc[start_date:].index
    else:
        dates = y.loc[start_date:end_date].index

    rows = []

    for date in dates:
        y_history = y.loc[:date].iloc[:-1]

        row = {
            "date": date,
            "actual": float(y.loc[date]),
        }

        for tau in QUANTILES:
            row[f"q{tau:.2f}"] = float(
                np.quantile(y_history.to_numpy(), tau)
            )

        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def pinball_loss(
    actual: np.ndarray,
    forecast: np.ndarray,
    tau: float,
) -> np.ndarray:
    return pinball_loss_numpy(actual, forecast, tau)


def average_pinball_loss(
    forecast_df: pd.DataFrame,
    tau: float,
) -> float:
    column = f"q{tau:.2f}"
    losses = pinball_loss(
        forecast_df["actual"].to_numpy(),
        forecast_df[column].to_numpy(),
        tau,
    )
    return float(np.mean(losses))


def choose_hac_lag(n_obs: int) -> int:
    if HAC_MAXLAGS is not None:
        return int(HAC_MAXLAGS)

    return max(
        0,
        int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))),
    )


def hac_standard_error_of_mean(
    values: Iterable[float],
) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]

    n_obs = len(values)

    if n_obs < 2:
        return np.nan

    centered = values - values.mean()
    max_lag = min(choose_hac_lag(n_obs), n_obs - 1)

    long_run_variance = (
        np.dot(centered, centered) / n_obs
    )

    for lag in range(1, max_lag + 1):
        covariance = (
            np.dot(centered[lag:], centered[:-lag])
            / n_obs
        )
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance

    return math.sqrt(
        max(long_run_variance, 0.0) / n_obs
    )


# ---------------------------------------------------------------------
# Representative recursive forecasts
# ---------------------------------------------------------------------

def row_to_spec(row: pd.Series) -> ModelSpec:
    return ModelSpec(
        family=str(row["family"]),
        tau=float(row["tau"]),
        nonlinear_layers=int(row["nonlinear_layers"]),
        hidden_dim=int(row["hidden_dim"]),
        alpha=float(row["alpha"]),
        lam=float(row["lambda"]),
    )


def run_representative_forecasts(
    family: str,
    mapping: pd.DataFrame,
    sample_name: str,
    cache: list[dict],
) -> Dict[Tuple, pd.DataFrame]:
    """
    Run only unique representative specifications.

    If several target-r values map to the same candidate, that candidate
    is trained once and its forecast path is reused.
    """
    sample_dir = FORECAST_DIR / sample_name / family
    sample_dir.mkdir(parents=True, exist_ok=True)

    forecasts_by_spec: Dict[Tuple, pd.DataFrame] = {}

    unique_specs = (
        mapping[
            [
                "family",
                "tau",
                "nonlinear_layers",
                "hidden_dim",
                "alpha",
                "lambda",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"\n{family}, {sample_name}: "
        f"{len(unique_specs)} unique representative specifications"
    )

    for i, row in unique_specs.iterrows():
        spec = row_to_spec(row)

        print(
            f"\nRepresentative {i + 1}/{len(unique_specs)}: "
            f"tau={spec.tau:.2f}, "
            f"layers={spec.nonlinear_layers}, "
            f"dim={spec.hidden_dim}, alpha={spec.alpha}, "
            f"lambda={spec.lam}"
        )

        forecasts = recursive_forecasts(cache, spec)
        forecasts_by_spec[spec.key] = forecasts

        spec_file = (
            sample_dir
            / (
                f"tau_{spec.tau:.2f}"
                f"_layers_{spec.nonlinear_layers}"
                f"_dim_{spec.hidden_dim}"
                f"_alpha_{spec.alpha:g}"
                f"_lambda_{spec.lam:g}.csv"
            )
        )
        forecasts.to_csv(spec_file)

    # Save target-r aliases as well, so each table row has an explicit file.
    for _, row in mapping.iterrows():
        spec = row_to_spec(row)
        forecasts = forecasts_by_spec[spec.key]

        target_file = (
            sample_dir
            / (
                f"{family}_r{float(row['target_r']):.1f}"
                f"_q{spec.tau:.2f}.csv"
            )
        )
        forecasts.to_csv(target_file)

    return forecasts_by_spec


# ---------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------

def build_complexity_table(
    family: str,
    sample_name: str,
    mapping: pd.DataFrame,
    model_forecasts: Dict[Tuple, pd.DataFrame],
    naive_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build paper-style complexity-by-quantile table.

    Row r=0:
        normalized absolute naive loss

    Rows r>0:
        normalized model-minus-naive loss differential
        with HAC standard error
    """
    detailed_rows = []
    formatted_rows = []

    # r = 0 benchmark row
    benchmark_row = {
        "family": family,
        "sample": sample_name,
        "complexity": 0.0,
    }

    for tau in QUANTILES:
        normalization = tau * (1.0 - tau)

        naive_loss_series = pinball_loss(
            naive_forecasts["actual"].to_numpy(),
            naive_forecasts[f"q{tau:.2f}"].to_numpy(),
            tau,
        )

        estimate = (
            np.mean(naive_loss_series)
            / normalization
            * DISPLAY_SCALE
        )

        benchmark_row[f"{tau:.2f}"] = f"{estimate:.2f}"

        detailed_rows.append({
            "family": family,
            "sample": sample_name,
            "target_r": 0.0,
            "tau": tau,
            "achieved_r": 0.0,
            "estimate": estimate,
            "hac_standard_error": np.nan,
            "n_obs": len(naive_loss_series),
            "nonlinear_layers": np.nan,
            "hidden_dim": np.nan,
            "alpha": np.nan,
            "lambda": np.nan,
            "validation_loss": np.nan,
        })

    formatted_rows.append(benchmark_row)

    # r > 0 rows
    for target_r in TARGET_R_GRID:
        formatted_row = {
            "family": family,
            "sample": sample_name,
            "complexity": target_r,
        }

        for tau in QUANTILES:
            mapping_row = mapping[
                np.isclose(mapping["tau"], tau)
                & np.isclose(mapping["target_r"], target_r)
            ].iloc[0]

            spec = row_to_spec(mapping_row)
            model_df = model_forecasts[spec.key]

            aligned = naive_forecasts[
                ["actual", f"q{tau:.2f}"]
            ].rename(
                columns={f"q{tau:.2f}": "naive_forecast"}
            ).join(
                model_df[[f"q{tau:.2f}"]].rename(
                    columns={
                        f"q{tau:.2f}": "model_forecast"
                    }
                ),
                how="inner",
            )

            naive_losses = pinball_loss(
                aligned["actual"].to_numpy(),
                aligned["naive_forecast"].to_numpy(),
                tau,
            )

            model_losses = pinball_loss(
                aligned["actual"].to_numpy(),
                aligned["model_forecast"].to_numpy(),
                tau,
            )

            normalized_differential = (
                (model_losses - naive_losses)
                / (tau * (1.0 - tau))
                * DISPLAY_SCALE
            )

            estimate = float(
                np.mean(normalized_differential)
            )
            standard_error = float(
                hac_standard_error_of_mean(
                    normalized_differential
                )
            )

            formatted_row[f"{tau:.2f}"] = (
                f"{estimate:.2f}\n"
                f"({standard_error:.2f})"
            )

            detailed_rows.append({
                "family": family,
                "sample": sample_name,
                "target_r": target_r,
                "tau": tau,
                "achieved_r": float(
                    mapping_row["achieved_r"]
                ),
                "distance_to_target_r": float(
                    mapping_row["distance_to_target_r"]
                ),
                "estimate": estimate,
                "hac_standard_error": standard_error,
                "n_obs": len(aligned),
                "nonlinear_layers": int(
                    mapping_row["nonlinear_layers"]
                ),
                "hidden_dim": int(
                    mapping_row["hidden_dim"]
                ),
                "alpha": float(mapping_row["alpha"]),
                "lambda": float(mapping_row["lambda"]),
                "validation_loss": float(
                    mapping_row["validation_loss"]
                ),
            })

        formatted_rows.append(formatted_row)

    detailed = pd.DataFrame(detailed_rows)
    formatted = pd.DataFrame(formatted_rows)

    return detailed, formatted


def save_table_outputs(
    family: str,
    sample_name: str,
    detailed: pd.DataFrame,
    formatted: pd.DataFrame,
) -> None:
    detailed_path = (
        TABLE_DIR
        / f"table_{sample_name}_{family}_detailed.csv"
    )
    formatted_path = (
        TABLE_DIR
        / f"table_{sample_name}_{family}_formatted.csv"
    )
    latex_path = (
        TABLE_DIR
        / f"table_{sample_name}_{family}.tex"
    )

    detailed.to_csv(detailed_path, index=False)
    formatted.to_csv(formatted_path, index=False)

    formatted.to_latex(
        latex_path,
        index=False,
        escape=False,
        column_format="lll" + "c" * len(QUANTILES),
        caption=(
            f"{sample_name.title()} forecast accuracy by "
            f"complexity for {family}."
        ),
        label=f"tab:{sample_name}_{family}",
    )

    print(f"\nSaved: {detailed_path}")
    print(f"Saved: {formatted_path}")
    print(f"Saved: {latex_path}")

    print("\n" + "=" * 90)
    print(f"{sample_name.upper()} TABLE: {family}")
    print("=" * 90)
    print(formatted.to_string(index=False))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
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

    print("\nBuilding naive validation forecasts...")
    naive_validation = recursive_naive_forecasts(
        VALIDATION_START,
        VALIDATION_END,
    )

    print("\nBuilding naive test forecasts...")
    naive_test = recursive_naive_forecasts(
        TEST_START,
        TEST_END,
    )

    naive_validation.to_csv(
        OUTPUT_DIR / "naive_validation_forecasts.csv"
    )
    naive_test.to_csv(
        OUTPUT_DIR / "naive_test_forecasts.csv"
    )

    for family in FAMILIES:
        print("\n" + "#" * 100)
        print(f"FAMILY: {family}")
        print("#" * 100)

        candidates = build_complexity_candidates(family)
        mapping = map_target_complexities(
            family,
            candidates,
        )

        validation_forecasts = run_representative_forecasts(
            family=family,
            mapping=mapping,
            sample_name="validation",
            cache=validation_cache,
        )

        test_forecasts = run_representative_forecasts(
            family=family,
            mapping=mapping,
            sample_name="test",
            cache=test_cache,
        )

        validation_detailed, validation_formatted = (
            build_complexity_table(
                family=family,
                sample_name="validation",
                mapping=mapping,
                model_forecasts=validation_forecasts,
                naive_forecasts=naive_validation,
            )
        )

        test_detailed, test_formatted = (
            build_complexity_table(
                family=family,
                sample_name="test",
                mapping=mapping,
                model_forecasts=test_forecasts,
                naive_forecasts=naive_test,
            )
        )

        save_table_outputs(
            family,
            "validation",
            validation_detailed,
            validation_formatted,
        )

        save_table_outputs(
            family,
            "test",
            test_detailed,
            test_formatted,
        )

    print("\nComplexity mapping and tables complete.")


if __name__ == "__main__":
    main()