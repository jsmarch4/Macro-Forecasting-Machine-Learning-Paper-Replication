from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
DNN_DIR = RESULTS_DIR / "dnn"

VALIDATION_RESULTS_DIR = DNN_DIR / "validation" / "results"
COMBINED_DIR = DNN_DIR / "combined"
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def load_validation_results(tau: float) -> pd.DataFrame:
    path = VALIDATION_RESULTS_DIR / f"q{tau:.2f}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing validation-results file: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "tau",
        "nonlinear_layers",
        "hidden_dim",
        "alpha",
        "lambda",
        "validation_loss",
    }
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return df


def main() -> None:
    best_rows = []
    alpha_rows = []

    for tau in QUANTILES:
        df = load_validation_results(tau)

        best = df.loc[df["validation_loss"].idxmin()]

        best_rows.append({
            "tau": tau,
            "best_nonlinear_layers": int(best["nonlinear_layers"]),
            "best_hidden_dim": int(best["hidden_dim"]),
            "best_alpha": float(best["alpha"]),
            "best_lambda": float(best["lambda"]),
            "best_validation_loss": float(best["validation_loss"]),
        })

        grouped = (
            df.groupby("alpha", as_index=False)["validation_loss"]
            .mean()
            .rename(
                columns={
                    "validation_loss": "average_validation_loss"
                }
            )
        )
        grouped.insert(0, "tau", tau)
        alpha_rows.append(grouped)

    best_summary = pd.DataFrame(best_rows)
    alpha_summary = pd.concat(alpha_rows, ignore_index=True)

    best_path = (
        COMBINED_DIR
        / "best_hyperparameters_by_quantile.csv"
    )
    alpha_path = (
        COMBINED_DIR
        / "average_validation_loss_by_activation.csv"
    )

    best_summary.to_csv(best_path, index=False)
    alpha_summary.to_csv(alpha_path, index=False)

    print("\nBest validation model by quantile")
    print(best_summary.to_string(index=False))

    print("\nAverage validation loss by alpha")
    print(alpha_summary.to_string(index=False))

    print(f"\nSaved: {best_path}")
    print(f"Saved: {alpha_path}")


if __name__ == "__main__":
    main()