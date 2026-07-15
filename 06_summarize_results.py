import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

rows = []

for tau in quantiles:
    file = RESULTS_DIR / f"dnn_q{tau:.2f}_validation_results.csv"
    df = pd.read_csv(file)

    best = df.loc[df["validation_loss"].idxmin()]

    rows.append({
        "tau": tau,
        "best_nonlinear_layers": int(best["nonlinear_layers"]),
        "best_hidden_dim": int(best["hidden_dim"]),
        "best_alpha": float(best["alpha"]),
        "best_lambda": float(best["lambda"]),
        "best_validation_loss": float(best["validation_loss"]),
    })

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS_DIR / "best_hyperparameters_by_quantile.csv", index=False)

print("\nBest validation model by quantile")
print(summary)

print("\nAverage validation loss by alpha")
alpha_summary = []

for tau in quantiles:
    file = RESULTS_DIR / f"dnn_q{tau:.2f}_validation_results.csv"
    df = pd.read_csv(file)

    grouped = (
        df.groupby("alpha")["validation_loss"]
        .mean()
        .reset_index()
    )

    grouped["tau"] = tau
    alpha_summary.append(grouped)

alpha_summary = pd.concat(alpha_summary)
print(alpha_summary)

alpha_summary.to_csv(
    RESULTS_DIR / "average_validation_loss_by_activation.csv",
    index=False
)