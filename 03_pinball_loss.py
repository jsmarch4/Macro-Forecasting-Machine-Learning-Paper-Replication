import pandas as pd
import numpy as np

df = pd.read_csv(
    "Results/naive_quantile_forecasts.csv",
    index_col=0,
    parse_dates=True
)

quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

results = []

for tau in quantiles:

    forecast_col = f"q{tau:.2f}"

    q = df[forecast_col]
    y = df["actual"]

    loss = np.where(
        y >= q,
        tau * (y - q),
        (1 - tau) * (q - y)
    )

    results.append({
        "quantile": tau,
        "average_pinball_loss": loss.mean()
    })

results = pd.DataFrame(results)

print(results)

print("\nOverall average pinball loss:")
print(results["average_pinball_loss"].mean())


results.to_csv("Results/naive_pinball_loss.csv", index=False)

