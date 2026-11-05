from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/replication_dataset.csv")
OUTPUT_DIR = Path("results/benchmark_audit")
TARGET = "y_unrate_change_1m_ahead"

TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

PERIODS = {
    "validation": ("1980-02-01", "2000-01-01"),
    "test": ("2000-02-01", "2019-12-01"),
}

PAPER = {
    "validation": (95.20, 81.57, 65.62, 60.82, 70.29, 89.41, 107.81),
    "test": (80.65, 63.69, 54.43, 54.04, 61.63, 79.56, 95.59),
}

def main():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    y = df[TARGET].dropna().astype(float).sort_index()

    if y.index.has_duplicates:
        raise ValueError("Duplicate target dates")

    rows = []

    for period, (start, end) in PERIODS.items():
        dates = y.loc[start:end].index

        for date in dates:
            history = y.loc[:date].iloc[:-1]
            actual = float(y.loc[date])

            for tau in TAUS:
                forecast = float(history.quantile(tau))
                error = actual - forecast
                loss = max(tau * error, (tau - 1) * error)

                rows.append({
                    "period": period,
                    "date": date,
                    "tau": tau,
                    "n_history": len(history),
                    "actual": actual,
                    "forecast": forecast,
                    "error": error,
                    "pinball_loss": loss,
                    "normalized_loss": 100 * loss / (tau * (1 - tau)),
                })

    monthly = pd.DataFrame(rows)

    summary = (
        monthly.groupby(["period", "tau"], sort=False)
        .agg(
            n_obs=("date", "size"),
            benchmark_loss=("normalized_loss", "mean"),
        )
        .reset_index()
    )

    summary["paper_loss"] = summary.apply(
        lambda r: PAPER[r["period"]][TAUS.index(r["tau"])],
        axis=1,
    )
    summary["difference_from_paper"] = (
        summary["benchmark_loss"] - summary["paper_loss"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly.to_csv(
        OUTPUT_DIR / "monthly_forecasts_losses.csv", index=False
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("Saved results to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
