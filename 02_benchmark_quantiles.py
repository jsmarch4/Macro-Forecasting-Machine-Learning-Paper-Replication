import pandas as pd
import numpy as np

# Load cleaned dataset
data = pd.read_csv(
    "Data/replication_dataset.csv",
    index_col=0,
    parse_dates=True
)

target = "y_unrate_change_1m_ahead"
y = data[target]

# Quantiles used in the paper
quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Out-of-sample period
forecast_start = "1980-01-01"

# Store forecasts
forecasts = []

for date in y.loc[forecast_start:].index:
    # Only use information before the forecast date
    past_y = y.loc[:date].iloc[:-1]

    row = {"date": date}

    for tau in quantiles:
        row[f"q{tau:.2f}"] = past_y.quantile(tau)

    row["actual"] = y.loc[date]
    forecasts.append(row)

forecast_df = pd.DataFrame(forecasts).set_index("date")

print(forecast_df.head())
print(forecast_df.shape)

forecast_df.to_csv("Results/naive_quantile_forecasts.csv")

print("Saved naive_quantile_forecasts.csv")