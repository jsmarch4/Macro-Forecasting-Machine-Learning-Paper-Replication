"""Construct replication dataset using FRED-MD transformations and original target scaling."""
import numpy as np
import pandas as pd

RAW_PATH = "data/2026-05-MD.csv"
OUTPUT_PATH = "data/replication_dataset.csv"
TARGET = "y_unrate_change_1m_ahead"

raw = pd.read_csv(RAW_PATH)
transform_row = raw.loc[raw["sasdate"] == "Transform:"].iloc[0]
df = raw.loc[raw["sasdate"] != "Transform:"].copy()
df["sasdate"] = pd.to_datetime(df["sasdate"])
df = df.set_index("sasdate").apply(pd.to_numeric, errors="coerce")


def fred_transform(x, tcode):
    x = x.astype(float)
    if pd.isna(tcode):
        return pd.Series(np.nan, index=x.index)
    tcode = int(tcode)
    if tcode == 1:
        return x.copy()
    if tcode == 2:
        return x.diff()
    if tcode == 3:
        return x.diff().diff()
    if tcode in (4, 5, 6):
        if x.min(skipna=True) <= 1e-6:
            return pd.Series(np.nan, index=x.index)
        logged = np.log(x)
        return {4: lambda: logged, 5: lambda: logged.diff(),
                6: lambda: logged.diff().diff()}[tcode]()
    if tcode == 7:
        return ((x - x.shift(1)) / x.shift(1)).diff()
    raise ValueError(f"Unknown transformation code: {tcode}")


X_transformed = pd.DataFrame(
    {col: fred_transform(df[col], transform_row[col]) for col in df.columns},
    index=df.index,
)

# The first two observations are unavailable for second-difference transforms.
X_transformed = X_transformed.iloc[2:].copy()
df = df.iloc[2:].copy()

excluded = [
    "ACOGNO", "ANDENOx", "TWEXAFEGSMTHx", "UMCSENTx", "VIXCLSx",
    "NONBORRES", "TOTRESNS",
]
X_transformed = X_transformed.drop(columns=excluded, errors="ignore")

# Match process_target(): standardize the complete transformed target
# BEFORE merging predictors and removing incomplete observations.
y = df["UNRATE"].shift(-1) - df["UNRATE"]
target_mean = np.nanmean(y.to_numpy())
target_std = np.nanstd(y.to_numpy(), ddof=0)
if not np.isfinite(target_std) or target_std <= 0:
    raise ValueError("Cannot standardize UNRATE target: invalid standard deviation")
y = (y - target_mean) / target_std

replication_data = X_transformed.copy()
replication_data[TARGET] = y
replication_data = replication_data.loc["1960-01-01":].copy()
replication_data = replication_data.replace([np.inf, -np.inf], np.nan)

missing_by_variable = replication_data.isna().sum()
missing_by_variable = missing_by_variable[missing_by_variable > 0]
rows_with_missing = replication_data.isna().any(axis=1)
print("\nMissing values before dropping incomplete rows:")
print(missing_by_variable)
print("Total missing values:", int(missing_by_variable.sum()))
print("Rows containing at least one missing value:", int(rows_with_missing.sum()))
print("\nDates dropped because of missing values:")
print(replication_data.index[rows_with_missing])

# Match format_data.py: remove incomplete rows; never fill missing predictors.
replication_data = replication_data.dropna().copy()
replication_data.to_csv(OUTPUT_PATH)

print("\nDataset construction complete.")
print(f"Target mean (before standardization): {target_mean:.8f}")
print(f"Target std (before standardization): {target_std:.8f}")
print(f"Replication dataset shape: {replication_data.shape}")
print(f"Date range: {replication_data.index.min()} to {replication_data.index.max()}")
print("Remaining missing values:", int(replication_data.isna().sum().sum()))
print(f"Saved {OUTPUT_PATH}")
