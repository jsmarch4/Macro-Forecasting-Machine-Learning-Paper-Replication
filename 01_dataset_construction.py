import pandas as pd
import numpy as np


# ------------------------------------------------------------
# Load raw FRED-MD file
# ------------------------------------------------------------

raw = pd.read_csv("data/2026-05-MD.csv")

# Save transformation codes before dropping Transform row
transform_row = raw[raw["sasdate"] == "Transform:"].iloc[0]

# Drop transformation row
df = raw[raw["sasdate"] != "Transform:"].copy()

# Parse dates
df["sasdate"] = pd.to_datetime(df["sasdate"])
df = df.set_index("sasdate")

# Convert all series to numeric
df = df.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------
# FRED-MD transformations
# Matches prepare_missing() / transxf() in
# Domenico's py_generate_freddataDG.py
# ------------------------------------------------------------

def fred_transform(x, tcode):
    """
    Apply the FRED-MD transformation used in the original
    forecasting-data generation code.
    """

    x = x.astype(float)
    n = len(x)

    # Initialize transformed series as missing
    y = pd.Series(
        np.full(n, np.nan),
        index=x.index,
        dtype=float,
    )

    if pd.isna(tcode):
        return y

    tcode = int(tcode)
    small = 1e-6

    # 1 => Level
    if tcode == 1:
        y = x.copy()

    # 2 => First difference
    elif tcode == 2:
        y = x.diff()

    # 3 => Second difference
    elif tcode == 3:
        y = x.diff().diff()

    # 4 => Natural log
    elif tcode == 4:
        if x.min(skipna=True) > small:
            y = np.log(x)

    # 5 => First difference of natural log
    elif tcode == 5:
        if x.min(skipna=True) > small:
            y = np.log(x).diff()

    # 6 => Second difference of natural log
    elif tcode == 6:
        if x.min(skipna=True) > small:
            y = np.log(x).diff().diff()

    # 7 => First difference of percent change
    elif tcode == 7:
        pct_change = (x - x.shift(1)) / x.shift(1)
        y = pct_change.diff()

    else:
        raise ValueError(
            f"Unknown transformation code: {tcode}"
        )

    return y


# ------------------------------------------------------------
# Apply transformations
# ------------------------------------------------------------

# Build all transformed columns together rather than inserting
# one at a time. This avoids pandas DataFrame fragmentation.
transformed_columns = {
    col: fred_transform(df[col], transform_row[col])
    for col in df.columns
}

X_transformed = pd.DataFrame(
    transformed_columns,
    index=df.index,
)


# ------------------------------------------------------------
# Remove first two observations
# ------------------------------------------------------------

# Some transformations require two lags. The original code
# removes the first two months after transformation.
X_transformed = X_transformed.iloc[2:].copy()
df = df.iloc[2:].copy()


# ------------------------------------------------------------
# Remove predictors excluded by original forecasting code
# ------------------------------------------------------------

# Variables excluded because their histories are unbalanced.
unbalanced_predictors = [
    "ACOGNO",
    "ANDENOx",
    "TWEXAFEGSMTHx",
    "UMCSENTx",
    "VIXCLSx",
]

# Reserve variables explicitly removed in the original code.
reserve_predictors = [
    "NONBORRES",
    "TOTRESNS",
]

excluded_predictors = (
    unbalanced_predictors
    + reserve_predictors
)

X_transformed = X_transformed.drop(
    columns=[
        col
        for col in excluded_predictors
        if col in X_transformed.columns
    ]
)


# ------------------------------------------------------------
# Construct unemployment target
# ------------------------------------------------------------

target = "y_unrate_change_1m_ahead"

# UNRATE has transformation code 2 in FRED-MD.
# For a one-month horizon, the original code therefore uses
# the next month's first difference in UNRATE:
#
#     UNRATE_{t+1} - UNRATE_t
#
# This is constructed directly from the raw UNRATE level here.
y = df["UNRATE"].shift(-1) - df["UNRATE"]


# ------------------------------------------------------------
# Combine predictors and target
# ------------------------------------------------------------

replication_data = X_transformed.copy()
replication_data[target] = y


# ------------------------------------------------------------
# Match usable sample start
# ------------------------------------------------------------

# The original forecasting-data generation code begins the
# usable sample in 1960.
replication_data = replication_data.loc[
    "1960-01-01":
].copy()


# ------------------------------------------------------------
# Handle remaining missing observations
# Matches code_for_paper/format_data.py
# ------------------------------------------------------------

# Original code first treats positive/negative infinity as missing.
replication_data = replication_data.replace(
    [np.inf, -np.inf],
    np.nan,
)


# Diagnostic: report what will be removed.
missing_by_variable = replication_data.isna().sum()
missing_by_variable = missing_by_variable[
    missing_by_variable > 0
]

rows_with_missing = replication_data.isna().any(axis=1)

print("\nMissing values before dropping incomplete rows:")
print(missing_by_variable)

print(
    "Total missing values:",
    int(missing_by_variable.sum()),
)

print(
    "Rows containing at least one missing value:",
    int(rows_with_missing.sum()),
)

print("\nDates dropped because of missing values:")
print(
    replication_data.index[
        rows_with_missing
    ]
)


# The original format_data.py uses rs.dropna().
# No forward filling or backward filling is performed.
replication_data = replication_data.dropna().copy()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

replication_data.to_csv(
    "data/replication_dataset.csv"
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nDataset construction complete.")

print(
    f"Replication dataset shape: "
    f"{replication_data.shape}"
)

print(
    f"Date range: "
    f"{replication_data.index.min()} "
    f"to {replication_data.index.max()}"
)

print(
    "Remaining missing values:",
    int(replication_data.isna().sum().sum()),
)

print(
    "Saved data/replication_dataset.csv"
)