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

# Dates
df["sasdate"] = pd.to_datetime(df["sasdate"])
df = df.set_index("sasdate")

# Convert data to numeric
df = df.apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------
# FRED-MD transformation function
# Mirrors prepare_missing.m / transxf()
# ------------------------------------------------------------

def fred_transform(x, tcode):
    """
    Apply FRED-MD transformation code to one pandas Series.
    Codes follow the FRED-MD appendix and MATLAB prepare_missing.m.
    """

    x = x.astype(float)
    small = 1e-6

    if pd.isna(tcode):
        return x * np.nan

    tcode = int(tcode)

    if tcode == 1:
        # Level: x_t
        return x

    elif tcode == 2:
        # First difference: x_t - x_{t-1}
        return x.diff()

    elif tcode == 3:
        # Second difference
        return x.diff().diff()

    elif tcode == 4:
        # Natural log
        if x.min(skipna=True) < small:
            return x * np.nan
        return np.log(x)

    elif tcode == 5:
        # First difference of natural log
        if x.min(skipna=True) <= small:
            return x * np.nan
        return np.log(x).diff()

    elif tcode == 6:
        # Second difference of natural log
        if x.min(skipna=True) <= small:
            return x * np.nan
        return np.log(x).diff().diff()

    elif tcode == 7:
        # First difference of percent change
        pct_change = (x - x.shift(1)) / x.shift(1)
        return pct_change.diff()

    else:
        raise ValueError(f"Unknown transformation code: {tcode}")

def remove_outliers_fred_md(X):
    """
    Official FRED-MD style outlier removal:
    replace x with NaN if |x - median| > 10 * IQR.
    """
    X_clean = X.copy()

    for col in X_clean.columns:
        series = X_clean[col]

        median = series.median(skipna=True)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        outlier_mask = (series - median).abs() > 10 * iqr
        X_clean.loc[outlier_mask, col] = np.nan

    return X_clean

# ------------------------------------------------------------
# Apply transformations to predictors
# ------------------------------------------------------------

X_transformed = pd.DataFrame(index=df.index)

for col in df.columns:
    tcode = transform_row[col]
    X_transformed[col] = fred_transform(df[col], tcode)


# MATLAB code removes first two months after transformations
X_transformed = X_transformed.iloc[2:].copy()
df = df.iloc[2:].copy()
X_transformed = remove_outliers_fred_md(X_transformed)

# ------------------------------------------------------------
# Construct target variable from raw UNRATE level
# ------------------------------------------------------------

target = "y_unrate_change_1m_ahead"

# Forecast change in unemployment rate one month ahead:
# UNRATE_{t+1} - UNRATE_t
y = df["UNRATE"].shift(-1) - df["UNRATE"]


# ------------------------------------------------------------
# Combine predictors and target
# ------------------------------------------------------------

replication_data = X_transformed.copy()
replication_data[target] = y

# Drop rows where target is missing
replication_data = replication_data.dropna(subset=[target])

# Fill missing predictor values
predictor_cols = replication_data.columns.drop(target)
replication_data[predictor_cols] = (
    replication_data[predictor_cols]
    .ffill()
    .bfill()
)

# Save
replication_data.to_csv("data/replication_dataset.csv")

print("Dataset construction complete.")
print(f"Replication dataset shape: {replication_data.shape}")
print(f"Date range: {replication_data.index.min()} to {replication_data.index.max()}")
print("Saved data/replication_dataset.csv")