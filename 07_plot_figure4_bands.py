from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Use "dnn" or "linear_activation"
MODEL_PREFIX = "dnn"

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

VALIDATION_START = pd.Timestamp("1980-01-01")
TEST_START = pd.Timestamp("2000-01-01")
PLOT_END = pd.Timestamp("2024-01-01")

# Replace this with the exact target name when you are ready.
TARGET_LABEL = "Unemployment Rate (%)"

# Zoomed view limits
ZOOM_Y_MIN = -1.0
ZOOM_Y_MAX = 1.0


# ------------------------------------------------------------
# Load forecast files
# ------------------------------------------------------------

def read_forecast_file(file_path, quantile):
    """
    Read one quantile forecast file.
    """
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, parse_dates=["date"])
    quantile_column = f"q{quantile:.2f}"

    required_columns = {"date", quantile_column, "actual"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path} is missing columns: {sorted(missing_columns)}"
        )

    return df[["date", quantile_column, "actual"]].copy()


def load_one_quantile(quantile):
    """
    Load validation and test forecasts for one quantile.
    """
    validation_file = (
        RESULTS_DIR
        / f"{MODEL_PREFIX}_q{quantile:.2f}_validation_forecasts.csv"
    )

    test_file = (
        RESULTS_DIR
        / f"{MODEL_PREFIX}_q{quantile:.2f}_test_forecasts.csv"
    )

    validation_df = read_forecast_file(validation_file, quantile)
    test_df = read_forecast_file(test_file, quantile)

    if test_df is None:
        raise FileNotFoundError(
            f"Missing required test forecast file:\n{test_file}"
        )

    if validation_df is not None:
        combined = pd.concat(
            [validation_df, test_df],
            ignore_index=True
        )
    else:
        combined = test_df
        print(
            f"Validation forecasts not found for q={quantile:.2f}; "
            "plotting test period only."
        )

    combined = (
        combined
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return combined


def load_all_forecasts():
    """
    Merge all required quantile forecast files into one dataframe.
    """
    merged = None

    for quantile in QUANTILES:
        quantile_df = load_one_quantile(quantile)
        quantile_column = f"q{quantile:.2f}"

        if merged is None:
            merged = quantile_df
        else:
            merged = merged.merge(
                quantile_df[["date", quantile_column]],
                on="date",
                how="inner"
            )

    merged = merged[
        (merged["date"] >= VALIDATION_START)
        & (merged["date"] <= PLOT_END)
    ].copy()

    merged = merged.sort_values("date").reset_index(drop=True)

    if merged.empty:
        raise ValueError("No forecast observations available for plotting.")

    return merged


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def format_x_axis(ax):
    """
    Format the date axis with readable two-year labels.
    """
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)


def make_plot(df, title, output_name, y_limits=None):
    """
    Create and save one Figure 4-style forecast plot.
    """
    fig, ax = plt.subplots(figsize=(18, 7))

        # 90% predictive interval: light blue
    ax.fill_between(
        df["date"],
        df["q0.05"],
        df["q0.95"],
        color="#9ecae1",
        alpha=0.55,
        linewidth=0,
        label="90% prediction interval"
    )

    # 50% predictive interval: darker blue
    ax.fill_between(
        df["date"],
        df["q0.25"],
        df["q0.75"],
        color="#3182bd",
        alpha=0.55,
        linewidth=0,
        label="50% prediction interval"
    )

    # Median forecast: dark blue
    ax.plot(
        df["date"],
        df["q0.50"],
        color="#08519c",
        linewidth=2.0,
        label="Median forecast"
    )

    # Actual unemployment: orange
    ax.plot(
        df["date"],
        df["actual"],
        color="#e67e22",
        linewidth=2.2,
        label="Actual unemployment"
    )

    # Zero line
    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.45
    )

    # Mark beginning of test period if validation data are included
    if df["date"].min() < TEST_START:
        ax.axvline(
            TEST_START,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75
        )

        ax.text(
            TEST_START,
            0.98,
            "Test period",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=10
        )

    if y_limits is not None:
        ax.set_ylim(y_limits)

    ax.set_xlim(df["date"].min(), df["date"].max())

    ax.set_title(
        title,
        fontsize=16,
        pad=14
    )

    ax.set_xlabel(
        "Date",
        fontsize=12
    )

    ax.set_ylabel(
        TARGET_LABEL,
        fontsize=12
    )

    format_x_axis(ax)

    ax.grid(
        True,
        alpha=0.18,
        linewidth=0.8
    )

    handles, labels = ax.get_legend_handles_labels()

    order = [3, 2, 1, 0]

    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        frameon=True,
        framealpha=0.95
    )

    plt.tight_layout()

    png_file = FIGURES_DIR / f"{output_name}.png"
    pdf_file = FIGURES_DIR / f"{output_name}.pdf"

    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print(f"Saved PNG: {png_file}")
    print(f"Saved PDF: {pdf_file}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    forecasts = load_all_forecasts()

    print(
        f"\nPlotting {len(forecasts)} observations "
        f"from {forecasts['date'].min():%Y-%m} "
        f"to {forecasts['date'].max():%Y-%m}."
    )


    make_plot(
        df=forecasts,
        title="Recursive Density Forecasts of U.S. Unemployment — Zoomed View",
        output_name=f"{MODEL_PREFIX}_figure4_zoomed",
        y_limits=(ZOOM_Y_MIN, ZOOM_Y_MAX)
    )


if __name__ == "__main__":
    main()