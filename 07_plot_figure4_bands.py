from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_PREFIX = "dnn"
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

VALIDATION_START = pd.Timestamp("1980-01-01")
TEST_START = pd.Timestamp("2000-01-01")
PLOT_END = pd.Timestamp("2024-01-01")

# The second figure keeps the full date range but clips extreme y-values.
ZOOM_Y_MIN = -2.0
ZOOM_Y_MAX = 2.0

TARGET_LABEL = "Monthly Change in Unemployment Rate (percentage points)"

NBER_RECESSIONS = [
    (pd.Timestamp("1980-01-01"), pd.Timestamp("1980-07-01")),
    (pd.Timestamp("1981-07-01"), pd.Timestamp("1982-11-01")),
    (pd.Timestamp("1990-07-01"), pd.Timestamp("1991-03-01")),
    (pd.Timestamp("2001-03-01"), pd.Timestamp("2001-11-01")),
    (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01")),
    (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
]


def read_forecast_file(file_path: Path, quantile: float):
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, parse_dates=["date"])

    standardized_quantile = f"q{quantile:.2f}"
    raw_quantile = f"q{quantile:.2f}_raw"

    # The training scripts save standardized values in qXX/actual and
    # percentage-point values in qXX_raw/actual_raw. Plot the raw values.
    if raw_quantile in df.columns and "actual_raw" in df.columns:
        result = df[["date", raw_quantile, "actual_raw"]].copy()
        result = result.rename(
            columns={
                raw_quantile: standardized_quantile,
                "actual_raw": "actual",
            }
        )
        return result

    # Backward-compatible fallback for older forecast files that contain
    # only raw values under the original column names.
    required_columns = {"date", standardized_quantile, "actual"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path} is missing raw forecast columns "
            f"({raw_quantile}, actual_raw) and fallback columns: "
            f"{sorted(missing_columns)}"
        )

    print(
        f"Warning: {file_path.name} has no *_raw columns; "
        "using qXX and actual as stored."
    )
    return df[["date", standardized_quantile, "actual"]].copy()


def load_one_quantile(quantile: float) -> pd.DataFrame:
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
            ignore_index=True,
        )
    else:
        combined = test_df
        print(
            f"Validation forecasts not found for q={quantile:.2f}; "
            "plotting test period only."
        )

    return (
        combined
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_all_forecasts() -> pd.DataFrame:
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
                how="inner",
            )

    merged = merged[
        (merged["date"] >= VALIDATION_START)
        & (merged["date"] <= PLOT_END)
    ].copy()

    merged = merged.sort_values("date").reset_index(drop=True)

    if merged.empty:
        raise ValueError("No forecast observations available for plotting.")

    return merged


def format_x_axis(
    ax,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    span_years = max(1, end_date.year - start_date.year)

    if span_years <= 8:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(base=2))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)


def shade_recessions(
    ax,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    label_used = False

    for recession_start, recession_end in NBER_RECESSIONS:
        if recession_end < start_date or recession_start > end_date:
            continue

        ax.axvspan(
            max(recession_start, start_date),
            min(recession_end, end_date),
            color="gray",
            alpha=0.18,
            linewidth=0,
            label="NBER recession" if not label_used else None,
        )
        label_used = True


def make_plot(
    df: pd.DataFrame,
    title: str,
    output_name: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    plot_df = df.copy()

    if start_date is not None:
        plot_df = plot_df[plot_df["date"] >= start_date]

    if end_date is not None:
        plot_df = plot_df[plot_df["date"] <= end_date]

    if plot_df.empty:
        raise ValueError(
            f"No data available for requested window: "
            f"{start_date} to {end_date}."
        )

    plot_start = plot_df["date"].min()
    plot_end = plot_df["date"].max()

    fig, ax = plt.subplots(figsize=(18, 7))

    shade_recessions(ax, plot_start, plot_end)

    ax.fill_between(
        plot_df["date"],
        plot_df["q0.05"],
        plot_df["q0.95"],
        color="#9ecae1",
        alpha=0.55,
        linewidth=0,
        label="90% prediction interval",
    )

    ax.fill_between(
        plot_df["date"],
        plot_df["q0.25"],
        plot_df["q0.75"],
        color="#3182bd",
        alpha=0.55,
        linewidth=0,
        label="50% prediction interval",
    )

    ax.plot(
        plot_df["date"],
        plot_df["q0.50"],
        color="#08519c",
        linewidth=2.0,
        label="Median forecast",
    )

    ax.plot(
        plot_df["date"],
        plot_df["actual"],
        color="#e67e22",
        linewidth=2.2,
        label="Actual unemployment change",
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.45,
    )

    if plot_start < TEST_START <= plot_end:
        ax.axvline(
            TEST_START,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
        )
        ax.text(
            TEST_START,
            0.98,
            "Test period",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=10,
        )

    ax.set_xlim(plot_start, plot_end)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(TARGET_LABEL, fontsize=12)

    format_x_axis(ax, plot_start, plot_end)
    ax.grid(True, alpha=0.18, linewidth=0.8)

    handles, labels = ax.get_legend_handles_labels()
    preferred_order = [
        "Actual unemployment change",
        "Median forecast",
        "50% prediction interval",
        "90% prediction interval",
        "NBER recession",
    ]

    ordered = [
        (handles[labels.index(label)], label)
        for label in preferred_order
        if label in labels
    ]

    ax.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    plt.tight_layout()

    png_file = FIGURES_DIR / f"{output_name}.png"
    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved PNG: {png_file}")


def main() -> None:
    forecasts = load_all_forecasts()

    print(
        f"\nPlotting {len(forecasts)} observations "
        f"from {forecasts['date'].min():%Y-%m} "
        f"to {forecasts['date'].max():%Y-%m}."
    )

    make_plot(
        df=forecasts,
        title="Deep Neural Network Density Forecasts",
        output_name="dnn_figure4_full",
    )

    make_plot(
        df=forecasts,
        title=(
            "Deep Neural Network Density Forecasts — "
            "Full Sample, Zoomed Y-Axis"
        ),
        output_name="dnn_figure4_zoomed",
        y_limits=(ZOOM_Y_MIN, ZOOM_Y_MAX),
    )


if __name__ == "__main__":
    main()