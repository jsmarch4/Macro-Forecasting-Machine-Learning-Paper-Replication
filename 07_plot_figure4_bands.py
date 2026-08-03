from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
DNN_DIR = RESULTS_DIR / "dnn"

VALIDATION_FORECAST_DIR = DNN_DIR / "validation" / "forecasts"
TEST_FORECAST_DIR = DNN_DIR / "test" / "forecasts"

FIGURES_DIR = Path("figures") / "dnn"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

VALIDATION_START = pd.Timestamp("1980-01-01")
TEST_START = pd.Timestamp("2000-01-01")
PLOT_END = pd.Timestamp("2024-01-01")

ZOOM_Y_MIN = -2.0
ZOOM_Y_MAX = 2.0

TARGET_LABEL = (
    "Monthly Change in Unemployment Rate "
    "(percentage points)"
)

NBER_RECESSIONS = [
    (pd.Timestamp("1980-01-01"), pd.Timestamp("1980-07-01")),
    (pd.Timestamp("1981-07-01"), pd.Timestamp("1982-11-01")),
    (pd.Timestamp("1990-07-01"), pd.Timestamp("1991-03-01")),
    (pd.Timestamp("2001-03-01"), pd.Timestamp("2001-11-01")),
    (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01")),
    (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01")),
]


def read_forecast_file(
    file_path: Path,
    quantile: float,
) -> pd.DataFrame | None:
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path, parse_dates=["date"])

    standard_column = f"q{quantile:.2f}"
    raw_column = f"q{quantile:.2f}_raw"

    if raw_column in df.columns and "actual_raw" in df.columns:
        result = df[["date", raw_column, "actual_raw"]].copy()
        return result.rename(
            columns={
                raw_column: standard_column,
                "actual_raw": "actual",
            }
        )

    required = {"date", standard_column, "actual"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{file_path} is missing raw columns "
            f"({raw_column}, actual_raw) and fallback columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Warning: {file_path.name} has no *_raw columns; "
        "using qXX and actual as stored."
    )
    return df[["date", standard_column, "actual"]].copy()


def load_one_quantile(quantile: float) -> pd.DataFrame:
    validation_file = (
        VALIDATION_FORECAST_DIR / f"q{quantile:.2f}.csv"
    )
    test_file = TEST_FORECAST_DIR / f"q{quantile:.2f}.csv"

    validation_df = read_forecast_file(
        validation_file,
        quantile,
    )
    test_df = read_forecast_file(
        test_file,
        quantile,
    )

    if test_df is None:
        raise FileNotFoundError(
            f"Missing required test forecast file: {test_file}"
        )

    if validation_df is None:
        print(
            f"Validation forecasts not found for "
            f"q={quantile:.2f}; plotting test period only."
        )
        combined = test_df
    else:
        combined = pd.concat(
            [validation_df, test_df],
            ignore_index=True,
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

    if merged is None:
        raise ValueError("No forecast files were loaded.")

    merged = merged[
        (merged["date"] >= VALIDATION_START)
        & (merged["date"] <= PLOT_END)
    ].copy()

    merged = merged.sort_values("date").reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "No forecast observations available for plotting."
        )

    return merged


def format_x_axis(
    axis: plt.Axes,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    span_years = max(1, end_date.year - start_date.year)

    if span_years <= 8:
        axis.xaxis.set_major_locator(mdates.YearLocator(base=1))
    else:
        axis.xaxis.set_major_locator(mdates.YearLocator(base=2))

    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )
    axis.tick_params(axis="x", labelsize=10)
    axis.tick_params(axis="y", labelsize=10)


def shade_recessions(
    axis: plt.Axes,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    label_used = False

    for recession_start, recession_end in NBER_RECESSIONS:
        if recession_end < start_date or recession_start > end_date:
            continue

        axis.axvspan(
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
    y_limits: tuple[float, float] | None = None,
) -> None:
    plot_df = df.copy()

    if plot_df.empty:
        raise ValueError("No data available for plotting.")

    plot_start = plot_df["date"].min()
    plot_end = plot_df["date"].max()

    figure, axis = plt.subplots(figsize=(18, 7))

    shade_recessions(axis, plot_start, plot_end)

    axis.fill_between(
        plot_df["date"],
        plot_df["q0.05"],
        plot_df["q0.95"],
        color="#9ecae1",
        alpha=0.55,
        linewidth=0,
        label="90% prediction interval",
    )

    axis.fill_between(
        plot_df["date"],
        plot_df["q0.25"],
        plot_df["q0.75"],
        color="#3182bd",
        alpha=0.55,
        linewidth=0,
        label="50% prediction interval",
    )

    axis.plot(
        plot_df["date"],
        plot_df["q0.50"],
        color="#08519c",
        linewidth=2.0,
        label="Median forecast",
    )

    axis.plot(
        plot_df["date"],
        plot_df["actual"],
        color="#e67e22",
        linewidth=2.2,
        label="Actual unemployment change",
    )

    axis.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.45,
    )

    if plot_start < TEST_START <= plot_end:
        axis.axvline(
            TEST_START,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
        )
        axis.text(
            TEST_START,
            0.98,
            "Test period",
            transform=axis.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=10,
        )

    axis.set_xlim(plot_start, plot_end)

    if y_limits is not None:
        axis.set_ylim(*y_limits)

    axis.set_title(title, fontsize=16, pad=14)
    axis.set_xlabel("Date", fontsize=12)
    axis.set_ylabel(TARGET_LABEL, fontsize=12)

    format_x_axis(axis, plot_start, plot_end)
    axis.grid(True, alpha=0.18, linewidth=0.8)

    handles, labels = axis.get_legend_handles_labels()
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

    axis.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    figure.tight_layout()

    output_path = FIGURES_DIR / f"{output_name}.png"
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved PNG: {output_path}")


def main() -> None:
    forecasts = load_all_forecasts()

    print(
        f"\nPlotting {len(forecasts)} observations "
        f"from {forecasts['date'].min():%Y-%m} "
        f"to {forecasts['date'].max():%Y-%m}."
    )

    make_plot(
        forecasts,
        "Deep Neural Network Density Forecasts",
        "figure4_full",
    )

    make_plot(
        forecasts,
        (
            "Deep Neural Network Density Forecasts — "
            "Full Sample, Zoomed Y-Axis"
        ),
        "figure4_zoomed",
        y_limits=(ZOOM_Y_MIN, ZOOM_Y_MAX),
    )


if __name__ == "__main__":
    main()