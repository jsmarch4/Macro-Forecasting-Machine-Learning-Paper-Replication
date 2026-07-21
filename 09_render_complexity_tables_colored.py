"""
09_render_complexity_tables_colored.py

Render the four complexity-by-quantile tables produced by
08_complexity_tables.py as publication-style PNG images.

The cell coloring follows the paper:

- Negative loss differentials are green.
- Positive loss differentials are red.
- Color darkness depends on |estimate / HAC standard error|:
    light:  z < 1.28
    medium: 1.28 <= z < 1.65
    dark:   z >= 1.65

The r = 0 row is the normalized naive benchmark and is left unshaded.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
TABLE_DIR = RESULTS_DIR / "complexity" / "tables"
FIGURE_DIR = RESULTS_DIR / "complexity" / "table_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TABLE_SPECS = [
    {
        "family": "linear_activation",
        "sample": "validation",
        "title": "A. Linear Activation Network — Validation",
    },
    {
        "family": "linear_activation",
        "sample": "test",
        "title": "B. Linear Activation Network — Test",
    },
    {
        "family": "dnn",
        "sample": "validation",
        "title": "C. Deep Neural Network — Validation",
    },
    {
        "family": "dnn",
        "sample": "test",
        "title": "D. Deep Neural Network — Test",
    },
]

GREEN_LIGHT = "#E8F3E8"
GREEN_MEDIUM = "#B7DDB7"
GREEN_DARK = "#75B875"

RED_LIGHT = "#F8E8E8"
RED_MEDIUM = "#EDB6B6"
RED_DARK = "#D97979"

BENCHMARK_FILL = "#F1F1F1"
HEADER_FILL = "#FFFFFF"


def parse_cell(cell):
    if pd.isna(cell):
        return float("nan"), float("nan")

    numbers = re.findall(r"[-+]?\d*\.?\d+", str(cell))

    if not numbers:
        return float("nan"), float("nan")

    estimate = float(numbers[0])
    standard_error = float(numbers[1]) if len(numbers) > 1 else float("nan")

    return estimate, standard_error


def load_formatted_table(family, sample):
    path = TABLE_DIR / f"table_{sample}_{family}_formatted.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing table file: {path}\n"
            "Run 08_complexity_tables.py first."
        )

    df = pd.read_csv(path)

    for column in ["family", "sample"]:
        if column in df.columns:
            df = df.drop(columns=column)

    if "complexity" not in df.columns:
        raise ValueError(f"{path} does not contain a complexity column.")

    return df


def quantile_label(column):
    if column == "complexity":
        return "Complexity"

    return rf"$\tau={float(column):.2f}$"


def paper_cell_color(estimate, standard_error):
    if pd.isna(estimate):
        return HEADER_FILL

    if pd.isna(standard_error) or standard_error <= 0:
        z_score = 0.0
    else:
        z_score = abs(estimate / standard_error)

    if estimate < 0:
        if z_score >= 1.65:
            return GREEN_DARK
        if z_score >= 1.28:
            return GREEN_MEDIUM
        return GREEN_LIGHT

    if estimate > 0:
        if z_score >= 1.65:
            return RED_DARK
        if z_score >= 1.28:
            return RED_MEDIUM
        return RED_LIGHT

    return HEADER_FILL


def render_table(ax, df, panel_title):
    ax.axis("off")

    display_df = df.copy()
    display_df["complexity"] = display_df["complexity"].map(
        lambda value: f"{float(value):.1f}"
    )

    columns = list(display_df.columns)
    quantile_columns = [column for column in columns if column != "complexity"]

    table = ax.table(
        cellText=display_df.astype(str).values.tolist(),
        colLabels=[quantile_label(column) for column in columns],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.115] + [0.885 / len(quantile_columns)] * len(quantile_columns),
        edges="closed",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.72)

    n_rows = len(display_df)
    n_cols = len(columns)

    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.visible_edges = "BRTL"
        cell.PAD = 0.028

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.0)

    for column in range(n_cols):
        benchmark_cell = table[(1, column)]
        benchmark_cell.set_facecolor(BENCHMARK_FILL)
        benchmark_cell.visible_edges = "B"
        benchmark_cell.set_linewidth(0.7)
        benchmark_cell.set_text_props(weight="medium")

    for column in range(n_cols):
        bottom_cell = table[(n_rows, column)]
        bottom_cell.visible_edges = "B"
        bottom_cell.set_linewidth(1.0)

    for table_row in range(2, n_rows + 1):
        source_row = table_row - 1
        table[(table_row, 0)].set_text_props(weight="medium")

        for column_index, column_name in enumerate(quantile_columns, start=1):
            estimate, standard_error = parse_cell(
                display_df.iloc[source_row][column_name]
            )

            table[(table_row, column_index)].set_facecolor(
                paper_cell_color(estimate, standard_error)
            )

    ax.text(
        0.5,
        1.045,
        panel_title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def add_figure_notes(fig):
    fig.text(
        0.5,
        0.024,
        (
            "Notes: At complexity 0.0, entries are normalized losses for the recursively "
            "estimated unconditional-quantile benchmark. For complexity above 0.0, entries "
            "are normalized model-minus-benchmark loss differentials, with HAC standard "
            "errors in parentheses. Negative values (green) indicate that the model "
            "outperforms the benchmark; positive values (red) indicate underperformance. "
            "Shading darkness is based on |estimate / standard error|: light below 1.28, "
            "medium from 1.28 to 1.65, and dark above 1.65."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )


def save_sample_comparison(sample, table_number):
    specs = [spec for spec in TABLE_SPECS if spec["sample"] == sample]

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.8))

    for ax, spec in zip(axes, specs):
        df = load_formatted_table(spec["family"], spec["sample"])
        render_table(ax, df, spec["title"])

    sample_title = (
        "Validation Sample (1980–1999)"
        if sample == "validation"
        else "Test Sample (2000–2024)"
    )

    fig.suptitle(
        (
            f"Table {table_number}. Accuracy of the Forecast of the Monthly Change "
            f"in the Unemployment Rate One Month Ahead\n{sample_title}"
        ),
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )

    add_figure_notes(fig)

    fig.tight_layout(
        rect=[0.018, 0.08, 0.982, 0.955],
        h_pad=2.1,
    )

    output_path = FIGURE_DIR / f"table_{table_number}_{sample}_linear_vs_dnn.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    print(f"Saved: {output_path}")


def save_combined_four_panel():
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 12.2))

    panel_titles = [
        "A. Linear Activation Network — Validation",
        "B. Linear Activation Network — Test",
        "C. Deep Neural Network — Validation",
        "D. Deep Neural Network — Test",
    ]

    for ax, spec, title in zip(axes.flat, TABLE_SPECS, panel_titles):
        df = load_formatted_table(spec["family"], spec["sample"])
        render_table(ax, df, title)

    fig.suptitle(
        (
            "Forecast Accuracy by Complexity and Quantile\n"
            "Validation on the Left, Test on the Right"
        ),
        fontsize=15,
        fontweight="bold",
        y=0.993,
    )

    add_figure_notes(fig)

    fig.tight_layout(
        rect=[0.012, 0.075, 0.988, 0.958],
        h_pad=2.0,
        w_pad=1.0,
    )

    output_path = (
        FIGURE_DIR / "complexity_tables_linear_vs_dnn_four_panel.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    save_sample_comparison("validation", 2)
    save_sample_comparison("test", 3)
    save_combined_four_panel()

    print(f"\nAll PNG table images saved in:\n{FIGURE_DIR}")


if __name__ == "__main__":
    main()